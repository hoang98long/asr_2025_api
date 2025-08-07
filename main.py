from fastapi import FastAPI, UploadFile, File
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datetime import datetime
import os
import time
import uuid

app = FastAPI()

MODEL_PATH = "models/phowhisper-large"
CHUNK_LENGTH_SEC = 30
TARGET_SR = 16000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
model.eval()

@app.post("/asr")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4()}.mp3"
    try:
        with open(temp_filename, "wb") as f:
            f.write(await file.read())
        waveform, sr = torchaudio.load(temp_filename)
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sr != TARGET_SR:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)(waveform)

    if waveform.shape[1] < 1000:
        return {
            "error": "Audio quá ngắn để xử lý.",
            "filename": file.filename,
            "timestamp": datetime.now().isoformat()
        }

    samples_per_chunk = CHUNK_LENGTH_SEC * TARGET_SR
    num_samples = waveform.shape[1]
    chunks = [
        waveform[:, start:min(start + samples_per_chunk, num_samples)]
        for start in range(0, num_samples, samples_per_chunk)
    ]

    full_transcription = ""
    chunk_timings = []

    for idx, chunk in enumerate(chunks):
        inputs = processor(
            chunk.squeeze().numpy(),
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            language="vi",
            task="transcribe"
        )

        inputs["input_features"] = inputs["input_features"].to(device)

        with torch.no_grad():
            t0 = time.time()
            predicted_ids = model.generate(
                inputs["input_features"],
                num_beams=5,
                max_new_tokens=440,  # Chọn sao cho không vượt giới hạn 448
                do_sample=False,
                suppress_tokens=[]
            )
            t1 = time.time()

        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        full_transcription += f"{text} "
        chunk_timings.append({"chunk": idx + 1, "duration_sec": round(t1 - t0, 2)})

    return {
        "filename": file.filename,
        "timestamp": datetime.now().isoformat(),
        "num_chunks": len(chunks),
        "chunk_length_sec": CHUNK_LENGTH_SEC,
        "timings": chunk_timings,
        "transcription": full_transcription.strip()
    }

@app.get("/")
def root():
    return {"message": "PhoWhisper ASR API is running."}
