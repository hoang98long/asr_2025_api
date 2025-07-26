from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import io
from datetime import datetime

app = FastAPI(title="ASR API - Whisper", description="Nhận dạng giọng nói bằng mô hình Whisper", version="1.0")

# ==== Load model 1 lần khi khởi động ====
model_path = "models/phowhisper-large"
sampling_rate_target = 16000
chunk_length_sec = 30

processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)
model.eval()


@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    try:
        # Đọc file âm thanh
        audio_bytes = await file.read()
        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))

        # Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample nếu cần
        if sr != sampling_rate_target:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate_target)
            waveform = resampler(waveform)

        # Chia chunk
        num_samples = waveform.shape[1]
        samples_per_chunk = chunk_length_sec * sampling_rate_target
        chunks = [
            waveform[:, i:min(i + samples_per_chunk, num_samples)]
            for i in range(0, num_samples, samples_per_chunk)
        ]

        # Nhận dạng từng chunk
        full_transcription = ""
        for idx, chunk in enumerate(chunks):
            inputs = processor(chunk.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                predicted_ids = model.generate(inputs["input_features"])
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            full_transcription += f"[Đoạn {idx+1}]\n{text}\n\n"

        # Tuỳ chọn: Lưu file ra nếu muốn
        # with open("recognized_output_asr.txt", "w", encoding="utf-8") as f:
        #     f.write(full_transcription)

        return JSONResponse({
            "filename": file.filename,
            "timestamp": datetime.now().isoformat(),
            "segments": len(chunks),
            "transcription": full_transcription.strip()
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
