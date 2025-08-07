from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

model_id = "vinai/PhoWhisper-large"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
save_path = "models/phowhisper-large"
processor.save_pretrained(save_path)
model.save_pretrained(save_path)
print(f"Đã tải và lưu model vào: {save_path}")


