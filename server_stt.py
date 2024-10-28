from fastapi import FastAPI, UploadFile, HTTPException, Request
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import numpy as np
import librosa
import io

app = FastAPI()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# TODO: Modelを動的に選択し、reloadするように修正
model_id = "openai/whisper-large-v3-turbo"
# model_id = "openai/whisper-tiny"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)
processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

def get_generate_kwargs(language: str):
    return {
        "num_beams": 1,
        "return_timestamps": False,
        "language": language,
    }

@app.post("/stt/file")
async def stt_from_file(file: UploadFile, language: str = "japanese"):
    try:
        contents = await file.read()
        audio, sr = librosa.load(io.BytesIO(contents), sr=16000)

        result = pipe(audio, generate_kwargs=get_generate_kwargs(language))
        return {"text": result["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT error: {e}")

@app.post("/stt/bytes")
async def stt_from_bytes(request: Request, language: str = "japanese"):
    try:
        audio_data = await request.body()
        audio = np.frombuffer(audio_data, np.int16).astype(np.float32) / 32768.0
        result = pipe(audio, generate_kwargs=get_generate_kwargs(language))
        return {"text": result["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)