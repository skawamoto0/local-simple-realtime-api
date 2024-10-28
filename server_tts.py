from fastapi import FastAPI, HTTPException
from style_bert_vits2.tts_model import TTSModel
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from pathlib import Path
import torch
from fastapi.responses import Response
import soundfile as sf
from io import BytesIO
from huggingface_hub import hf_hub_download
from pydantic import BaseModel

app = FastAPI()

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# bert model の load
bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

# TTS model の download
model_file = "jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors"
config_file = "jvnv-F1-jp/config.json"
style_file = "jvnv-F1-jp/style_vectors.npy"

for file in [model_file, config_file, style_file]:
    print(file)
    hf_hub_download("litagin/style_bert_vits2_jvnv", file, local_dir="models")

# TTS model の load
assets_root = Path("models")
model = TTSModel(
    model_path=assets_root / model_file,
    config_path=assets_root / config_file,
    style_vec_path=assets_root / style_file,
    device=device,
)

class TTSRequest(BaseModel):
    text: str

@app.post("/tts")
async def generate_tts(request: TTSRequest):
    try:
        sr, audio = model.infer(text=request.text)

        # 音声をWAV形式でエンコード
        wav_io = BytesIO()
        sf.write(wav_io, audio, sr, format='WAV')
        wav_io.seek(0)

        # レスポンスとしてWAVファイルを返す
        return Response(content=wav_io.read(), media_type="audio/wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)