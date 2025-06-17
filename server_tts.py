import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote
from fastapi.middleware.cors import CORSMiddleware
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import Response
from huggingface_hub import hf_hub_download
from pydantic import BaseModel

# style-bert-vits2 の関連モジュールをインポート
from style_bert_vits2.tts_model import TTSModel

# bert_models はモデルロード時に内部で必要になる可能性があるためインポートしておく
try:
    from style_bert_vits2.nlp import bert_models
except ImportError:
    print("Warning: style_bert_vits2.nlp.bert_models not found. BERT preloading might be skipped if not done internally by TTSModel.")
    bert_models = None # フォールバック

# 定数をインポート (存在しない場合はフォールバック値を定義)
try:
    from style_bert_vits2.constants import (
        DEFAULT_ASSIST_TEXT_WEIGHT, DEFAULT_LENGTH, DEFAULT_LINE_SPLIT,
        DEFAULT_NOISE, DEFAULT_NOISEW, DEFAULT_SDP_RATIO,
        DEFAULT_SPLIT_INTERVAL, DEFAULT_STYLE, DEFAULT_STYLE_WEIGHT, Languages)
    # このモデルは日本語なのでデフォルト言語を JP に設定
    ln = Languages.JP
except ImportError:
    print("Warning: style_bert_vits2.constants not found. Using fallback default values.")
    DEFAULT_SDP_RATIO = 0.2
    DEFAULT_NOISE = 0.6
    DEFAULT_NOISEW = 0.8
    DEFAULT_LENGTH = 1.0
    DEFAULT_LINE_SPLIT = False
    DEFAULT_SPLIT_INTERVAL = 0.5
    DEFAULT_ASSIST_TEXT_WEIGHT = 0.7
    DEFAULT_STYLE = "Neutral"  # モデルに存在するスタイルか確認が必要
    DEFAULT_STYLE_WEIGHT = 7.0
    class Languages: # Enum の代わり
        JP = "JP"
        EN = "EN"
        ZH = "ZH"
    ln = Languages.JP

# --- ロガー設定 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI アプリケーション初期化 ---
app = FastAPI(
    title="Style-BERT-VITS2 API (Custom)",
    description="A simplified API server for Style-BERT-VITS2 based on a single preloaded model.",
    version="0.1.0",
)

# --- CORS 設定 ---
# 許可するオリジンのリスト
# Next.js の開発サーバーのオリジンを追加します
# 必要に応じて本番環境のオリジンも追加してください
origins = [
    "http://localhost:3000", # Next.js のデフォルト開発サーバー
    # "https://your-production-nextjs-app.com", # 本番環境のドメイン例
]

# CORSミドルウェアを追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 許可するオリジンのリストを指定
    allow_credentials=True, # クレデンシャル（Cookieなど）を許可するかどうか
    allow_methods=["*"],    # すべてのHTTPメソッド（GET, POSTなど）を許可
    allow_headers=["*"],    # すべてのHTTPヘッダーを許可
)

# --- デバイス設定 ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

# --- BERTモデルのロード (公式実装に合わせて事前にロード) ---
# ユーザー指定のモデルを使う
bert_model_name = "ku-nlp/deberta-v2-large-japanese-char-wwm"
logger.info(f"Loading BERT model and tokenizer: {bert_model_name} for language {Languages.JP}...")
try:
    if bert_models:
        bert_models.load_model(Languages.JP, bert_model_name)
        bert_models.load_tokenizer(Languages.JP, bert_model_name)
        logger.info("BERT model and tokenizer loaded successfully.")
    else:
        logger.warning("bert_models module not available, skipping explicit BERT preloading.")
except Exception as e:
    logger.error(f"Error loading BERT model/tokenizer: {e}", exc_info=True)
    # BERTがロードできなくてもTTSモデルによっては動作する可能性があるので継続

# --- TTSモデルのダウンロードとロード ---
model_repo = "litagin/style_bert_vits2_jvnv"
model_dir_name = "jvnv-F1-jp"
model_file = f"{model_dir_name}/jvnv-F1-jp_e160_s14000.safetensors"
config_file = f"{model_dir_name}/config.json"
style_file = f"{model_dir_name}/style_vectors.npy"
assets_root = Path("models")
assets_root.mkdir(parents=True, exist_ok=True) # ディレクトリ作成

logger.info("Downloading TTS model files...")
try:
    for file in [model_file, config_file, style_file]:
        logger.info(f"Checking/Downloading {file} from {model_repo}...")
        hf_hub_download(model_repo, file, local_dir=assets_root, local_dir_use_symlinks=False) # シンボリックリンクを避ける
    logger.info("TTS model files download complete.")
except Exception as e:
    logger.error(f"Error downloading TTS model files: {e}", exc_info=True)
    # ダウンロード失敗は致命的なので終了
    import sys
    sys.exit(1)

# --- カスタムモデルのロード ---
model_dir_name = "tarte"
model_file = f"{model_dir_name}/tart_e100_s2100.safetensors"
config_file = f"{model_dir_name}/config.json"
style_file = f"{model_dir_name}/style_vectors.npy"
assets_root = Path("models")
assets_root.mkdir(parents=True, exist_ok=True) # ディレクトリ作成


logger.info("Loading TTS model...")
try:
    model = TTSModel(
        model_path=assets_root / model_file,
        config_path=assets_root / config_file,
        style_vec_path=assets_root / style_file,
        device=device,
    )
    # モデル情報のログ出力
    logger.info(f"TTS model loaded successfully.")
    logger.info(f"  Model path: {model.model_path}")
    logger.info(f"  Config path: {model.config_path}")
    logger.info(f"  Style vector path: {model.style_vec_path}")
    logger.info(f"  Available speakers: {model.spk2id}")
    logger.info(f"  Available styles: {model.style2id}")

    # デフォルトスタイルの存在確認と調整
    if DEFAULT_STYLE not in model.style2id:
        available_styles = list(model.style2id.keys())
        if available_styles:
            original_default = DEFAULT_STYLE
            DEFAULT_STYLE = available_styles[0]
            logger.warning(f"Default style '{original_default}' not found in model. Using first available style '{DEFAULT_STYLE}' as default.")
        else:
            logger.error("No styles found in the loaded model. Default style cannot be set.")
            DEFAULT_STYLE = None # スタイル指定なしで動作させるか、エラーにする

except FileNotFoundError as e:
    logger.error(f"Error loading TTS model: {e}. Ensure model files exist in {assets_root / model_dir_name}", exc_info=True)
    import sys
    sys.exit(1)
except Exception as e:
    logger.error(f"An unexpected error occurred while loading the TTS model: {e}", exc_info=True)
    import sys
    sys.exit(1)

# --- ヘルパー関数 ---
def raise_validation_error(msg: str, param: str):
    """Raise FastAPI validation error."""
    logger.warning(f"Validation error: {msg} for parameter '{param}'")
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=[dict(type="invalid_params", msg=msg, loc=["query", param])],
    )

# --- レスポンスクラス ---
class AudioResponse(Response):
    """Custom Response class for audio/wav."""
    media_type = "audio/wav"

    @staticmethod
    def create_wav_response(audio: Any, sr: int) -> "AudioResponse":
        """Generates a WAV response using soundfile."""
        wav_io = BytesIO()
        try:
            sf.write(wav_io, audio, sr, format='WAV', subtype='PCM_16') # subtype指定推奨
            wav_io.seek(0)
            return AudioResponse(content=wav_io.getvalue())
        except Exception as e:
            logger.error(f"Error writing WAV data: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error generating audio response")

# --- API エンドポイント ---

@app.api_route(
    "/voice",
    methods=["GET", "POST"],
    response_class=AudioResponse,
    summary="Generate speech (Official API compatible)",
    description="Generates speech audio from text using the preloaded model. Compatible with official API parameters, but ignores model/speaker selection.",
)
async def voice_compatible(
    request: Request,
    text: str = Query(..., min_length=1, description="Text to synthesize."),
    encoding: Optional[str] = Query(None, description="Decode text with specified encoding (e.g., 'utf-8'). Needed if text is URL-encoded."),
    # --- Ignored parameters (for compatibility) ---
    model_name: Optional[str] = Query(None, description="[Ignored] Model name."),
    model_id: Optional[int] = Query(None, description="[Ignored] Model ID."),
    speaker_name: Optional[str] = Query(None, description="[Ignored] Speaker name. Uses default speaker (ID 0)."),
    speaker_id: Optional[int] = Query(0, description="[Ignored] Speaker ID. Uses default speaker (ID 0)."),
    reference_audio_path: Optional[str] = Query(None, description="[Ignored] Reference audio path for style transfer."),
    # --- Parameters passed to model.infer ---
    sdp_ratio: float = Query(DEFAULT_SDP_RATIO, description="SDP/DP mixing ratio.", ge=0.0, le=1.0),
    noise: float = Query(DEFAULT_NOISE, description="Noise scale.", ge=0.0),
    noisew: float = Query(DEFAULT_NOISEW, description="SDP noise scale.", ge=0.0),
    length: float = Query(DEFAULT_LENGTH, description="Length scale (speech speed).", gt=0.0),
    language: Optional[str] = Query(ln, description=f"Language of the text (e.g., {Languages.JP}, {Languages.EN}, {Languages.ZH}). Defaults to {ln}."),
    auto_split: bool = Query(DEFAULT_LINE_SPLIT, description="Automatically split text by newline characters."),
    split_interval: float = Query(DEFAULT_SPLIT_INTERVAL, description="Silence duration (seconds) between split lines.", ge=0.0),
    assist_text: Optional[str] = Query(None, description="Assistance text for mimicking voice/emotion."),
    assist_text_weight: float = Query(DEFAULT_ASSIST_TEXT_WEIGHT, description="Weight of the assistance text effect.", ge=0.0, le=1.0),
    style: Optional[str] = Query(DEFAULT_STYLE, description=f"Speech style (e.g., {', '.join(model.style2id.keys())}). Defaults to '{DEFAULT_STYLE}'."),
    style_weight: float = Query(DEFAULT_STYLE_WEIGHT, description="Weight of the style effect.", ge=0.0),
):
    """
    Infer text to speech using the preloaded model.
    This endpoint mimics the official API but uses a fixed model and speaker.
    """
    # Log request details (excluding potentially long text)
    query_params_log = {k: v for k, v in request.query_params.items() if k != 'text'}
    logger.info(f"{request.client.host}:{request.client.port} requested /voice with params: {unquote(str(query_params_log))}")
    if request.method == "GET":
        logger.warning("Using GET for /voice is not recommended due to potential URL length limits and encoding issues. Use POST instead.")

    # Decode text if encoding is specified
    decoded_text = text
    if encoding:
        try:
            decoded_text = unquote(text, encoding=encoding)
            logger.info(f"Decoded text using encoding: {encoding}")
        except Exception as e:
            raise_validation_error(f"Failed to decode text with encoding '{encoding}': {e}", "encoding")

    # --- Parameter validation and adjustments for the fixed model ---
    # Use fixed speaker ID (e.g., 0) from the loaded model
    actual_speaker_id = 0
    if actual_speaker_id not in model.id2spk:
        available_ids = list(model.id2spk.keys())
        if not available_ids:
            logger.error("No speakers found in the loaded model.")
            raise HTTPException(status_code=500, detail="Internal Server Error: TTS model has no speakers configured.")
        actual_speaker_id = available_ids[0]
        logger.warning(f"Default speaker ID 0 not found. Using first available speaker ID: {actual_speaker_id}")
    logger.info(f"Using fixed speaker ID: {actual_speaker_id} ('speaker_id' and 'speaker_name' parameters are ignored).")

    # Validate selected style
    actual_style = style if style else DEFAULT_STYLE # Handle None case explicitly
    if actual_style is None: # Case where DEFAULT_STYLE was also None
         logger.warning("No style specified and no default style available. Proceeding without style.")
    elif actual_style not in model.style2id:
        available_styles = list(model.style2id.keys())
        logger.warning(f"Style '{actual_style}' not found. Available styles: {available_styles}. Falling back to default style: '{DEFAULT_STYLE}'.")
        # Fallback to the adjusted DEFAULT_STYLE (which should exist or be None)
        actual_style = DEFAULT_STYLE
        if actual_style is None: # Check again if default is None
             logger.warning("Default style is not available. Proceeding without style.")
        elif actual_style not in model.style2id: # Should not happen if DEFAULT_STYLE logic is correct, but as a safeguard
            logger.error(f"Fallback style '{actual_style}' is also invalid. Cannot proceed.")
            raise HTTPException(status_code=500, detail=f"Internal Server Error: Invalid style configuration.")

    # Language parameter (pass through, model should handle it)
    actual_language = language
    if actual_language is None:
        actual_language = ln # Default if not provided
    # Optional: Add check if model supports the language if necessary

    # Log final parameters used for inference
    logger.info(f"Inference parameters: speaker_id={actual_speaker_id}, language={actual_language}, style='{actual_style}', sdp_ratio={sdp_ratio}, noise={noise}, noisew={noisew}, length={length}, ...")

    # --- Run TTS Inference ---
    try:
        logger.info(f"Starting TTS inference for text: '{decoded_text[:100]}...'")
        sr, audio = model.infer(
            text=decoded_text,
            language=actual_language,
            speaker_id=actual_speaker_id, # Use the fixed speaker ID
            # reference_audio_path is ignored
            sdp_ratio=sdp_ratio,
            noise=noise,
            noise_w=noisew,
            length=length,
            line_split=auto_split,
            split_interval=split_interval,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            use_assist_text=bool(assist_text),
            style=actual_style, # Use the validated/adjusted style
            style_weight=style_weight,
        )
        logger.info("TTS inference completed successfully.")

        # --- Generate and return audio response ---
        return AudioResponse.create_wav_response(audio, sr)

    except HTTPException:
        # Re-raise validation errors
        raise
    except Exception as e:
        logger.error(f"TTS inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: TTS inference failed.")

# --- Original /tts endpoint (optional, keep for simple use cases) ---
class TTSRequest(BaseModel):
    text: str

@app.post(
    "/tts",
    response_class=AudioResponse,
    summary="Generate speech (Simple)",
    description="Generates speech audio from text using default model settings.",
    tags=["Simple Endpoint"], # Optional tag for docs
)
async def generate_tts(request: TTSRequest):
    """Simple TTS endpoint using default parameters."""
    logger.info(f"{request.client.host}:{request.client.port} requested /tts (simple endpoint)")
    try:
        # Use default speaker and style for the simple endpoint
        actual_speaker_id = 0
        if actual_speaker_id not in model.id2spk:
             available_ids = list(model.id2spk.keys())
             if not available_ids: raise HTTPException(status_code=500, detail="TTS model has no speakers.")
             actual_speaker_id = available_ids[0]

        actual_style = DEFAULT_STYLE # Use the potentially adjusted default style
        if actual_style is None and model.style2id: # If default is None but styles exist, pick first
            actual_style = list(model.style2id.keys())[0]
            logger.warning(f"Simple endpoint: No default style, using first available: '{actual_style}'")
        elif actual_style is None:
            logger.warning("Simple endpoint: No style specified or available.")


        logger.info(f"Starting TTS inference for text: '{request.text[:100]}...' (simple endpoint)")
        sr, audio = model.infer(
            text=request.text,
            language=ln, # Default language
            speaker_id=actual_speaker_id, # Default speaker
            style=actual_style, # Default style
            # Other parameters use model defaults
        )
        logger.info("Inference completed successfully (simple endpoint).")

        return AudioResponse.create_wav_response(audio, sr)

    except Exception as e:
        logger.error(f"TTS inference failed (simple endpoint): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: TTS error.")

# --- Main execution block ---
if __name__ == "__main__":
    port = 8003
    host = "0.0.0.0"
    logger.info(f"Starting Style-BERT-VITS2 API server on http://{host}:{port}")
    logger.info(f"Preloaded TTS model: {model_dir_name} ({model_repo})")
    logger.info(f"Using device: {device}")
    logger.info(f"API documentation available at: http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port, log_level="info") # Use info level for uvicorn logs during dev