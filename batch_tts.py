import requests
import os
from io import BytesIO
import soundfile as sf

# TTSサーバーのアドレスとポート
TTS_SERVER_URL = "http://100.89.111.138:8003/voice"  # または "/tts" に切り替え可能

# 保存先のディレクトリ
SAVE_DIR = "save_audio"
os.makedirs(SAVE_DIR, exist_ok=True)

# 生成するテキストのリスト
TEXTS = [
    "うん",
    "はい",
    "ええ",
    "なるほど",
    "ほうほう",
    "ふむふむ",
    "あー",
    "へぇー",
]

def save_wav(audio_bytes: bytes, filename: str):
    """WAVデータをファイルに保存する"""
    try:
        # BytesIOオブジェクトからNumPy配列に変換
        wav_io = BytesIO(audio_bytes)
        data, samplerate = sf.read(wav_io)

        # NumPy配列をWAVファイルとして保存
        sf.write(filename, data, samplerate)
        print(f"Audio saved to: {filename}")
    except Exception as e:
        print(f"Error saving audio: {e}")


def generate_and_save_audio(text: str, file_number: int):
    """TTSサーバーにリクエストを送り、オーディオを保存する"""
    params = {
        "text": text,
    }

    try:
        response = requests.get(TTS_SERVER_URL, params=params)  # または requests.post
        response.raise_for_status()  # HTTPエラーをチェック

        if response.headers.get("Content-Type") == "audio/wav":
            filename = os.path.join(SAVE_DIR, f"{file_number:02d}.wav")
            save_wav(response.content, filename)

        else:
            print(f"Unexpected response type: {response.headers.get('Content-Type')}")
            print(f"Response content: {response.content.decode()}") # エラー内容表示
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")


if __name__ == "__main__":
    for i, text in enumerate(TEXTS):
        generate_and_save_audio(text, i + 1)

    print("All audio files generated and saved.")