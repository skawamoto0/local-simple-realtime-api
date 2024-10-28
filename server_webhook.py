import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import webrtcvad
import math
import aiohttp
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

class Session:
    def __init__(self, websocket):
        self.websocket = websocket
        self.session_options = None
        self.vad = webrtcvad.Vad(3)
        self.sample_rate = 16000
        self.frame_duration = 20
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        self.frame_bytes = self.frame_size * 2
        self.min_speech_frames = 5
        self.buffer = bytes()
        self.audio_frames = []
        self.silence_counter = 0
        self.silence_duration = 200
        self.silence_frames = math.ceil(self.silence_duration / self.frame_duration)
        self.user_speech_started = False
        self.user_speech_to_assistant_speech_task = None  # stt (audio to text) -> llm (user text to assistant text) -> tts (assistant text to speech)
        self.user_text_to_assistant_speech_task = None  # llm (user text to assistant text) -> tts (assistant text to speech)
        self.stt_model_id = "openai/whisper-large-v3-turbo"
        self.llm_model_id = "google/gemma-2-2b-jpn-it"
        self.tts_model_id = "litagin/style_bert_vits2_jvnv"
        self.messages = []

    async def handle_text_message(self, data):
        try:
            message = json.loads(data)
            event = message.get('event')
            if event == 'startSession':
                self.session_options = message
                self.stt_model_id = message.get('sttModelId')
                self.llm_model_id = message.get('llmModelId')
                self.tts_model_id = message.get('ttsModelId')
                self.messages = message.get('messages', [])
            elif event == 'userMessage':
                message_text = message.get('message')
                # 実行中のタスクをキャンセル
                if self.user_text_to_assistant_speech_task and not self.user_text_to_assistant_speech_task.done():
                    self.user_text_to_assistant_speech_task.cancel()
                    try:
                        await self.user_text_to_assistant_speech_task
                    except asyncio.CancelledError:
                        pass
                if message_text:
                    self.user_text_to_assistant_speech_task = asyncio.create_task(self.user_message_to_assistant_speech(message_text))
        except json.JSONDecodeError:
            logger.error('JSONメッセージの解析に失敗しました')

    async def add_audio(self, audio_chunk):
        self.buffer += audio_chunk
        while len(self.buffer) >= self.frame_bytes:
            frame = self.buffer[: self.frame_bytes]
            self.buffer = self.buffer[self.frame_bytes :]
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            if is_speech:
                self.audio_frames.append(frame)
                self.silence_counter = 0
                if not self.user_speech_started and len(self.audio_frames) > self.min_speech_frames:
                    self.user_speech_started = True
                    asyncio.create_task(self.websocket.send_json({"event": "userSpeechStart"}))
                    """
                    # 既存のタスクが実行中の場合、キャンセル
                    if self.user_speech_to_assistant_speech_task and not self.user_speech_to_assistant_speech_task.done():
                        self.user_speech_to_assistant_speech_task.cancel()
                        try:
                            await self.user_speech_to_assistant_speech_task
                        except asyncio.CancelledError:
                            pass
                    """
            elif self.user_speech_started:
                self.silence_counter += 1
                if self.silence_counter >= self.silence_frames:
                    logger.info(f"len(audio_frames) is {len(self.audio_frames)}")
                    audio_data = b"".join(self.audio_frames)
                    self.user_speech_to_assistant_speech_task = asyncio.create_task(self.user_speech_to_assistant_speech(audio_data))
                    asyncio.create_task(self.websocket.send_json({"event": "userSpeechEnd"}))
                    self.audio_frames = []
                    self.user_speech_started = False
            else:
                self.silence_counter = 0

    async def process_stt(self, audio_data):
        async with aiohttp.ClientSession() as session:
            async with session.post("http://localhost:8001/stt/bytes", data=audio_data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    text = result.get("text")
                    if text:
                        # 無視するフレーズの処理
                        ignore_phrases = {"ご視聴ありがとうございました", "ご視聴ありがとうございました。", "ありがとうございました", "ありがとうございました。", "はい", "ん", "2", "。"}
                        if text not in ignore_phrases:
                            return text
                else:
                    logger.error(f"STTサーバーのリクエストがステータス{resp.status}で失敗しました")
                    return None

    async def process_llm(self, text):
        async with aiohttp.ClientSession() as session:
            send_messages = self.messages + [{"role": "user", "content": text}]
            async with session.post("http://localhost:8002/llm", json=send_messages) as resp:
                if resp.status == 200:
                    llm_response = await resp.json()
                    llm_text = llm_response["message"]
                    return llm_text
                else:
                    logger.error(f"LLMリクエストがステータス{resp.status}で失敗しました")
                    return None

    async def process_tts(self, text):
        async with aiohttp.ClientSession() as session:
            async with session.post("http://localhost:8003/tts", json={"text": text}) as resp:
                if resp.status == 200:
                    audio_bytes = await resp.read()
                    return audio_bytes
                else:
                    logger.error(f"TTSリクエストがステータス{resp.status}で失敗しました")
                    return None

    async def user_speech_to_assistant_speech(self, audio_data):
        logger.info('user_speech_to_assistant_speech')
        transcript = await self.process_stt(audio_data)
        if not transcript:
            return
        await self.websocket.send_json({"event": "userSpeechTranscript", "transcript": transcript})
        llm_text = await self.process_llm(transcript)
        if not llm_text:
            return
        await self.websocket.send_json({"event": "assistantMessageGenerated", "generatedMessageContent": llm_text})
        audio_bytes = await self.process_tts(llm_text)
        if not audio_bytes:
            return
        await self.websocket.send_json({"event": "assistantSpeechGenerated", "audioData": list(audio_bytes)})
        self.user_speech_to_assistant_speech_task = None

    async def user_message_to_assistant_speech(self, message_text):
        llm_text = await self.process_llm(message_text)
        if not llm_text:
            return
        await self.websocket.send_json({"event": "assistantMessageGenerated", "generatedMessageContent": llm_text})
        audio_bytes = await self.process_tts(llm_text)
        if not audio_bytes:
            return
        await self.websocket.send_json({"event": "assistantSpeechGenerated", "audioData": list(audio_bytes)})
        self.user_text_to_assistant_speech_task = None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = Session(websocket)
    try:
        while True:
            message = await websocket.receive()
            if 'text' in message:
                data = message['text']
                await session.handle_text_message(data)
            elif 'bytes' in message:
                data = message['bytes']
                await session.add_audio(data)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocketでエラーが発生しました: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
