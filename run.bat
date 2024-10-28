@echo off
start cmd /k "uvicorn server_webhook:app --host 0.0.0.0 --port 8000 --reload"
start cmd /k "uvicorn server_stt:app --host 0.0.0.0 --port 8001 --reload"
start cmd /k "uvicorn server_llm:app --host 0.0.0.0 --port 8002 --reload"
start cmd /k "uvicorn server_tts:app --host 0.0.0.0 --port 8003 --reload"