import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.pipeline import VoiceAssistantPipeline
from configs.config import AppConfig 


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = AppConfig()
pipeline = VoiceAssistantPipeline(config)


@app.get("/")
def home():
    return {
        "message": "Wake Word Backend is running"
    }


@app.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket):
    await websocket.accept()

    await websocket.send_json({
        "status": "Backend đã kết nối. Đang chờ âm thanh..."
    })

    try:
        while True:
            data = await websocket.receive_bytes()

            audio_chunk = (
                np.frombuffer(data, dtype=np.int16)
                .astype(np.float32)
                / 32768.0
            )

            result = pipeline.run_stream_chunk(
                audio_chunk=audio_chunk,
                sample_rate=16000
            )

            if result is not None:
                await websocket.send_json(result)

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")