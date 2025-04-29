from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.base import api_router
from config.config import settings
import os
from pyngrok import ngrok  # Thêm dòng này

app = FastAPI()

# Biến global để lưu trữ URL ngrok
NGROK_URL = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/api/ngrok-url")
def get_ngrok_url():
    """Endpoint trả về URL ngrok hiện tại"""
    global NGROK_URL
    return {"url": NGROK_URL}


if __name__ == "__main__":
    # Khởi tạo ngrok
    port = int(os.environ.get("PORT", 8888))
    ngrok_tunnel = ngrok.connect(port)  # Tạo tunnel
    public_url = ngrok_tunnel.public_url

    # Lưu URL vào biến global
    NGROK_URL = public_url

    print(f"Access your app here: {public_url}")

    # Chạy ứng dụng
    import uvicorn

    uvicorn.run("main:app", host=settings.API_HOST, port=port, reload=True)
