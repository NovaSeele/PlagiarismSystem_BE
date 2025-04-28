from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.base import api_router
from config.config import settings
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5173/*",
        "http://localhost:5174",
        "http://localhost:5174/*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
def read_root():
    return {"Hello": "World"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))  # lấy port từ biến môi trường
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
