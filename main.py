from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.base import api_router
from config.config import settings
from contextlib import asynccontextmanager
import os
from modules.fasttext import preload_fasttext_model


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     print("Preloading FastText model...")
#     fasttext_path = "D:/Code/NovaSeelePlagiarismSystem/backend/models/cc.en.300.bin.gz"

#     possible_paths = [
#         fasttext_path,
#         os.path.join("models", fasttext_path),
#         os.path.join("data", "models", fasttext_path),
#         os.path.join("..", "models", fasttext_path),
#         os.path.join(os.path.dirname(__file__), "models", fasttext_path),
#     ]

#     for path in possible_paths:
#         if os.path.exists(path):
#             fasttext_path = path
#             print(f"Found FastText model at {fasttext_path}")
#             break

#     model = preload_fasttext_model(fasttext_path)
#     if model is not None:
#         print("FastText model preloaded successfully!")
#     else:
#         print(
#             "Failed to preload FastText model. Will use WordNet only for semantic similarity."
#         )

#     yield


# app = FastAPI(lifespan=lifespan)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5174/*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
# app.include_router(api_router, prefix="/api")


@app.get("/")
def read_root():
    return {"Hello": "World"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
