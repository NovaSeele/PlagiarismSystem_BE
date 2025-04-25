from fastapi import APIRouter
from .routes.user import router as user_router
from .routes.pdf_metadata import router as pdf_metadata_router
from .routes.pdf_process import router as pdf_process_router
from .routes.plagiarism import router as plagiarism_router

api_router = APIRouter()
api_router.include_router(user_router, tags=["user"])
api_router.include_router(pdf_metadata_router, tags=["pdf_metadata"])
api_router.include_router(pdf_process_router, tags=["pdf_process"])
api_router.include_router(plagiarism_router, tags=["plagiarism"])



