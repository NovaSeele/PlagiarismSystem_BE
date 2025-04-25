from fastapi import APIRouter, Depends, Query, UploadFile, File
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from typing import List

from schemas.user import User, UserInDB, Token, UserCreate, UserUpdateAvatar, UserChangePassword, UserMSVUpdate
from schemas.pdf_metadata import PDFMetadata
from db.repositories.user import get_current_user

from fastapi import APIRouter, Depends, HTTPException
from db.session import get_collection
from models.user import get_current_user

from services.pdf_metadata import upload_metadata_pdf_service, get_pdf_metadata_by_name, get_all_pdf_metadata, get_all_pdf_contents

router = APIRouter()


@router.post("/upload_file", response_model=PDFMetadata)
def upload_file(file: UploadFile = File(...), current_user: UserInDB = Depends(get_current_user)):
    try:
        return upload_metadata_pdf_service(file, current_user)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get_file_info_by_name/{filename}")
def get_file_info_by_name(filename: str):
    metadata = get_pdf_metadata_by_name(filename)

    if metadata is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy file.")

    return metadata

@router.get("/get_all_pdf_metadata")
def get_all_pdf_metadata_route():
    return get_all_pdf_metadata()

@router.get("/get_all_pdf_contents")
def get_all_pdf_contents_route():
    return get_all_pdf_contents()

