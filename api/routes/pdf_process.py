from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional
import json
from pydantic import BaseModel

# from services.text_rank_keyword_vi import run_textrank
from services.pdf_metadata import process_single_pdf, process_multiple_pdfs

router = APIRouter()

@router.post("/upload_pdf", response_model=str)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a single PDF file and extract its content.
    Tải lên một file PDF và trích xuất nội dung của nó.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        # Sử dụng hàm process_single_pdf từ services.pdf_metadata
        text_content = process_single_pdf(file)

        return text_content

    except HTTPException as e:
        # Truyền lại HTTPException từ service
        raise e
    except Exception as e:
        # Xử lý các lỗi khác
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@router.post("/upload_multiple_pdfs", response_model=List[str])
async def upload_multiple_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload multiple PDF files and extract their content.
    Tải lên nhiều file PDF và trích xuất nội dung của chúng.
    """
    try:
        # Sử dụng hàm process_multiple_pdfs từ services.pdf_metadata
        results = process_multiple_pdfs(files)

        return results

    except HTTPException as e:
        # Truyền lại HTTPException từ service
        raise e
    except Exception as e:
        # Xử lý các lỗi khác
        raise HTTPException(
            status_code=500, detail=f"Error processing PDF files: {str(e)}"
        )
