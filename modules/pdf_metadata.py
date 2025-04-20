from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Trích xuất nội dung từ file PDF.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join(page.get_text("text") for page in doc)
    return text