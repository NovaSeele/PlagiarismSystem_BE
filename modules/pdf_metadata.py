from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
import PyPDF2

# Try to import fitz (PyMuPDF) but provide a fallback option
try:
    import fitz  # PyMuPDF

    USE_PYMUPDF = True
except ImportError:
    USE_PYMUPDF = False


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Trích xuất nội dung từ file PDF.
    """
    # Try to use PyMuPDF if available
    if USE_PYMUPDF:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = "\n".join(page.get_text("text") for page in doc)
            return text
        except Exception as e:
            # If PyMuPDF fails, fall back to PyPDF2
            pass

    # Fallback to PyPDF2
    try:
        reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")
