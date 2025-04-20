from typing import Optional

from fastapi import Depends, HTTPException, status

from schemas.pdf_metadata import PDFMetadata
from db.session import get_collection

def get_pdf_metadata_collection():
    pdf_metadata_collection = get_collection('pdf_metadata')
    return pdf_metadata_collection