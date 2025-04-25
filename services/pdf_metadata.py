# import fitz  # Incorrect import causing ModuleNotFoundError
import PyPDF2  # Use PyPDF2 for PDF processing

# import fitz  # This is the correct import for PyMuPDF in some environments

# Alternatively, you can try: from pymupdf import fitz

from typing import List, Dict, Tuple
from fastapi import Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordRequestForm

from datetime import datetime
from db.repositories.pdf_metadata import get_pdf_metadata_collection
from schemas.pdf_metadata import PDFMetadata, PDFMetadataInDB
from schemas.user import UserInDB
from db.session import db, fs

# Use the extract_text_from_pdf function from modules.pdf_metadata
from modules.pdf_metadata import extract_text_from_pdf

from bson import ObjectId
from pymongo.collection import Collection
from gridfs import GridFS

import json
import tempfile
import os
import io

# from services.text_rank_keyword_vi import run_textrank

from modules.keyword_classifier import categorize_combined

def upload_metadata_pdf_service(file: UploadFile, current_user: UserInDB):
    pdf_metadata_collection = db["pdf_metadata"]

    # Kiểm tra xem user đã tải lên file cùng tên chưa
    existing_metadata = pdf_metadata_collection.find_one(
        {"filename": file.filename, "user": current_user.username}
    )

    if existing_metadata:
        raise HTTPException(
            status_code=400, detail="File with this name already exists for this user."
        )

    pdf_bytes = file.file.read()  # Đọc nội dung file PDF

    # Tạo ID mới cho file trong GridFS
    file_id = ObjectId()

    # Lưu file vào GridFS
    with fs.open_upload_stream_with_id(file_id, file.filename) as stream:
        stream.write(pdf_bytes)

    extracted_text = extract_text_from_pdf(pdf_bytes)
    # Thay thế \n \n thành space
    extracted_text = extracted_text.replace("\n \n", " ")

    # keywords = run_textrank(extracted_text, stopwords, top_n=10)
    # keyword_categories = categorize_combined(keywords, categories, category_examples)

    metadata = {
        "filename": file.filename,
        "user": current_user.username,
        "content": extracted_text,
        "upload_at": datetime.now().isoformat(),
        # "categories": keyword_categories
    }

    pdf_metadata_collection.insert_one(metadata)  # Thêm metadata mới

    return PDFMetadata(**metadata)


def process_single_pdf(file: UploadFile) -> Tuple[str, int]:
    """
    Process a single PDF file to extract text content and page count.

    Args:
        file (UploadFile): The uploaded PDF file

    Returns:
        Tuple[str, int]: A tuple containing the extracted text and page count
    """
    try:
        # Đọc nội dung file PDF
        pdf_bytes = file.file.read()

        # Sử dụng hàm extract_text_from_pdf từ modules.pdf_metadata để trích xuất nội dung
        text_content = extract_text_from_pdf(pdf_bytes)
        
        # Thay thế \n \n thành space
        text_content = text_content.replace("\n \n", " ")

        # Đếm số trang sử dụng PyPDF2
        # with io.BytesIO(pdf_bytes) as pdf_stream:
        #     pdf_reader = PyPDF2.PdfReader(pdf_stream)
        #     page_count = len(pdf_reader.pages)

        return text_content

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


def process_multiple_pdfs(files: List[UploadFile]) -> List[Dict]:
    """
    Process multiple PDF files to extract text content and page count.

    Args:
        files (List[UploadFile]): List of uploaded PDF files

    Returns:
        List[Dict]: A list of dictionaries containing filename, content, and page count
    """
    results = []

    for file in files:
        # Kiểm tra xem file có phải là PDF không
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400, detail=f"File {file.filename} is not a PDF file"
            )

        try:
            # Reset file position
            file.file.seek(0)

            # Đọc nội dung file PDF
            pdf_bytes = file.file.read()

            # Sử dụng hàm extract_text_from_pdf từ modules.pdf_metadata để trích xuất nội dung
            text_content = extract_text_from_pdf(pdf_bytes)

            # Thay thế \n \n thành space
            text_content = text_content.replace("\n \n", " ")

            # Đếm số trang sử dụng PyPDF2
            with io.BytesIO(pdf_bytes) as pdf_stream:
                pdf_reader = PyPDF2.PdfReader(pdf_stream)
                page_count = len(pdf_reader.pages)

            # Thêm kết quả vào danh sách
            # results.append(
            #     {
            #         "filename": file.filename,
            #         "content": text_content,
            #         "page_count": page_count,
            #     }
            # )
            
            results.append(text_content)

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error processing {file.filename}: {str(e)}"
            )

    return results


def get_pdf_metadata_by_name(filename: str):
    """
    Lấy thông tin metadata của file PDF đã upload bằng filename.
    """
    pdf_metadata_collection = get_pdf_metadata_collection()

    # Tìm file theo filename
    file_data = pdf_metadata_collection.find_one({"filename": filename + ".pdf"})

    if not file_data:
        return None  # Trả về None nếu không tìm thấy file

    # return {
    #     "filename": file_data["filename"],
    #     "user": file_data["user"],
    #     "content": file_data["content"][:1000],  # Giới hạn nội dung trả về
    #     "upload_at": file_data["upload_at"]
    # }

    return file_data["content"]

def get_pdf_metadata_from_upload(file: UploadFile):
    """
    Lấy thông tin metadata của file PDF từ file được upload.
    """
    # Kiểm tra xem file có phải là PDF không
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail=f"File {file.filename} is not a PDF file"
        )

    try:
        # Reset file position
        file.file.seek(0)

        # Đọc nội dung file PDF
        pdf_bytes = file.file.read()

        # Trích xuất nội dung
        text_content = extract_text_from_pdf(pdf_bytes)

        # Đếm số trang
        with io.BytesIO(pdf_bytes) as pdf_stream:
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            page_count = len(pdf_reader.pages)

        return {
            "filename": file.filename,
            "content": text_content,
            "page_count": page_count
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing {file.filename}: {str(e)}"
        )


def get_all_pdf_metadata():
    """
    Lấy nội dung của tất cả các file PDF đã upload.

    Returns:
        list: Danh sách chứa nội dung của tất cả các file PDF
    """
    pdf_metadata_collection = get_pdf_metadata_collection()

    # Lấy tất cả các documents từ collection
    all_files = list(pdf_metadata_collection.find({}, {"content": 1}))

    # Nếu không có file nào, trả về danh sách trống
    if not all_files:
        return []

    # Trích xuất chỉ trường content từ mỗi document
    contents = [file.get("content", "") for file in all_files]

    return contents


def get_all_pdf_contents():
    """
    Lấy nội dung của tất cả các file PDF đã upload.

    Returns:
        list: Danh sách chứa nội dung của tất cả các file PDF
    """
    pdf_metadata_collection = get_pdf_metadata_collection()

    # Lấy tất cả các documents từ collection
    all_files = list(pdf_metadata_collection.find({}))

    # Nếu không có file nào, trả về danh sách trống
    if not all_files:
        return []

    # Chuyển đổi ObjectId thành string
    for file in all_files:
        file["_id"] = str(file["_id"])

    return all_files
