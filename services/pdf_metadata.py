from typing import List
from fastapi import Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordRequestForm

from datetime import datetime
from db.repositories.pdf_metadata import get_pdf_metadata_collection
from schemas.pdf_metadata import PDFMetadata, PDFMetadataInDB
from schemas.user import UserInDB
from db.session import db, fs

from modules.pdf_metadata import extract_text_from_pdf

from bson import ObjectId
from pymongo.collection import Collection
from gridfs import GridFS

import json

from services.text_rank_keyword_vi import run_textrank

from modules.keyword_classifier import categorize_combined

# Load categories and category examples from categories.json
with open(
    "D:/Code/NovaSeelePlagiarismSystem/backend/datasets/categories.json",
    "r",
    encoding="utf-8",
) as f:
    data = json.load(f)
    categories = data["categories"]
    category_examples = data["category_examples"]
    stopwords = data["stopwords"]


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

    keywords = run_textrank(extracted_text, stopwords, top_n=10)
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


def get_content_organized_by_categories():
    """
    Lấy tất cả nội dung PDF được tổ chức theo categories.

    Returns:
        dict: Dictionary với key là tên category và value là danh sách các file (filename và content) thuộc category đó
    """
    pdf_metadata_collection = get_pdf_metadata_collection()

    # Lấy tất cả các documents từ collection
    all_files = list(pdf_metadata_collection.find({}))

    # Nếu không có file nào, trả về dictionary trống thay vì None
    if not all_files:
        return {}

    # Dictionary để lưu trữ kết quả theo category
    result_by_category = {}

    # Duyệt qua từng file
    for file in all_files:
        filename = file.get("filename")
        content = file.get("content")

        # Duyệt qua từng category của file
        for category in file.get("categories", []):
            # Nếu category chưa tồn tại trong kết quả, tạo mới
            if category not in result_by_category:
                result_by_category[category] = []

            # Thêm thông tin file vào category
            result_by_category[category].append(
                {"filename": filename, "content": content}
            )

    return result_by_category


def get_pdf_content_by_categories(categories: list):
    """
    Lấy nội dung của tất cả các file PDF có chứa tất cả các categories được chỉ định.

    Args:
        categories (list): Danh sách các danh mục cần tìm kiếm

    Returns:
        list: Danh sách chứa nội dung của các file thuộc tất cả các category đó
    """
    pdf_metadata_collection = get_pdf_metadata_collection()

    # Tìm tất cả các file có chứa TẤT CẢ các category này trong mảng categories
    files = pdf_metadata_collection.find({"categories": {"$all": categories}})

    # Nếu không tìm thấy file nào
    if not files:
        return None

    # Tạo danh sách chứa nội dung của tất cả các file tìm được
    contents = []
    for file in files:
        contents.append({"filename": file["filename"], "content": file["content"]})

    return contents


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
