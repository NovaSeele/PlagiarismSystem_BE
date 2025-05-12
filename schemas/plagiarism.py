from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional


# New model for file comparison by name
class FileNameComparisonRequest(BaseModel):
    file1_name: str = Field(
        ..., description="Tên của file PDF thứ nhất để kiểm tra (không cần .pdf)"
    )
    file2_name: str = Field(
        ..., description="Tên của file PDF thứ hai để kiểm tra (không cần .pdf)"
    )


class PairPlagiarismRequest(BaseModel):
    text1: str = Field(..., description="Văn bản thứ nhất để kiểm tra", min_length=50)
    text2: str = Field(..., description="Văn bản thứ hai để kiểm tra", min_length=50)


class MultiPlagiarismRequest(BaseModel):
    texts: List[str] = Field(
        ..., description="Danh sách các văn bản để kiểm tra", min_items=2
    )


class PlagiarismResponse(BaseModel):
    overall_similarity_percentage: float
    results: dict


class LayeredPlagiarismResponse(BaseModel):
    results: dict


# Tạo model mới cho request chứa danh sách tên file để so sánh
class FilesComparisonRequest(BaseModel):
    filenames: List[str] = Field(
        ...,
        description="Danh sách tên các file PDF để kiểm tra (không cần .pdf)",
        min_items=2,
    )
