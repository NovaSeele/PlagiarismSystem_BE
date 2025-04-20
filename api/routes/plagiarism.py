from fastapi import APIRouter, HTTPException
from typing import Dict, List
from services.plagiarism import detect_plagiarism_from_contents
from services.pdf_metadata import get_content_organized_by_categories

router = APIRouter()

@router.get("/detect")
async def plagiarism_detection_api():
    """
    API endpoint để phát hiện đạo văn từ nội dung các file
    
    Trả về:
        Dict: Kết quả phát hiện đạo văn
    """
    try:
        # Lấy nội dung đã được tổ chức theo category
        contents_data = get_content_organized_by_categories()
        
        # Kiểm tra xem có dữ liệu không
        if not contents_data:
            return {
                "message": "Không tìm thấy nội dung nào để phân tích",
                "comparison_matrix": {},
                "detailed_report": {},
                "file_categories": {}
            }
        
        # Phát hiện đạo văn từ nội dung lấy được
        result = detect_plagiarism_from_contents(contents_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")