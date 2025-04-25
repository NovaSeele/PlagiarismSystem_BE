from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any
from pydantic import BaseModel, Field

from modules.bert_module import compare_two_texts, compare_multiple_texts
from modules.lsa_lda_module import (
    compare_texts_with_topic_modeling,
    compare_multiple_texts_with_topic_modeling,
)
from modules.lsa_lda_module_debug import (
    compare_texts_with_topic_modeling_debug,
    compare_multiple_texts_with_topic_modeling_debug,
)
from modules.fasttext_module import (
    compare_texts_with_fasttext,
    compare_multiple_texts_with_fasttext,
)
from modules.plagiarism import detect_plagiarism_layered
from modules.plagiarism_debug import detect_plagiarism_layered_debug
from modules.plagiarism_main_module import detect_plagiarism_layered_with_metadata

# Import the function to get all PDF metadata
from services.pdf_metadata import get_all_pdf_metadata, get_all_pdf_contents

router = APIRouter()


# Định nghĩa models Pydantic cho API
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


# Route cho việc so sánh hai văn bản với BERT
@router.post("/compare-pair", response_model=PlagiarismResponse)
async def compare_pair(request: PairPlagiarismRequest):
    """
    So sánh hai văn bản và trả về kết quả phân tích đạo văn sử dụng BERT.
    """
    try:
        results = compare_two_texts(request.text1, request.text2)
        return {
            "overall_similarity_percentage": results["overall_similarity_percentage"],
            "results": results,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Đã xảy ra lỗi khi phân tích: {str(e)}"
        )


# Route cho việc so sánh nhiều văn bản với BERT
@router.post("/compare-multiple", response_model=PlagiarismResponse)
async def compare_multiple(request: MultiPlagiarismRequest):
    """
    So sánh nhiều văn bản và trả về kết quả phân tích đạo văn sử dụng BERT.
    """
    try:
        if len(request.texts) < 2:
            raise HTTPException(
                status_code=400, detail="Cần ít nhất 2 văn bản để so sánh"
            )

        results = compare_multiple_texts(request.texts)

        # Tìm phần trăm tương đồng cao nhất giữa các cặp tài liệu
        max_similarity = 0
        if results["document_similarities"]:
            max_similarity = max(
                [
                    item["similarity_percentage"]
                    for item in results["document_similarities"]
                ]
            )

        return {"overall_similarity_percentage": max_similarity, "results": results}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Đã xảy ra lỗi khi phân tích: {str(e)}"
        )


# Route cho việc so sánh hai văn bản với LSA/LDA
@router.post("/topic-modeling/compare-pair", response_model=PlagiarismResponse)
async def compare_pair_topic_modeling(request: PairPlagiarismRequest):
    """
    So sánh hai văn bản và trả về kết quả phân tích đạo văn sử dụng LSA/LDA.
    """
    try:
        results = compare_texts_with_topic_modeling(request.text1, request.text2)
        return {
            "overall_similarity_percentage": results["overall_similarity_percentage"],
            "results": results,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Đã xảy ra lỗi khi phân tích: {str(e)}"
        )


# Route cho việc so sánh nhiều văn bản với LSA
@router.post("/topic-modeling/compare-multiple", response_model=PlagiarismResponse)
async def compare_multiple_topic_modeling(request: MultiPlagiarismRequest):
    """
    So sánh nhiều văn bản và trả về kết quả phân tích đạo văn sử dụng LSA.
    """
    try:
        if len(request.texts) < 2:
            raise HTTPException(
                status_code=400, detail="Cần ít nhất 2 văn bản để so sánh"
            )

        results = compare_multiple_texts_with_topic_modeling(request.texts)

        # Tìm phần trăm tương đồng cao nhất giữa các cặp tài liệu
        max_similarity = 0
        if results["document_similarities"]:
            max_similarity = max(
                [
                    item["similarity_percentage"]
                    for item in results["document_similarities"]
                ]
            )

        return {"overall_similarity_percentage": max_similarity, "results": results}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Đã xảy ra lỗi khi phân tích: {str(e)}"
        )


# Route cho việc so sánh nhiều văn bản với LSA
@router.post(
    "/topic-modeling/compare-multiple-debug", response_model=PlagiarismResponse
)
async def compare_multiple_topic_modeling_debug(request: MultiPlagiarismRequest):
    """
    So sánh nhiều văn bản và trả về kết quả phân tích đạo văn sử dụng LSA.
    """
    try:
        if len(request.texts) < 2:
            raise HTTPException(
                status_code=400, detail="Cần ít nhất 2 văn bản để so sánh"
            )

        results = compare_multiple_texts_with_topic_modeling_debug(request.texts)

        # Tìm phần trăm tương đồng cao nhất giữa các cặp tài liệu
        max_similarity = 0
        if results["document_similarities"]:
            max_similarity = max(
                [
                    item["similarity_percentage"]
                    for item in results["document_similarities"]
                ]
            )

        return {"overall_similarity_percentage": max_similarity, "results": results}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Đã xảy ra lỗi khi phân tích: {str(e)}"
        )


# Route cho việc so sánh hai văn bản với FastText
@router.post("/fasttext/compare-pair", response_model=PlagiarismResponse)
async def compare_pair_fasttext(request: PairPlagiarismRequest):
    """
    So sánh hai văn bản và trả về kết quả phân tích đạo văn sử dụng FastText embeddings.
    """
    try:
        results = compare_texts_with_fasttext(request.text1, request.text2)
        return {
            "overall_similarity_percentage": results["overall_similarity_percentage"],
            "results": results,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Đã xảy ra lỗi khi phân tích: {str(e)}"
        )


# Route cho việc so sánh nhiều văn bản với FastText
@router.post("/fasttext/compare-multiple", response_model=PlagiarismResponse)
async def compare_multiple_fasttext(request: MultiPlagiarismRequest):
    """
    So sánh nhiều văn bản và trả về kết quả phân tích đạo văn sử dụng FastText embeddings.
    """
    try:
        if len(request.texts) < 2:
            raise HTTPException(
                status_code=400, detail="Cần ít nhất 2 văn bản để so sánh"
            )

        results = compare_multiple_texts_with_fasttext(request.texts)

        # Tìm phần trăm tương đồng cao nhất giữa các cặp tài liệu
        max_similarity = 0
        if results["document_similarities"]:
            max_similarity = max(
                [
                    item["similarity_percentage"]
                    for item in results["document_similarities"]
                ]
            )

        return {"overall_similarity_percentage": max_similarity, "results": results}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Đã xảy ra lỗi khi phân tích: {str(e)}"
        )


# Route cho việc so sánh nhiều văn bản sử dụng phương pháp 3 lớp
@router.post("/layered-detection", response_model=LayeredPlagiarismResponse)
async def layered_plagiarism_detection(request: MultiPlagiarismRequest):
    """
    So sánh nhiều văn bản sử dụng phương pháp kiểm tra 3 lớp (LSA/LDA → FastText → BERT)
    để lọc dần và phát hiện đạo văn hiệu quả hơn.

    Phương pháp này tận dụng ưu điểm của từng thuật toán:
    - Lớp 1 (LSA/LDA): Phân tích topic nhanh chóng để lọc ra các cặp có khả năng tương đồng
    - Lớp 2 (FastText): Phân tích ngữ nghĩa với độ chính xác trung bình
    - Lớp 3 (BERT): Phân tích ngữ nghĩa với độ chính xác cao nhất

    Kết quả trả về chỉ bao gồm các cặp văn bản đã vượt qua cả 3 lớp lọc, với mức độ tương đồng cuối cùng.
    """
    try:
        if len(request.texts) < 2:
            raise HTTPException(
                status_code=400, detail="Cần ít nhất 2 văn bản để so sánh"
            )

        results = detect_plagiarism_layered(request.texts)
        return {"results": results}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Đã xảy ra lỗi khi phân tích: {str(e)}"
        )


# Route cho việc so sánh nhiều văn bản sử dụng phương pháp 3 lớp - bản debug chi tiết
@router.post("/layered-detection-debug", response_model=Dict[str, Any])
async def layered_plagiarism_detection_debug(request: MultiPlagiarismRequest):
    """
    Version debug chi tiết: So sánh nhiều văn bản sử dụng phương pháp kiểm tra 3 lớp
    (LSA/LDA → FastText → BERT) và cung cấp thông tin chi tiết về tất cả các cặp văn bản.

    Phương pháp này khác với bản thông thường ở chỗ:
    - Hiển thị TẤT CẢ các cặp văn bản, không chỉ những cặp vượt qua ngưỡng lọc
    - Hiển thị tỷ lệ tương đồng của TỪNG CẶP ở MỖI LAYER, kể cả khi không vượt qua ngưỡng lọc
    - Đánh dấu rõ các cặp vượt qua từng layer để theo dõi quá trình lọc

    Kết quả trả về bao gồm:
    - Số lượng tài liệu và thời gian thực thi
    - Thống kê tóm tắt số lượng cặp qua từng lớp lọc
    - Danh sách chi tiết tất cả các cặp và kết quả ở từng lớp lọc
    """
    try:
        if len(request.texts) < 2:
            raise HTTPException(
                status_code=400, detail="Cần ít nhất 2 văn bản để so sánh"
            )

        results = detect_plagiarism_layered_debug(request.texts)
        return results
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Đã xảy ra lỗi khi phân tích: {str(e)}"
        )


# Route mới để thực hiện kiểm tra đạo văn tự động trên tất cả các file PDF đã upload
@router.get("/auto-layered-detection-debug", response_model=Dict[str, Any])
async def auto_layered_plagiarism_detection_debug():
    """
    Tự động phân tích và so sánh tất cả các văn bản PDF đã được upload lên hệ thống.

    API này sẽ:
    1. Lấy tất cả nội dung PDF từ cơ sở dữ liệu
    2. Thực hiện phân tích đạo văn sử dụng phương pháp kiểm tra 3 lớp
    3. Trả về kết quả chi tiết về tất cả các cặp văn bản

    Không yêu cầu đầu vào nào, tự động xử lý tất cả tài liệu có sẵn.

    Kết quả trả về bao gồm:
    - Số lượng tài liệu và thời gian thực thi
    - Thống kê tóm tắt số lượng cặp qua từng lớp lọc
    - Danh sách chi tiết tất cả các cặp và kết quả ở từng lớp lọc
    """
    try:
        # Lấy tất cả nội dung PDF từ database (sử dụng hàm get_all_pdf_contents thay vì get_all_pdf_metadata)
        pdf_contents = get_all_pdf_contents()

        if len(pdf_contents) < 2:
            raise HTTPException(
                status_code=400,
                detail="Cần ít nhất 2 văn bản trong cơ sở dữ liệu để so sánh",
            )

        # Thực hiện phân tích đạo văn sử dụng phương thức mới với dữ liệu đầy đủ
        results = detect_plagiarism_layered_with_metadata(pdf_contents)
        return results
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Đã xảy ra lỗi khi phân tích: {str(e)}"
        )
