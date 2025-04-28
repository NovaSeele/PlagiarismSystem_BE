from fastapi import APIRouter, HTTPException, WebSocket
from typing import Dict, List, Any, Optional
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
from modules.detail_plagiarism_module import detect_plagiarism_detailed_with_metadata

# Import the function to get all PDF metadata
from services.pdf_metadata import (
    get_all_pdf_metadata,
    get_all_pdf_contents,
    get_pdf_metadata_by_name,
    get_pdf_contents_by_names,
)

# Add these imports
import asyncio
import json
from starlette.websockets import WebSocketDisconnect

router = APIRouter()

# Add a global variable to store active websocket connections
active_websockets = set()


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


# New API endpoint for comparing two PDFs by filename using detailed plagiarism detection
@router.post("/compare-pdfs-by-name", response_model=Dict[str, Any])
async def compare_pdfs_by_name(request: FileNameComparisonRequest):
    """
    So sánh chi tiết hai văn bản PDF bằng cách chỉ định tên file.

    API này sẽ:
    1. Lấy nội dung của hai file PDF từ cơ sở dữ liệu theo tên file
    2. Thực hiện phân tích đạo văn chi tiết sử dụng phương pháp kiểm tra 3 lớp
    3. Trả về kết quả phân tích chi tiết với các phần bị đạo văn cụ thể

    Kết quả trả về bao gồm:
    - Tỷ lệ tương đồng tổng thể giữa hai văn bản
    - Danh sách chi tiết các phần được phát hiện bởi mỗi lớp (LSA/LDA, FastText, BERT)
    - Danh sách tổng hợp các phần bị đạo văn không trùng lặp
    - Thông tin về loại phần được phát hiện (câu, đoạn, cụm từ)
    - Tỷ lệ tương đồng của từng phần được phát hiện
    """
    try:
        # Lấy nội dung của hai file từ cơ sở dữ liệu
        content1 = get_pdf_metadata_by_name(request.file1_name)
        content2 = get_pdf_metadata_by_name(request.file2_name)

        if content1 is None:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy file PDF có tên '{request.file1_name}'",
            )

        if content2 is None:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy file PDF có tên '{request.file2_name}'",
            )

        # Chuẩn bị dữ liệu tài liệu cho phân tích
        doc_data = [
            {"content": content1, "filename": f"{request.file1_name}.pdf"},
            {"content": content2, "filename": f"{request.file2_name}.pdf"},
        ]

        # Thực hiện phân tích đạo văn chi tiết
        results = detect_plagiarism_detailed_with_metadata(doc_data)

        return results
    except HTTPException as he:
        raise he
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Đã xảy ra lỗi khi phân tích: {str(e)}"
        )


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
        # Lấy tất cả nội dung PDF từ database
        pdf_contents = get_all_pdf_contents()

        if len(pdf_contents) < 2:
            raise HTTPException(
                status_code=400,
                detail="Cần ít nhất 2 văn bản trong cơ sở dữ liệu để so sánh",
            )

        # Add the await keyword here
        results = await detect_plagiarism_layered_with_metadata(
            pdf_contents, active_websockets
        )
        return results
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Đã xảy ra lỗi khi phân tích: {str(e)}"
        )


# Route mới để thực hiện kiểm tra đạo văn tự động trên danh sách các file PDF được chỉ định
@router.post("/auto-layered-detection-by-names", response_model=Dict[str, Any])
async def auto_layered_plagiarism_detection_by_names(request: FilesComparisonRequest):
    """
    Phân tích và so sánh các văn bản PDF được chỉ định bởi danh sách tên file.

    API này sẽ:
    1. Lấy nội dung PDF từ cơ sở dữ liệu dựa trên tên file được cung cấp
    2. Thực hiện phân tích đạo văn sử dụng phương pháp kiểm tra 3 lớp
    3. Trả về kết quả chi tiết về tất cả các cặp văn bản

    Dữ liệu đầu vào bao gồm danh sách tên các file PDF (không bao gồm phần mở rộng .pdf)

    Kết quả trả về bao gồm:
    - Số lượng tài liệu và thời gian thực thi
    - Thống kê tóm tắt số lượng cặp qua từng lớp lọc
    - Danh sách chi tiết tất cả các cặp và kết quả ở từng lớp lọc
    """
    try:
        # Kiểm tra đầu vào có ít nhất 2 filename
        if len(request.filenames) < 2:
            raise HTTPException(
                status_code=400, detail="Cần ít nhất 2 tên file để so sánh"
            )

        # Lấy nội dung của các file PDF dựa trên tên file
        pdf_contents = get_pdf_contents_by_names(request.filenames)

        # Kiểm tra xem có tìm thấy tất cả các file không
        if len(pdf_contents) < len(request.filenames):
            # Tìm các file không tồn tại
            found_filenames = [
                doc["filename"].replace(".pdf", "") for doc in pdf_contents
            ]
            missing_files = [f for f in request.filenames if f not in found_filenames]

            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy các file sau: {', '.join(missing_files)}",
            )

        # Kiểm tra xem có đủ file để so sánh không
        if len(pdf_contents) < 2:
            raise HTTPException(
                status_code=400,
                detail="Cần ít nhất 2 file tồn tại để so sánh",
            )

        # Add the await keyword here
        results = await detect_plagiarism_layered_with_metadata(
            pdf_contents, active_websockets
        )
        return results
    except HTTPException as he:
        raise he
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Đã xảy ra lỗi khi phân tích: {str(e)}"
        )


# Add this new WebSocket endpoint
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        while True:
            await asyncio.sleep(3600)  # Keep connection open
    except WebSocketDisconnect:
        pass
    finally:
        active_websockets.remove(websocket)
