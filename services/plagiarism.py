from collections import defaultdict
from typing import Dict, List
from modules.plagiarism import PlagiarismDetector

# Khởi tạo detector toàn cục để tái sử dụng
_detector = None

def get_detector():
    """Hàm tạo và trả về detector singleton"""
    global _detector
    if _detector is None:
        _detector = PlagiarismDetector(num_perm=128, threshold=0.5)
    return _detector

def detect_plagiarism_from_contents(contents_data: Dict[str, List[dict]]) -> Dict:
    """
    Phát hiện đạo văn từ dữ liệu theo định dạng nội dung theo danh mục
    
    Tham số:
        contents_data (Dict): Dữ liệu chứa nội dung của file theo danh mục
        
    Trả về:
        Dict: Kết quả phát hiện đạo văn
    """
    # Tạo từ điển chứa nội dung của các file (loại bỏ trùng lặp)
    unique_files = {}
    file_categories = defaultdict(list)
    
    # Xử lý dữ liệu đầu vào
    for category, files in contents_data.items():
        for file in files:
            filename = file["filename"]
            content = file["content"]
            
            # Lưu nội dung file và ghi nhận danh mục
            unique_files[filename] = content
            file_categories[filename].append(category)
    
    # Tạo danh sách các văn bản và ID tương ứng
    documents = []
    document_ids = []
    
    for filename, content in unique_files.items():
        documents.append(content)
        document_ids.append(filename)
    
    # Phát hiện đạo văn
    detector = get_detector()
    similarity_matrix, detailed_report = detector.detect_plagiarism(
        documents, document_ids
    )
    
    # Tạo kết quả trả về
    result = {
        "comparison_matrix": similarity_matrix,
        "detailed_report": detailed_report,
        "file_categories": {filename: categories for filename, categories in file_categories.items()}
    }
    
    return result