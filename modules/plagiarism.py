import numpy as np
import re
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class PlagiarismDetector:
    def __init__(self, num_perm=128, threshold=0.5, model_name='all-MiniLM-L6-v2'):
        """
        Khởi tạo bộ phát hiện đạo văn với nhiều phương pháp
        
        Tham số:
            num_perm (int): Số hoán vị cho MinHash
            threshold (float): Ngưỡng cho LSH
            model_name (str): Tên mô hình Sentence-Transformer
        """
        self.num_perm = num_perm
        self.threshold = threshold
        # Khởi tạo mô hình Sentence-Transformer
        try:
            self.model = SentenceTransformer(model_name)
            self.transformer_available = True
        except:
            print("Không thể tải mô hình Sentence-Transformer. Sẽ chỉ sử dụng Jaccard và MinHash.")
            self.transformer_available = False
            
    def preprocess_text(self, text):
        """Tiền xử lý văn bản: chuyển thành chữ thường, loại bỏ dấu câu"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def get_shingles(self, text, k=3):
        """Tạo k-shingles từ văn bản"""
        words = self.preprocess_text(text).split()
        return [' '.join(words[i:i+k]) for i in range(len(words)-k+1)]
    
    def jaccard_similarity(self, set1, set2):
        """Tính độ tương đồng Jaccard giữa hai tập hợp"""
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0
    
    def create_minhash(self, shingles):
        """Tạo MinHash từ shingles"""
        m = MinHash(num_perm=self.num_perm)
        for s in shingles:
            m.update(s.encode('utf-8'))
        return m
    
    def detect_plagiarism(self, documents, document_ids, weights=None):
        """
        Phát hiện đạo văn giữa các văn bản
        
        Tham số:
            documents (list): Danh sách các văn bản cần kiểm tra
            document_ids (list): Danh sách ID của các văn bản (để hiển thị kết quả)
            weights (dict): Trọng số cho mỗi phương pháp
                            {'jaccard': w1, 'minhash': w2, 'transformer': w3}
        
        Trả về:
            tuple: (Ma trận độ tương đồng, Báo cáo chi tiết)
        """
        if weights is None:
            # Mặc định trọng số bằng nhau
            if self.transformer_available:
                weights = {'jaccard': 0.3, 'minhash': 0.3, 'transformer': 0.4}
            else:
                weights = {'jaccard': 0.5, 'minhash': 0.5, 'transformer': 0}
        
        # Chuẩn hóa trọng số
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        n = len(documents)
        similarity_matrix = np.zeros((n, n))
        
        # Tạo shingles cho từng văn bản
        all_shingles = [set(self.get_shingles(doc)) for doc in documents]
        
        # Tạo MinHash
        minhashes = []
        for shingles in all_shingles:
            m = self.create_minhash(shingles)
            minhashes.append(m)
        
        # Tạo embedding Transformer (nếu có)
        embeddings = None
        if self.transformer_available:
            embeddings = self.model.encode(documents)
        
        # Tính toán độ tương đồng
        detailed_report = {}
        
        for i in range(n):
            for j in range(i, n):
                # Jaccard
                jaccard_sim = self.jaccard_similarity(all_shingles[i], all_shingles[j])
                
                # MinHash
                minhash_sim = minhashes[i].jaccard(minhashes[j])
                
                # Transformer
                transformer_sim = 0
                if self.transformer_available and embeddings is not None:
                    transformer_sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                    transformer_sim = max(0, min(1, transformer_sim))
                
                # Tính điểm tổng hợp
                weighted_sim = (
                    weights['jaccard'] * jaccard_sim +
                    weights['minhash'] * minhash_sim +
                    weights['transformer'] * transformer_sim
                )
                
                # Chuyển đổi thành phần trăm và lưu vào ma trận
                similarity_matrix[i, j] = similarity_matrix[j, i] = weighted_sim * 100
                
                # Thêm vào báo cáo chi tiết
                if i < j:
                    detailed_report[f"{document_ids[i]} vs {document_ids[j]}"] = {
                        "similarity_percentage": f"{similarity_matrix[i, j]:.2f}%",
                        "raw_score": float(similarity_matrix[i, j] / 100),
                        "components": {
                            "jaccard": float(jaccard_sim),
                            "minhash": float(minhash_sim),
                            "transformer": float(transformer_sim) if self.transformer_available else 0
                        }
                    }
        
        # Tạo dict chứa ma trận tương đồng
        similarity_dict = {}
        for i in range(n):
            similarity_dict[document_ids[i]] = {}
            for j in range(n):
                similarity_dict[document_ids[i]][document_ids[j]] = float(similarity_matrix[i, j])
        
        return similarity_dict, detailed_report