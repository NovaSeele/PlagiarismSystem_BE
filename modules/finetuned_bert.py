from typing import List, Dict, Any
import numpy as np
import torch
import re
import nltk
import time
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Tải NLTK data nếu cần
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


class FinetunedBertDetector:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FinetunedBertDetector, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"):
        if self.initialized:
            return

        print("Initializing BERT Detector...")
        self.model_name = model_name

        # Sentence-level similarity threshold (0.6)
        # This is the main threshold used to determine if two sentences are similar enough
        # to be considered potential plagiarism candidates at the sentence level.
        # BERT models produce higher-quality semantic representations than traditional methods,
        # so this threshold can be higher than for LSA (0.4) or FastText (0.5).
        # Value 0.6 means sentences need to be at least 60% similar in semantic meaning
        # according to the BERT embeddings to be flagged as similar.
        # This higher threshold helps reduce false positives while maintaining detection sensitivity.
        self.threshold = 0.6

        # Document-level plagiarism confidence thresholds
        # High confidence plagiarism threshold (0.8)
        # When overall document similarity exceeds this threshold, we have high confidence
        # that plagiarism has occurred. BERT models can capture deeper semantic relationships,
        # so we set this threshold higher than other methods (FastText: 0.75).
        # This helps ensure extremely high precision in plagiarism detection.
        self.plagiarism_threshold = 0.8  # High confidence plagiarism

        # Potential plagiarism threshold (0.65)
        # Documents with similarity between 0.65 and 0.8 are flagged as potential plagiarism.
        # This range captures cases where significant content similarity exists,
        # but may include legitimate paraphrasing or similar topics with different expressions.
        # These cases warrant human review to make a final determination.
        self.potential_plagiarism_threshold = 0.65  # Potential plagiarism

        # Load BERT model and tokenizer
        try:
            print(f"Loading BERT model: {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            print("BERT model loaded successfully.")
        except Exception as e:
            raise ValueError(f"Failed to load BERT model: {str(e)}")

        self.initialized = True
        print("BERT detector initialized")

    def preprocess_text(self, text):
        """Preprocess text for BERT analysis"""
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Split into sentences
        sentences = sent_tokenize(text)

        return sentences

    def get_bert_embedding(self, text):
        """Generate BERT embedding for a piece of text"""
        # Tokenize and prepare for BERT
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use CLS token embedding as sentence representation
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()

        # Normalize embedding
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embedding = embeddings / norm

        return normalized_embedding[0]

    def get_sentence_embeddings(self, sentences):
        """Generate embeddings for a list of sentences"""
        return [self.get_bert_embedding(sentence) for sentence in sentences]

    def detect_plagiarism_pair(self, text1, text2):
        """Detect plagiarism between two texts using BERT"""
        # Preprocess texts
        sentences1 = self.preprocess_text(text1)
        sentences2 = self.preprocess_text(text2)

        # Get sentence embeddings
        sentence_embeddings1 = self.get_sentence_embeddings(sentences1)
        sentence_embeddings2 = self.get_sentence_embeddings(sentences2)

        # Calculate sentence-level similarity matrix
        similarity_matrix = cosine_similarity(
            sentence_embeddings1, sentence_embeddings2
        )

        # Find similar sentence pairs
        similar_pairs = []
        for i in range(len(sentences1)):
            max_sim = np.max(similarity_matrix[i])
            j = np.argmax(similarity_matrix[i])

            # Only include pairs above the similarity threshold
            if max_sim > self.threshold:
                similar_pairs.append(
                    {
                        "text1_sentence": sentences1[i],
                        "text2_sentence": sentences2[j],
                        "similarity_score": round(float(max_sim), 3),
                        "similarity_percentage": round(float(max_sim) * 100, 1),
                    }
                )

        # Calculate document-level embedding by averaging sentence embeddings
        doc_embedding1 = np.mean(sentence_embeddings1, axis=0)
        doc_embedding2 = np.mean(sentence_embeddings2, axis=0)

        # Calculate document similarity
        doc_similarity = float(
            cosine_similarity([doc_embedding1], [doc_embedding2])[0, 0]
        )

        # Calculate overall similarity (weighted average of document and sentence similarity)
        if similarity_matrix.size > 0:
            max_similarities = np.max(similarity_matrix, axis=1)
            sentence_similarity = float(np.mean(max_similarities))
        else:
            sentence_similarity = 0.0

        # Weighted average (document similarity has higher weight in BERT)
        doc_weight = 0.7
        sentence_weight = 0.3
        overall_similarity = (doc_weight * doc_similarity) + (
            sentence_weight * sentence_similarity
        )

        # Calculate similarity percentage
        similarity_percentage = round(overall_similarity * 100, 2)

        # Determine plagiarism status
        is_plagiarized = similarity_percentage >= self.plagiarism_threshold * 100
        potential_plagiarism = (
            similarity_percentage >= self.potential_plagiarism_threshold * 100
            and similarity_percentage < self.plagiarism_threshold * 100
        )

        # Get word counts
        word_count1 = sum(len(sentence.split()) for sentence in sentences1)
        word_count2 = sum(len(sentence.split()) for sentence in sentences2)

        # Compile results
        results = {
            "summary": {
                "text1_word_count": word_count1,
                "text2_word_count": word_count2,
                "text1_sentence_count": len(sentences1),
                "text2_sentence_count": len(sentences2),
            },
            "document_similarity": {
                "similarity_score": round(doc_similarity, 3),
                "similarity_percentage": round(doc_similarity * 100, 2),
            },
            "sentence_similarity": {
                "similar_sentence_pairs": similar_pairs,
                "overall_similarity": sentence_similarity,
                "overall_similarity_percentage": round(sentence_similarity * 100, 2),
            },
            "overall_similarity_percentage": similarity_percentage,
            "is_plagiarized": is_plagiarized,
            "potential_plagiarism": potential_plagiarism,
        }

        return results


# Khởi tạo singleton detector
bert_detector = FinetunedBertDetector()


# Function cho so sánh một cặp văn bản với BERT
def compare_texts_with_bert(text1: str, text2: str):
    """
    So sánh hai văn bản sử dụng BERT embeddings và trả về kết quả phân tích

    Args:
        text1: Văn bản thứ nhất
        text2: Văn bản thứ hai

    Returns:
        dict: Kết quả phân tích tương đồng
    """
    results = bert_detector.detect_plagiarism_pair(text1, text2)
    return results


# Function cho so sánh nhiều văn bản với BERT
def compare_multiple_texts_with_bert(texts: List[str]):
    """
    So sánh nhiều văn bản sử dụng BERT embeddings và trả về kết quả phân tích

    Args:
        texts: Danh sách các văn bản cần so sánh

    Returns:
        dict: Kết quả phân tích tương đồng
    """
    if len(texts) < 2:
        raise ValueError("Cần ít nhất 2 văn bản để so sánh")

    start_time = time.time()
    text_count = len(texts)

    # Tạo kết quả
    results = {
        "document_count": text_count,
        "document_similarities": [],
        "document_pairs": [],
        "is_plagiarized": False,
        "potential_plagiarism": False,
    }

    # So sánh từng cặp văn bản
    for i in range(text_count):
        for j in range(i + 1, text_count):
            # Gọi hàm so sánh cặp văn bản
            pair_result = compare_texts_with_bert(texts[i], texts[j])

            # Lấy thông tin tương đồng
            overall_similarity = pair_result["overall_similarity_percentage"] / 100.0

            # Cập nhật trạng thái đạo văn
            if pair_result["is_plagiarized"]:
                results["is_plagiarized"] = True
            if pair_result["potential_plagiarism"]:
                results["potential_plagiarism"] = True

            # Thêm vào kết quả nếu có tương đồng đáng kể
            if overall_similarity > 0.3:
                # Lưu kết quả cho cặp văn bản
                results["document_pairs"].append(
                    {
                        "doc1_index": i,
                        "doc2_index": j,
                        "document_similarity": pair_result["document_similarity"][
                            "similarity_score"
                        ],
                        "sentence_similarity": pair_result["sentence_similarity"][
                            "overall_similarity"
                        ],
                        "overall_similarity": overall_similarity,
                        "overall_similarity_percentage": round(
                            overall_similarity * 100, 2
                        ),
                        "similar_sentence_pairs": pair_result["sentence_similarity"][
                            "similar_sentence_pairs"
                        ][
                            :10
                        ],  # Lấy 10 cặp tương đồng cao nhất
                        "is_plagiarized": pair_result["is_plagiarized"],
                        "potential_plagiarism": pair_result["potential_plagiarism"],
                    }
                )

                # Lưu tóm tắt tương đồng
                results["document_similarities"].append(
                    {
                        "doc_pair": f"doc{i+1}-doc{j+1}",
                        "similarity_percentage": round(overall_similarity * 100, 2),
                        "is_plagiarized": pair_result["is_plagiarized"],
                        "potential_plagiarism": pair_result["potential_plagiarism"],
                    }
                )

    # Sắp xếp kết quả theo độ tương đồng
    results["document_pairs"].sort(key=lambda x: x["overall_similarity"], reverse=True)
    results["document_similarities"].sort(
        key=lambda x: x["similarity_percentage"], reverse=True
    )

    # Thêm thời gian thực thi
    end_time = time.time()
    results["execution_time_seconds"] = round(end_time - start_time, 2)

    return results
