from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import numpy as np
import re
import nltk
import time
import faiss
import itertools
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Tải NLTK data nếu cần
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")


# Tạo router
router = APIRouter()


# Class PlagiarismDetector
class PlagiarismDetector:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PlagiarismDetector, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, bert_model="all-mpnet-base-v2"):
        if self.initialized:
            return

        print("Initializing Plagiarism Detector...")
        # Initialize BERT model
        self.bert_model = SentenceTransformer(bert_model)

        # Sentence-level similarity threshold (0.5)
        # This threshold determines when two sentences are considered similar enough
        # to be included in the "similar_sentence_pairs" list in the results.
        # Value of 0.5 means sentences need to be at least 50% similar in their
        # BERT embedding representations to be considered a match.
        # For BERT models, this is a balanced threshold that:
        # - Is sensitive enough to detect paraphrased content
        # - Strict enough to avoid too many false positives
        # This threshold is used in the _analyze_semantic_similarity method.
        self.threshold = 0.5

        # Plagiarism thresholds
        # High confidence plagiarism threshold (0.75)
        # When the overall similarity between documents exceeds 75%, content is
        # classified as definitely plagiarized with high confidence.
        # This high threshold helps ensure that flagged content truly represents
        # substantial copying rather than coincidental similarity.
        # Used in both detect_plagiarism_pair and detect_plagiarism_multi methods.
        self.plagiarism_threshold = 0.75  # High confidence plagiarism

        # Potential plagiarism threshold (0.60)
        # Documents with similarity between 60-75% are flagged as potentially plagiarized.
        # This range typically indicates:
        # - Significant paraphrasing
        # - Partial copying with modifications
        # - Content from common sources
        # These cases require human review to make a final determination.
        self.potential_plagiarism_threshold = 0.60  # Potential plagiarism

        # Initialize tools for English text processing
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

        # FAISS configuration - Sentence similarity threshold for efficient multi-document comparison (0.7)
        # This is a higher threshold than the standard sentence threshold (0.5) because:
        # - It's used specifically in the FAISS-based approximate nearest neighbor search
        # - FAISS returns multiple potential matches which may include false positives
        # - The higher value (0.7) ensures we only keep high-quality matches
        # - Helps maintain efficiency when comparing many documents by reducing low-value matches
        # This threshold is used in the detect_plagiarism_multi method with FAISS.
        self.faiss_threshold = 0.7

        self.initialized = True
        print("Plagiarism detector initialized")

    def preprocess_text(self, text):
        """Preprocess English text with stemming and stopword removal"""
        # Normalize whitespace and convert to lowercase
        text = re.sub(r"\s+", " ", text).strip().lower()

        # Split into sentences
        sentences = sent_tokenize(text)

        # Tokenize each sentence
        tokenized_sentences = []
        for sentence in sentences:
            # Tokenize
            tokens = word_tokenize(sentence)
            # Remove punctuation and stopwords
            filtered_tokens = [
                self.stemmer.stem(word)
                for word in tokens
                if word.isalnum() and word not in self.stop_words
            ]
            tokenized_sentences.append(filtered_tokens)

        return sentences, tokenized_sentences

    def preprocess_for_semantic(self, text):
        """Preprocess text for semantic analysis (keeping original form for BERT)"""
        text = re.sub(r"\s+", " ", text).strip()
        sentences = sent_tokenize(text)
        return sentences

    def highlight_common_phrases(self, text1, text2):
        """Find exact matching phrases between two texts"""
        # Tokenize texts
        tokens1 = word_tokenize(text1.lower())
        tokens2 = word_tokenize(text2.lower())

        # Create n-grams (phrases) of varying lengths
        common_phrases = []

        # Check for phrases of different lengths (3-7 words)
        for n in range(3, 8):
            ngrams1 = set()
            for i in range(len(tokens1) - n + 1):
                ngrams1.add(" ".join(tokens1[i : i + n]))

            ngrams2 = set()
            for i in range(len(tokens2) - n + 1):
                ngrams2.add(" ".join(tokens2[i : i + n]))

            # Find common n-grams
            common = ngrams1.intersection(ngrams2)
            for phrase in common:
                if (
                    len(phrase.split()) >= 3
                ):  # Only include phrases with at least 3 words
                    common_phrases.append(
                        {"phrase": phrase, "word_count": len(phrase.split())}
                    )

        return common_phrases

    def detect_plagiarism_pair(self, text1, text2):
        """Detect plagiarism between two texts"""
        # Preprocess for semantic analysis (keep original form)
        sentences1 = self.preprocess_for_semantic(text1)
        sentences2 = self.preprocess_for_semantic(text2)

        # Word counts
        word_count1 = len(word_tokenize(text1))
        word_count2 = len(word_tokenize(text2))

        # Get semantic similarity
        semantic_similarity = self._analyze_semantic_similarity(sentences1, sentences2)

        # Find common phrases
        common_phrases = self.highlight_common_phrases(text1, text2)

        # Get similarity percentage
        similarity_percentage = semantic_similarity["overall_similarity_percentage"]

        # Determine plagiarism status
        is_plagiarized = similarity_percentage >= self.plagiarism_threshold * 100
        potential_plagiarism = (
            similarity_percentage >= self.potential_plagiarism_threshold * 100
            and similarity_percentage < self.plagiarism_threshold * 100
        )

        # Compile results
        results = {
            "summary": {
                "text1_word_count": word_count1,
                "text2_word_count": word_count2,
                "text1_sentence_count": len(sentences1),
                "text2_sentence_count": len(sentences2),
            },
            "semantic_similarity": semantic_similarity,
            "common_phrases": common_phrases,
            "overall_similarity_percentage": similarity_percentage,
            "is_plagiarized": is_plagiarized,
            "potential_plagiarism": potential_plagiarism,
        }

        return results

    def detect_plagiarism_multi(self, texts):
        """Detect plagiarism between multiple texts using FAISS"""
        start_time = time.time()
        text_count = len(texts)

        print(f"\n== BERT Module: Bắt đầu xử lý {text_count} văn bản ==")

        # Extract sentences from all texts
        all_sentences = []
        all_sentence_indices = (
            []
        )  # Keep track of which document each sentence belongs to

        print(f"BERT: Đang trích xuất câu từ các văn bản...")
        for i, text in enumerate(texts):
            print(f"  BERT: Đang xử lý văn bản {i+1}/{text_count}")
            sentences = self.preprocess_for_semantic(text)
            all_sentences.extend(sentences)
            all_sentence_indices.extend([i] * len(sentences))
        print(
            f"BERT: Đã trích xuất tổng cộng {len(all_sentences)} câu từ {text_count} văn bản"
        )

        # Generate embeddings for all sentences
        print(f"BERT: Đang tạo embedding cho {len(all_sentences)} câu...")
        all_embeddings = self.bert_model.encode(all_sentences)
        print(f"BERT: Đã tạo embedding xong")

        # Create FAISS index
        print(f"BERT: Đang xây dựng chỉ mục FAISS...")
        dimension = all_embeddings.shape[1]
        index = faiss.IndexFlatIP(
            dimension
        )  # Inner product is equivalent to cosine similarity for normalized vectors

        # Normalize vectors for cosine similarity
        faiss.normalize_L2(all_embeddings)

        # Add vectors to the index
        index.add(all_embeddings)
        print(f"BERT: Đã xây dựng chỉ mục FAISS xong")

        # Set up results structure
        results = {
            "document_count": text_count,
            "document_similarities": [],
            "document_pairs": [],
            "is_plagiarized": False,  # Will be set to True if any document pair is plagiarized
            "potential_plagiarism": False,  # Will be set to True if any document pair has potential plagiarism
        }

        # Calculate similarity between all document pairs
        total_pairs = text_count * (text_count - 1) // 2
        print(f"BERT: Bắt đầu so sánh {total_pairs} cặp văn bản...")

        pair_count = 0
        for i, j in itertools.combinations(range(text_count), 2):
            pair_count += 1
            if pair_count % 5 == 0 or pair_count == total_pairs:
                print(
                    f"  BERT: Đang xử lý cặp {pair_count}/{total_pairs} ({round(pair_count/total_pairs*100, 1)}%) - Văn bản {i} & {j}"
                )

            # Get indices of sentences from documents i and j
            i_indices = [
                idx for idx, doc_idx in enumerate(all_sentence_indices) if doc_idx == i
            ]
            j_indices = [
                idx for idx, doc_idx in enumerate(all_sentence_indices) if doc_idx == j
            ]

            if not i_indices or not j_indices:
                continue

            # Get embeddings for sentences in documents i and j
            i_embeddings = all_embeddings[i_indices]
            j_embeddings = all_embeddings[j_indices]

            # Compute similarity matrix
            sim_matrix = np.zeros((len(i_indices), len(j_indices)))

            # Use FAISS to find similar sentences
            k = min(5, len(j_indices))  # Search for top-k similar sentences
            for idx, emb in enumerate(i_embeddings):
                emb_reshaped = emb.reshape(1, -1)
                scores, neighbors = index.search(emb_reshaped, k)

                # Filter neighbors that belong to document j
                for neighbor_idx, score in zip(neighbors[0], scores[0]):
                    if neighbor_idx in j_indices:
                        j_pos = j_indices.index(neighbor_idx)
                        sim_matrix[idx, j_pos] = max(sim_matrix[idx, j_pos], score)

            # Find similar sentence pairs
            similar_pairs = []
            for i_idx in range(len(i_indices)):
                for j_idx in range(len(j_indices)):
                    if sim_matrix[i_idx, j_idx] > self.faiss_threshold:
                        similar_pairs.append(
                            {
                                "doc1_sentence": all_sentences[i_indices[i_idx]],
                                "doc2_sentence": all_sentences[j_indices[j_idx]],
                                "similarity_score": float(sim_matrix[i_idx, j_idx]),
                                "similarity_percentage": round(
                                    float(sim_matrix[i_idx, j_idx]) * 100, 1
                                ),
                            }
                        )

            # Calculate overall similarity
            if sim_matrix.size > 0:
                doc_similarity = float(np.mean(np.max(sim_matrix, axis=1)))
            else:
                doc_similarity = 0.0

            # If documents are similar, find common phrases
            if doc_similarity > 0.3 or len(similar_pairs) > 0:
                # Find common phrases
                common_phrases = self.highlight_common_phrases(texts[i], texts[j])

                # Calculate similarity percentage
                similarity_percentage = round(doc_similarity * 100, 2)

                # Determine plagiarism status for this pair
                is_pair_plagiarized = (
                    similarity_percentage >= self.plagiarism_threshold * 100
                )
                potential_pair_plagiarism = (
                    similarity_percentage >= self.potential_plagiarism_threshold * 100
                    and similarity_percentage < self.plagiarism_threshold * 100
                )

                # Update global plagiarism flags
                if is_pair_plagiarized:
                    results["is_plagiarized"] = True
                if potential_pair_plagiarism:
                    results["potential_plagiarism"] = True

                # Store results
                results["document_pairs"].append(
                    {
                        "doc1_index": i,
                        "doc2_index": j,
                        "semantic_similarity": doc_similarity,
                        "overall_similarity": doc_similarity,
                        "overall_similarity_percentage": similarity_percentage,
                        "similar_sentence_pairs": similar_pairs[
                            :10
                        ],  # Limit to top 10 similar pairs
                        "common_phrases": common_phrases[
                            :10
                        ],  # Limit to top 10 common phrases
                        "is_plagiarized": is_pair_plagiarized,
                        "potential_plagiarism": potential_pair_plagiarism,
                    }
                )

                results["document_similarities"].append(
                    {
                        "doc_pair": f"doc{i+1}-doc{j+1}",
                        "similarity_percentage": similarity_percentage,
                        "is_plagiarized": is_pair_plagiarized,
                        "potential_plagiarism": potential_pair_plagiarism,
                    }
                )

        # Sort document pairs by similarity
        results["document_pairs"].sort(
            key=lambda x: x["overall_similarity"], reverse=True
        )
        results["document_similarities"].sort(
            key=lambda x: x["similarity_percentage"], reverse=True
        )

        # Add execution time
        end_time = time.time()
        results["execution_time_seconds"] = round(end_time - start_time, 2)

        print(f"BERT: Đã hoàn thành so sánh {total_pairs} cặp văn bản")
        print(f"BERT: Thời gian thực hiện: {round(time.time() - start_time, 2)} giây")

        return results

    def _analyze_semantic_similarity(self, sentences1, sentences2):
        """Analyze semantic similarity using BERT"""
        # Encode sentences
        embeddings1 = self.bert_model.encode(sentences1)
        embeddings2 = self.bert_model.encode(sentences2)

        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)

        # Find similar sentence pairs
        similar_pairs = []

        for i in range(len(sentences1)):
            max_sim = np.max(similarity_matrix[i])
            j = np.argmax(similarity_matrix[i])

            if max_sim > self.threshold:
                similar_pairs.append(
                    {
                        "text1_sentence": sentences1[i],
                        "text2_sentence": sentences2[j],
                        "similarity_score": round(float(max_sim), 3),
                        "similarity_percentage": round(float(max_sim) * 100, 1),
                    }
                )

        # Calculate overall semantic similarity
        if similarity_matrix.size > 0:
            # Use the maximum similarity for each sentence in text1
            max_similarities = np.max(similarity_matrix, axis=1)
            overall_similarity = np.mean(max_similarities)
        else:
            overall_similarity = 0.0

        return {
            "similar_sentence_pairs": similar_pairs,
            "overall_similarity": float(overall_similarity),
            "overall_similarity_percentage": round(float(overall_similarity) * 100, 2),
        }


# Khởi tạo singleton detector
detector = PlagiarismDetector()


# Function cho so sánh một cặp văn bản
def compare_two_texts(text1: str, text2: str):
    """
    So sánh hai văn bản và trả về kết quả phân tích đạo văn

    Args:
        text1: Văn bản thứ nhất
        text2: Văn bản thứ hai

    Returns:
        dict: Kết quả phân tích đạo văn
    """
    results = detector.detect_plagiarism_pair(text1, text2)
    return results


# Function cho so sánh nhiều văn bản
def compare_multiple_texts(texts: List[str]):
    """
    So sánh nhiều văn bản và trả về kết quả phân tích đạo văn

    Args:
        texts: Danh sách các văn bản cần so sánh

    Returns:
        dict: Kết quả phân tích đạo văn
    """
    if len(texts) < 2:
        raise ValueError("Cần ít nhất 2 văn bản để so sánh")

    results = detector.detect_plagiarism_multi(texts)
    return results
