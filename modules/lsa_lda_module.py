from typing import List, Dict, Any
import numpy as np
import re
import nltk
import time
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import hashlib

# Tải NLTK data nếu cần
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")


class TopicModelingDetector:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TopicModelingDetector, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return

        print("Initializing Topic Modeling Detector...")
        # Initialize tools for text processing
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

        # LSA configuration - fixed to ensure consistency across all comparisons
        self.n_components_lsa = 10  # Fixed number of topics for LSA
        # Similarity threshold for sentence-level similarity detection.
        # When comparing sentences using cosine similarity, only pairs with
        # similarity scores greater than this threshold (0.4) will be considered as similar.
        # Lower value (e.g., 0.3) would be more sensitive and find more potential matches
        # (potentially more false positives), while higher value (e.g., 0.6) would be
        # more strict and only detect highly similar content (potentially missing some plagiarism).
        # This value was chosen empirically to balance detection sensitivity with precision.
        self.threshold = 0.4

        # LDA configuration - fixed to ensure consistency across all comparisons
        self.n_components_lda = 5  # Fixed number of topics for LDA

        self.initialized = True
        print("Topic modeling detector initialized")

    def _get_document_hash(self, text):
        """Generate a consistent hash for a document to enable deterministic ordering"""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _sort_documents(self, text1, text2):
        """Sort two documents by their hash to ensure consistent ordering"""
        hash1 = self._get_document_hash(text1)
        hash2 = self._get_document_hash(text2)

        if hash1 < hash2:
            return (text1, text2, False)  # No swap needed
        else:
            return (text2, text1, True)  # Swapped

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

    def _create_tfidf_matrix(self, documents):
        """Create TF-IDF matrix from documents"""
        if not documents:
            raise ValueError("No documents provided for TF-IDF vectorization")

        # Use constant parameters for consistency across all comparisons
        # min_df=1 ensures even rare terms are included
        vectorizer = TfidfVectorizer(stop_words="english", min_df=1, max_df=0.95)
        tfidf_matrix = vectorizer.fit_transform(documents)

        return tfidf_matrix, vectorizer

    def _create_count_matrix(self, documents):
        """Create count matrix from documents for LDA"""
        if not documents:
            raise ValueError("No documents provided for Count vectorization")

        # Use constant parameters for consistency across all comparisons
        # min_df=1 ensures even rare terms are included
        vectorizer = CountVectorizer(stop_words="english", min_df=1, max_df=0.95)
        count_matrix = vectorizer.fit_transform(documents)

        return count_matrix, vectorizer

    def _apply_lsa(self, matrix, n_components=None):
        """Apply Latent Semantic Analysis (LSA)"""
        if n_components is None:
            # Use a fixed number for n_components (10) to maintain consistency
            # regardless of matrix size or number of documents
            n_components = 10

        # Make sure n_components is not larger than the matrix dimensions, but at least 1
        n_components = min(n_components, matrix.shape[1] - 1, matrix.shape[0] - 1)
        n_components = max(1, n_components)

        # Apply SVD
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        lsa_result = svd.fit_transform(matrix)

        # Calculate explained variance
        explained_variance = svd.explained_variance_ratio_.sum()

        return lsa_result, svd, explained_variance

    def _apply_lda(self, matrix, n_components=None):
        """Apply Latent Dirichlet Allocation (LDA)"""
        if n_components is None:
            # Use a fixed number for n_components (5) to maintain consistency
            # regardless of matrix size or number of documents
            n_components = 5

        # Make sure n_components is not larger than the number of features, but at least 1
        n_components = min(n_components, matrix.shape[1] - 1)
        n_components = max(1, n_components)

        # Apply LDA
        lda = LatentDirichletAllocation(
            n_components=n_components, random_state=42, max_iter=10
        )
        lda_result = lda.fit_transform(matrix)

        return lda_result, lda

    def detect_plagiarism_lsa(self, text1, text2):
        """Detect plagiarism between two texts using LSA"""
        # Sort texts for consistent processing
        sorted_text1, sorted_text2, swapped = self._sort_documents(text1, text2)

        # Process both texts
        sentences1, _ = self.preprocess_text(sorted_text1)
        sentences2, _ = self.preprocess_text(sorted_text2)

        # Handle case when either document has no sentences
        if not sentences1 or not sentences2:
            return {
                "similar_sentence_pairs": [],
                "overall_similarity": 0.0,
                "overall_similarity_percentage": 0.0,
                "explained_variance": 0.0,
                "top_topics": [],
            }

        # Create a combined corpus
        all_documents = sentences1 + sentences2

        # Create TF-IDF matrix
        tfidf_matrix, vectorizer = self._create_tfidf_matrix(all_documents)

        # Apply LSA
        lsa_result, svd, explained_variance = self._apply_lsa(tfidf_matrix)

        # Split the result back to the two texts
        doc1_vectors = lsa_result[: len(sentences1)]
        doc2_vectors = lsa_result[len(sentences1) :]

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(doc1_vectors, doc2_vectors)

        # Find similar sentence pairs
        similar_pairs = []
        for i in range(len(sentences1)):
            max_sim = np.max(similarity_matrix[i])
            j = np.argmax(similarity_matrix[i])

            if max_sim > self.threshold:
                # Map sentence pairs back to original order if needed
                if not swapped:
                    similar_pairs.append(
                        {
                            "text1_sentence": sentences1[i],
                            "text2_sentence": sentences2[j],
                            "similarity_score": round(float(max_sim), 3),
                            "similarity_percentage": round(float(max_sim) * 100, 1),
                        }
                    )
                else:
                    similar_pairs.append(
                        {
                            "text1_sentence": sentences2[j],
                            "text2_sentence": sentences1[i],
                            "similarity_score": round(float(max_sim), 3),
                            "similarity_percentage": round(float(max_sim) * 100, 1),
                        }
                    )

        # Calculate overall semantic similarity in a symmetrical way
        if similarity_matrix.size > 0:
            # Calculate maximum similarities in both directions
            # 1. For each sentence in text1, find max similarity with any sentence in text2
            max_similarities_1to2 = np.max(similarity_matrix, axis=1)
            mean_similarity_1to2 = float(np.mean(max_similarities_1to2))

            # 2. For each sentence in text2, find max similarity with any sentence in text1
            max_similarities_2to1 = np.max(similarity_matrix, axis=0)
            mean_similarity_2to1 = float(np.mean(max_similarities_2to1))

            # Use the average of both directions for a symmetrical measure
            overall_similarity = (mean_similarity_1to2 + mean_similarity_2to1) / 2.0
        else:
            overall_similarity = 0.0

        # Get top terms for each topic (for explanation)
        feature_names = vectorizer.get_feature_names_out()
        topic_terms = []
        for topic_idx, topic in enumerate(svd.components_):
            top_features_idx = topic.argsort()[:-11:-1]  # Get indices of top 10 terms
            top_terms = [feature_names[i] for i in top_features_idx]
            topic_terms.append({"topic_id": topic_idx, "terms": top_terms})

        lsa_results = {
            "similar_sentence_pairs": similar_pairs,
            "overall_similarity": overall_similarity,
            "overall_similarity_percentage": round(overall_similarity * 100, 2),
            "explained_variance": round(explained_variance * 100, 2),
            "top_topics": topic_terms[:5],  # Include top 5 topics
        }

        return lsa_results

    def detect_plagiarism_lda(self, text1, text2):
        """Detect plagiarism between two texts using LDA"""
        # Sort texts for consistent processing
        sorted_text1, sorted_text2, swapped = self._sort_documents(text1, text2)

        # Process both texts
        sentences1, _ = self.preprocess_text(sorted_text1)
        sentences2, _ = self.preprocess_text(sorted_text2)

        # Handle case when either document has no sentences
        if not sentences1 or not sentences2:
            return {
                "similar_sentence_pairs": [],
                "overall_similarity": 0.0,
                "overall_similarity_percentage": 0.0,
                "top_topics": [],
            }

        # Create a combined corpus
        all_documents = sentences1 + sentences2

        # Create count matrix
        count_matrix, vectorizer = self._create_count_matrix(all_documents)

        # Apply LDA
        lda_result, lda_model = self._apply_lda(count_matrix)

        # Split the result back to the two texts
        doc1_vectors = lda_result[: len(sentences1)]
        doc2_vectors = lda_result[len(sentences1) :]

        # Calculate topic distribution similarity
        similarity_matrix = cosine_similarity(doc1_vectors, doc2_vectors)

        # Find similar sentence pairs based on topic distributions
        similar_pairs = []
        for i in range(len(sentences1)):
            max_sim = np.max(similarity_matrix[i])
            j = np.argmax(similarity_matrix[i])

            if max_sim > self.threshold:
                # Map sentence pairs back to original order if needed
                if not swapped:
                    similar_pairs.append(
                        {
                            "text1_sentence": sentences1[i],
                            "text2_sentence": sentences2[j],
                            "similarity_score": round(float(max_sim), 3),
                            "similarity_percentage": round(float(max_sim) * 100, 1),
                        }
                    )
                else:
                    similar_pairs.append(
                        {
                            "text1_sentence": sentences2[j],
                            "text2_sentence": sentences1[i],
                            "similarity_score": round(float(max_sim), 3),
                            "similarity_percentage": round(float(max_sim) * 100, 1),
                        }
                    )

        # Calculate overall topic distribution similarity in a symmetrical way
        if similarity_matrix.size > 0:
            # Calculate maximum similarities in both directions
            # 1. For each sentence in text1, find max similarity with any sentence in text2
            max_similarities_1to2 = np.max(similarity_matrix, axis=1)
            mean_similarity_1to2 = float(np.mean(max_similarities_1to2))

            # 2. For each sentence in text2, find max similarity with any sentence in text1
            max_similarities_2to1 = np.max(similarity_matrix, axis=0)
            mean_similarity_2to1 = float(np.mean(max_similarities_2to1))

            # Use the average of both directions for a symmetrical measure
            overall_similarity = (mean_similarity_1to2 + mean_similarity_2to1) / 2.0
        else:
            overall_similarity = 0.0

        # Get top terms for each topic (for explanation)
        feature_names = vectorizer.get_feature_names_out()
        topic_terms = []
        for topic_idx, topic in enumerate(lda_model.components_):
            top_features_idx = topic.argsort()[:-11:-1]  # Get indices of top 10 terms
            top_terms = [feature_names[i] for i in top_features_idx]
            topic_terms.append({"topic_id": topic_idx, "terms": top_terms})

        lda_results = {
            "similar_sentence_pairs": similar_pairs,
            "overall_similarity": overall_similarity,
            "overall_similarity_percentage": round(overall_similarity * 100, 2),
            "top_topics": topic_terms[:5],  # Include top 5 topics
        }

        return lda_results

    def detect_plagiarism_pair(self, text1, text2):
        """Detect plagiarism between two texts using both LSA and LDA"""
        # Get word counts
        word_count1 = len(word_tokenize(text1))
        word_count2 = len(word_tokenize(text2))

        # Get sentence counts
        sentences1 = sent_tokenize(text1)
        sentences2 = sent_tokenize(text2)

        # Get LSA and LDA results (they will internally sort the texts for consistent processing)
        lsa_results = self.detect_plagiarism_lsa(text1, text2)
        lda_results = self.detect_plagiarism_lda(text1, text2)

        # Combine results
        results = {
            "summary": {
                "text1_word_count": word_count1,
                "text2_word_count": word_count2,
                "text1_sentence_count": len(sentences1),
                "text2_sentence_count": len(sentences2),
            },
            "lsa_similarity": lsa_results,
            "lda_similarity": lda_results,
        }

        # Calculate overall similarity with weighted average
        # LSA gets higher weight since it's generally more accurate for similarity
        lsa_weight = 0.7
        lda_weight = 0.3

        overall_similarity = (
            lsa_weight * lsa_results["overall_similarity"]
            + lda_weight * lda_results["overall_similarity"]
        )

        results["overall_similarity_percentage"] = round(overall_similarity * 100, 2)

        return results

    def detect_plagiarism_multi(self, texts):
        """Detect plagiarism between multiple texts using pairwise comparison"""
        start_time = time.time()
        text_count = len(texts)

        # Initialize results
        results = {
            "document_count": text_count,
            "document_similarities": [],
            "document_pairs": [],
        }

        # Compare all document pairs using the same method as in detect_plagiarism_pair
        # This ensures consistency regardless of how many documents are compared
        for i in range(text_count):
            for j in range(i + 1, text_count):
                # Store original indices for reporting
                original_i, original_j = i, j

                # Use the same method as when comparing just two texts directly
                pair_result = self.detect_plagiarism_pair(texts[i], texts[j])

                # Get the overall similarity from the pair result
                overall_similarity = (
                    pair_result["overall_similarity_percentage"] / 100.0
                )

                # Extract similar sentence pairs
                similar_sentence_pairs = []
                if (
                    "lsa_similarity" in pair_result
                    and "similar_sentence_pairs" in pair_result["lsa_similarity"]
                ):
                    for pair in pair_result["lsa_similarity"]["similar_sentence_pairs"]:
                        similar_sentence_pairs.append(
                            {
                                "doc1_sentence": pair["text1_sentence"],
                                "doc2_sentence": pair["text2_sentence"],
                                "similarity_score": pair["similarity_score"],
                                "similarity_percentage": pair["similarity_percentage"],
                            }
                        )

                # Only keep top 10 most similar pairs
                similar_sentence_pairs.sort(
                    key=lambda x: x["similarity_score"], reverse=True
                )
                similar_sentence_pairs = similar_sentence_pairs[:10]

                # Add to results if significant similarity found or similar sentences found
                if overall_similarity > 0.3 or len(similar_sentence_pairs) > 0:
                    # Store document pair results with original indices
                    results["document_pairs"].append(
                        {
                            "doc1_index": original_i,
                            "doc2_index": original_j,
                            "lsa_similarity": pair_result["lsa_similarity"][
                                "overall_similarity"
                            ],
                            "overall_similarity": overall_similarity,
                            "overall_similarity_percentage": round(
                                overall_similarity * 100, 2
                            ),
                            "similar_sentence_pairs": similar_sentence_pairs,
                        }
                    )

                    # Store simple similarity results for summary
                    results["document_similarities"].append(
                        {
                            "doc_pair": f"doc{original_i+1}-doc{original_j+1}",
                            "similarity_percentage": round(overall_similarity * 100, 2),
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

        return results


# Khởi tạo singleton detector
topic_detector = TopicModelingDetector()


# Function cho so sánh một cặp văn bản với LSA/LDA
def compare_texts_with_topic_modeling(text1: str, text2: str):
    """
    So sánh hai văn bản sử dụng topic modeling (LSA/LDA) và trả về kết quả phân tích

    Args:
        text1: Văn bản thứ nhất
        text2: Văn bản thứ hai

    Returns:
        dict: Kết quả phân tích tương đồng
    """
    results = topic_detector.detect_plagiarism_pair(text1, text2)
    return results


# Function cho so sánh nhiều văn bản với LSA
def compare_multiple_texts_with_topic_modeling(texts: List[str]):
    """
    So sánh nhiều văn bản sử dụng LSA và trả về kết quả phân tích

    Gọi trực tiếp compare_texts_with_topic_modeling cho từng cặp văn bản
    để đảm bảo kết quả đồng nhất với so sánh từng cặp riêng lẻ.

    Args:
        texts: Danh sách các văn bản cần so sánh

    Returns:
        dict: Kết quả phân tích tương đồng
    """
    start_time = time.time()

    if len(texts) < 2:
        raise ValueError("Cần ít nhất 2 văn bản để so sánh")

    text_count = len(texts)

    # Initialize results
    results = {
        "document_count": text_count,
        "document_similarities": [],
        "document_pairs": [],
    }

    # Iterate through all pairs of texts
    for i in range(text_count):
        for j in range(i + 1, text_count):
            # Call the same function used for individual pair comparison
            pair_result = compare_texts_with_topic_modeling(texts[i], texts[j])

            # Get overall similarity from the pair result
            overall_similarity = pair_result["overall_similarity_percentage"] / 100.0

            # Extract similar sentence pairs
            similar_sentence_pairs = []
            if (
                "lsa_similarity" in pair_result
                and "similar_sentence_pairs" in pair_result["lsa_similarity"]
            ):
                for pair in pair_result["lsa_similarity"]["similar_sentence_pairs"]:
                    similar_sentence_pairs.append(
                        {
                            "doc1_sentence": pair["text1_sentence"],
                            "doc2_sentence": pair["text2_sentence"],
                            "similarity_score": pair["similarity_score"],
                            "similarity_percentage": pair["similarity_percentage"],
                        }
                    )

            # Only keep top 10 most similar pairs
            similar_sentence_pairs.sort(
                key=lambda x: x["similarity_score"], reverse=True
            )
            similar_sentence_pairs = similar_sentence_pairs[:10]

            # Add to results if significant similarity found or similar sentences found
            if overall_similarity > 0.3 or len(similar_sentence_pairs) > 0:
                # Store document pair results
                results["document_pairs"].append(
                    {
                        "doc1_index": i,
                        "doc2_index": j,
                        "lsa_similarity": pair_result["lsa_similarity"][
                            "overall_similarity"
                        ],
                        "overall_similarity": overall_similarity,
                        "overall_similarity_percentage": round(
                            overall_similarity * 100, 2
                        ),
                        "similar_sentence_pairs": similar_sentence_pairs,
                    }
                )

                # Store simple similarity results for summary
                results["document_similarities"].append(
                    {
                        "doc_pair": f"doc{i+1}-doc{j+1}",
                        "similarity_percentage": round(overall_similarity * 100, 2),
                    }
                )

    # Sort document pairs by similarity
    results["document_pairs"].sort(key=lambda x: x["overall_similarity"], reverse=True)
    results["document_similarities"].sort(
        key=lambda x: x["similarity_percentage"], reverse=True
    )

    # Add execution time
    end_time = time.time()
    results["execution_time_seconds"] = round(end_time - start_time, 2)

    return results
