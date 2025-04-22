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

# Tải NLTK data nếu cần
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")


class TopicModelingDetectorDebug:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TopicModelingDetectorDebug, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return

        print("Initializing Debug Topic Modeling Detector...")
        # Initialize tools for text processing
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

        # LSA configuration
        self.n_components_lsa = 100  # Number of topics for LSA
        self.threshold = 0.4

        # LDA configuration
        self.n_components_lda = 20  # Number of topics for LDA

        self.initialized = True
        print("Debug Topic modeling detector initialized")

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
        vectorizer = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.9)
        tfidf_matrix = vectorizer.fit_transform(documents)
        return tfidf_matrix, vectorizer

    def _create_count_matrix(self, documents):
        """Create count matrix from documents for LDA"""
        vectorizer = CountVectorizer(stop_words="english", min_df=2, max_df=0.9)
        count_matrix = vectorizer.fit_transform(documents)
        return count_matrix, vectorizer

    def _apply_lsa(self, matrix, n_components=None):
        """Apply Latent Semantic Analysis (LSA)"""
        if n_components is None:
            n_components = self.n_components_lsa

        # Make sure n_components is not larger than the number of features
        n_components = min(n_components, matrix.shape[1] - 1, matrix.shape[0] - 1)

        # Apply SVD
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        lsa_result = svd.fit_transform(matrix)

        # Calculate explained variance
        explained_variance = svd.explained_variance_ratio_.sum()

        return lsa_result, svd, explained_variance

    def _apply_lda(self, matrix, n_components=None):
        """Apply Latent Dirichlet Allocation (LDA)"""
        if n_components is None:
            n_components = self.n_components_lda

        # Make sure n_components is not larger than the number of features
        n_components = min(n_components, matrix.shape[1] - 1)

        # Apply LDA
        lda = LatentDirichletAllocation(
            n_components=n_components, random_state=42, max_iter=10
        )
        lda_result = lda.fit_transform(matrix)

        return lda_result, lda

    def detect_plagiarism_lsa(self, text1, text2):
        """Detect plagiarism between two texts using LSA"""
        # Process both texts
        sentences1, _ = self.preprocess_text(text1)
        sentences2, _ = self.preprocess_text(text2)

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
            overall_similarity = float(np.mean(max_similarities))
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
        # Process both texts
        sentences1, _ = self.preprocess_text(text1)
        sentences2, _ = self.preprocess_text(text2)

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
                similar_pairs.append(
                    {
                        "text1_sentence": sentences1[i],
                        "text2_sentence": sentences2[j],
                        "similarity_score": round(float(max_sim), 3),
                        "similarity_percentage": round(float(max_sim) * 100, 1),
                    }
                )

        # Calculate overall topic distribution similarity
        if similarity_matrix.size > 0:
            max_similarities = np.max(similarity_matrix, axis=1)
            overall_similarity = float(np.mean(max_similarities))
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

        # Get LSA and LDA results
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
        """Detect plagiarism between multiple texts using LSA"""
        start_time = time.time()
        text_count = len(texts)

        # Initialize results
        results = {
            "document_count": text_count,
            "document_similarities": [],
            "document_pairs": [],
        }

        # Process all documents
        all_sentences = []
        sentence_counts = []

        for text in texts:
            sentences, _ = self.preprocess_text(text)
            all_sentences.extend(sentences)
            sentence_counts.append(len(sentences))

        # Create TF-IDF matrix for all sentences
        tfidf_matrix, vectorizer = self._create_tfidf_matrix(all_sentences)

        # Apply LSA
        lsa_result, svd, explained_variance = self._apply_lsa(tfidf_matrix)

        # Split the LSA results by document
        doc_vectors = []
        start_idx = 0
        for count in sentence_counts:
            doc_vectors.append(lsa_result[start_idx : start_idx + count])
            start_idx += count

        # Compare all document pairs
        for i in range(text_count):
            for j in range(i + 1, text_count):
                if len(doc_vectors[i]) == 0 or len(doc_vectors[j]) == 0:
                    continue

                # Calculate similarity matrix between documents i and j
                similarity_matrix = cosine_similarity(doc_vectors[i], doc_vectors[j])

                # Find similar sentence pairs
                similar_pairs = []
                for i_idx in range(len(doc_vectors[i])):
                    max_sim = np.max(similarity_matrix[i_idx])
                    j_idx = np.argmax(similarity_matrix[i_idx])

                    if max_sim > self.threshold:
                        similar_pairs.append(
                            {
                                "doc1_sentence": all_sentences[
                                    sum(sentence_counts[:i]) + i_idx
                                ],
                                "doc2_sentence": all_sentences[
                                    sum(sentence_counts[:j]) + j_idx
                                ],
                                "similarity_score": float(max_sim),
                                "similarity_percentage": round(float(max_sim) * 100, 1),
                            }
                        )

                # Calculate overall similarity
                if similarity_matrix.size > 0:
                    max_similarities = np.max(similarity_matrix, axis=1)
                    overall_similarity = float(np.mean(max_similarities))
                else:
                    overall_similarity = 0.0

                # Add to results if significant similarity found
                if overall_similarity > 0.3 or len(similar_pairs) > 0:
                    # Store results
                    results["document_pairs"].append(
                        {
                            "doc1_index": i,
                            "doc2_index": j,
                            "lsa_similarity": overall_similarity,
                            "overall_similarity": overall_similarity,
                            "overall_similarity_percentage": round(
                                overall_similarity * 100, 2
                            ),
                            "similar_sentence_pairs": similar_pairs[
                                :10
                            ],  # Limit to top 10
                        }
                    )

                    results["document_similarities"].append(
                        {
                            "doc_pair": f"doc{i+1}-doc{j+1}",
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
topic_detector_debug = TopicModelingDetectorDebug()


# Function cho so sánh một cặp văn bản với LSA/LDA
def compare_texts_with_topic_modeling_debug(text1: str, text2: str):
    """
    So sánh hai văn bản sử dụng topic modeling (LSA/LDA) phiên bản debug và trả về kết quả phân tích

    Args:
        text1: Văn bản thứ nhất
        text2: Văn bản thứ hai

    Returns:
        dict: Kết quả phân tích tương đồng
    """
    results = topic_detector_debug.detect_plagiarism_pair(text1, text2)
    return results


# Function cho so sánh nhiều văn bản với LSA
def compare_multiple_texts_with_topic_modeling_debug(texts: List[str]):
    """
    So sánh nhiều văn bản sử dụng LSA phiên bản debug và trả về kết quả phân tích

    Args:
        texts: Danh sách các văn bản cần so sánh

    Returns:
        dict: Kết quả phân tích tương đồng
    """
    if len(texts) < 2:
        raise ValueError("Cần ít nhất 2 văn bản để so sánh")

    results = topic_detector_debug.detect_plagiarism_multi(texts)
    return results
