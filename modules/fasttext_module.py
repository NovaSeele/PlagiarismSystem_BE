from typing import List, Dict, Any
import numpy as np
import re
import nltk
import time
import gensim
import gensim.downloader as gensim_downloader
from gensim.models import FastText
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Tải NLTK data nếu cần
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")


class FastTextDetector:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FastTextDetector, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(
        self, model_name="fasttext-wiki-news-subwords-300", download_if_missing=True
    ):
        if self.initialized:
            return

        print("Initializing FastText Detector...")
        self.model_name = model_name
        # Sentence-level similarity threshold (0.5)
        # This threshold is used when finding similar sentence pairs between documents.
        # Only sentence pairs with cosine similarity scores above this threshold
        # will be included in the result's "similar_sentence_pairs" list.
        # Value of 0.5 means sentences need to be at least 50% similar in meaning
        # based on their FastText vector representations.
        # This is moderate sensitivity - balances finding meaningful matches without too many false positives.
        self.threshold = 0.5

        # Plagiarism thresholds
        # High confidence plagiarism threshold (0.75)
        # When the overall similarity score between two documents is above this threshold (75%),
        # the system will confidently classify it as plagiarism.
        # This high threshold helps ensure that false positives are minimized
        # and that detected plagiarism cases have strong evidence.
        self.plagiarism_threshold = 0.75  # High confidence plagiarism

        # Potential plagiarism threshold (0.60)
        # When the overall similarity is between 0.60 and 0.75 (60%-75%),
        # the content is flagged as "potential plagiarism" - meaning it needs further review.
        # This helps identify borderline cases that might represent partial plagiarism,
        # paraphrasing, or content with common sources but not direct copying.
        self.potential_plagiarism_threshold = 0.60  # Potential plagiarism

        # Initialize tools for English text processing
        self.stop_words = set(stopwords.words("english"))

        # Load FastText model
        try:
            print(f"Loading FastText model: {self.model_name}...")
            self.model = gensim_downloader.load(self.model_name)
            print("FastText model loaded successfully.")
        except Exception as e:
            if download_if_missing:
                print(f"Error loading model: {str(e)}. Will attempt to download.")
                try:
                    self.model = gensim_downloader.load(self.model_name)
                    print("FastText model downloaded and loaded successfully.")
                except Exception as download_error:
                    print(f"Failed to download model: {str(download_error)}")
                    # Fallback to smaller model
                    fallback_model = "glove-wiki-gigaword-100"
                    print(f"Falling back to {fallback_model}...")
                    self.model = gensim_downloader.load(fallback_model)
            else:
                raise ValueError(f"Failed to load FastText model: {str(e)}")

        self.initialized = True
        print("FastText detector initialized")

    def preprocess_text(self, text):
        """Preprocess English text for FastText analysis"""
        # Normalize whitespace and convert to lowercase
        text = re.sub(r"\s+", " ", text).strip().lower()

        # Split into sentences
        sentences = sent_tokenize(text)

        # Tokenize each sentence (without removing stopwords for embeddings)
        tokenized_sentences = []
        filtered_sentences = []

        for sentence in sentences:
            # Tokenize
            tokens = word_tokenize(sentence)
            tokenized_sentences.append(tokens)

            # Filter tokens (for secondary analyses)
            filtered_tokens = [
                word
                for word in tokens
                if word.isalnum() and word not in self.stop_words
            ]
            filtered_sentences.append(filtered_tokens)

        return sentences, tokenized_sentences, filtered_sentences

    def get_sentence_embedding(self, tokens):
        """Generate sentence embedding by averaging word vectors"""
        vectors = []
        for token in tokens:
            try:
                vectors.append(self.model[token])
            except KeyError:
                # Skip words not in vocabulary
                continue

        if vectors:
            # Average the vectors
            embedding = np.mean(vectors, axis=0)
            # Normalize the vector
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        else:
            # Return zero vector if no words were in vocabulary
            return np.zeros(self.model.vector_size)

    def get_document_embedding(self, sentences):
        """Generate document embedding by averaging sentence embeddings"""
        sentence_embeddings = []
        for sentence in sentences:
            sentence_embeddings.append(self.get_sentence_embedding(sentence))

        if sentence_embeddings:
            doc_embedding = np.mean(sentence_embeddings, axis=0)
            # Normalize
            norm = np.linalg.norm(doc_embedding)
            if norm > 0:
                doc_embedding = doc_embedding / norm
            return doc_embedding
        else:
            return np.zeros(self.model.vector_size)

    def find_common_phrases(self, tokenized_sentences1, tokenized_sentences2, n=3):
        """Find common n-grams between two texts"""
        # Create n-grams for both texts
        ngrams1 = set()
        for sentence in tokenized_sentences1:
            for i in range(len(sentence) - n + 1):
                ngrams1.add(" ".join(sentence[i : i + n]))

        ngrams2 = set()
        for sentence in tokenized_sentences2:
            for i in range(len(sentence) - n + 1):
                ngrams2.add(" ".join(sentence[i : i + n]))

        # Find common phrases
        common_phrases = list(ngrams1.intersection(ngrams2))

        # Sort by length (number of tokens)
        common_phrases.sort(key=lambda x: len(x.split()), reverse=True)

        return [
            {"phrase": phrase, "word_count": len(phrase.split())}
            for phrase in common_phrases
        ]

    def detect_plagiarism_pair(self, text1, text2):
        """Detect plagiarism between two texts using FastText"""
        # Preprocess texts
        original_sentences1, tokenized_sentences1, filtered_sentences1 = (
            self.preprocess_text(text1)
        )
        original_sentences2, tokenized_sentences2, filtered_sentences2 = (
            self.preprocess_text(text2)
        )

        # Generate sentence embeddings
        sentence_embeddings1 = [
            self.get_sentence_embedding(sentence) for sentence in tokenized_sentences1
        ]
        sentence_embeddings2 = [
            self.get_sentence_embedding(sentence) for sentence in tokenized_sentences2
        ]

        # Calculate document embeddings
        doc_embedding1 = self.get_document_embedding(tokenized_sentences1)
        doc_embedding2 = self.get_document_embedding(tokenized_sentences2)

        # Calculate document-level similarity
        doc_similarity = float(
            cosine_similarity([doc_embedding1], [doc_embedding2])[0, 0]
        )

        # Calculate sentence-level similarities
        similarity_matrix = cosine_similarity(
            sentence_embeddings1, sentence_embeddings2
        )

        # Find similar sentence pairs
        similar_pairs = []
        for i in range(len(original_sentences1)):
            max_sim = np.max(similarity_matrix[i])
            j = np.argmax(similarity_matrix[i])

            if max_sim > self.threshold:
                similar_pairs.append(
                    {
                        "text1_sentence": original_sentences1[i],
                        "text2_sentence": original_sentences2[j],
                        "similarity_score": round(float(max_sim), 3),
                        "similarity_percentage": round(float(max_sim) * 100, 1),
                    }
                )

        # Calculate sentence similarity average
        if similarity_matrix.size > 0:
            max_similarities = np.max(similarity_matrix, axis=1)
            sentence_similarity = float(np.mean(max_similarities))
        else:
            sentence_similarity = 0.0

        # Find common phrases
        common_phrases = self.find_common_phrases(
            tokenized_sentences1, tokenized_sentences2
        )

        # Get word counts
        word_count1 = sum(len(sentence) for sentence in tokenized_sentences1)
        word_count2 = sum(len(sentence) for sentence in tokenized_sentences2)

        # Calculate overall similarity (balanced between document and sentence level)
        doc_weight = 0.5
        sentence_weight = 0.5
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

        # Compile results
        results = {
            "summary": {
                "text1_word_count": word_count1,
                "text2_word_count": word_count2,
                "text1_sentence_count": len(original_sentences1),
                "text2_sentence_count": len(original_sentences2),
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
            "common_phrases": common_phrases[:10],  # Limit to top 10 phrases
            "overall_similarity_percentage": similarity_percentage,
            "is_plagiarized": is_plagiarized,
            "potential_plagiarism": potential_plagiarism,
        }

        return results

    def detect_plagiarism_multi(self, texts):
        """Detect plagiarism between multiple texts using FastText"""
        start_time = time.time()
        text_count = len(texts)

        print(f"\n== FastText Module: Bắt đầu xử lý {text_count} văn bản ==")

        # Initialize results
        results = {
            "document_count": text_count,
            "document_similarities": [],
            "document_pairs": [],
            "is_plagiarized": False,  # Will be set to True if any document pair is plagiarized
            "potential_plagiarism": False,  # Will be set to True if any document pair has potential plagiarism
        }

        # Process all documents
        print(f"FastText: Đang tiền xử lý {text_count} văn bản...")
        original_sentences_list = []
        tokenized_sentences_list = []
        document_embeddings = []

        for i, text in enumerate(texts):
            print(f"  FastText: Đang xử lý văn bản {i+1}/{text_count}")
            original_sentences, tokenized_sentences, _ = self.preprocess_text(text)
            original_sentences_list.append(original_sentences)
            tokenized_sentences_list.append(tokenized_sentences)

            # Calculate document embedding
            doc_embedding = self.get_document_embedding(tokenized_sentences)
            document_embeddings.append(doc_embedding)

        print(f"FastText: Đã hoàn thành tiền xử lý văn bản")

        # Calculate total number of pairs
        total_pairs = text_count * (text_count - 1) // 2
        print(f"FastText: Bắt đầu so sánh {total_pairs} cặp văn bản...")

        # Compare all document pairs
        pair_count = 0
        for i in range(text_count):
            for j in range(i + 1, text_count):
                pair_count += 1
                if pair_count % 5 == 0 or pair_count == total_pairs:
                    print(
                        f"  FastText: Đang xử lý cặp {pair_count}/{total_pairs} ({round(pair_count/total_pairs*100, 1)}%) - Văn bản {i} & {j}"
                    )

                # Calculate document similarity
                doc_similarity = float(
                    cosine_similarity(
                        [document_embeddings[i]], [document_embeddings[j]]
                    )[0, 0]
                )

                # Generate sentence embeddings for documents i and j
                sentence_embeddings_i = [
                    self.get_sentence_embedding(sentence)
                    for sentence in tokenized_sentences_list[i]
                ]

                sentence_embeddings_j = [
                    self.get_sentence_embedding(sentence)
                    for sentence in tokenized_sentences_list[j]
                ]

                # Calculate sentence similarity matrix
                similarity_matrix = cosine_similarity(
                    sentence_embeddings_i, sentence_embeddings_j
                )

                # Find similar sentence pairs
                similar_pairs = []
                for i_idx in range(len(original_sentences_list[i])):
                    max_sim = np.max(similarity_matrix[i_idx])
                    j_idx = np.argmax(similarity_matrix[i_idx])

                    if max_sim > self.threshold:
                        similar_pairs.append(
                            {
                                "doc1_sentence": original_sentences_list[i][i_idx],
                                "doc2_sentence": original_sentences_list[j][j_idx],
                                "similarity_score": float(max_sim),
                                "similarity_percentage": round(float(max_sim) * 100, 1),
                            }
                        )

                # Calculate overall sentence similarity
                if similarity_matrix.size > 0:
                    max_similarities = np.max(similarity_matrix, axis=1)
                    sentence_similarity = float(np.mean(max_similarities))
                else:
                    sentence_similarity = 0.0

                # Calculate overall similarity (balanced between document and sentence level)
                doc_weight = 0.5
                sentence_weight = 0.5
                overall_similarity = (doc_weight * doc_similarity) + (
                    sentence_weight * sentence_similarity
                )

                # Calculate similarity percentage
                similarity_percentage = round(overall_similarity * 100, 2)

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

                # Add to results if significant similarity found
                if overall_similarity > 0.3 or len(similar_pairs) > 0:
                    # Store results
                    results["document_pairs"].append(
                        {
                            "doc1_index": i,
                            "doc2_index": j,
                            "document_similarity": round(doc_similarity, 3),
                            "sentence_similarity": round(sentence_similarity, 3),
                            "overall_similarity": overall_similarity,
                            "overall_similarity_percentage": similarity_percentage,
                            "similar_sentence_pairs": similar_pairs[
                                :10
                            ],  # Limit to top 10
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

        print(f"FastText: Đã hoàn thành so sánh {total_pairs} cặp văn bản")
        print(
            f"FastText: Thời gian thực hiện: {round(time.time() - start_time, 2)} giây"
        )

        return results


# Khởi tạo singleton detector
fasttext_detector = FastTextDetector()


# Function cho so sánh một cặp văn bản với FastText
def compare_texts_with_fasttext(text1: str, text2: str):
    """
    So sánh hai văn bản sử dụng FastText embeddings và trả về kết quả phân tích

    Args:
        text1: Văn bản thứ nhất
        text2: Văn bản thứ hai

    Returns:
        dict: Kết quả phân tích tương đồng
    """
    results = fasttext_detector.detect_plagiarism_pair(text1, text2)
    return results


# Function cho so sánh nhiều văn bản với FastText
def compare_multiple_texts_with_fasttext(texts: List[str]):
    """
    So sánh nhiều văn bản sử dụng FastText embeddings và trả về kết quả phân tích

    Args:
        texts: Danh sách các văn bản cần so sánh

    Returns:
        dict: Kết quả phân tích tương đồng
    """
    if len(texts) < 2:
        raise ValueError("Cần ít nhất 2 văn bản để so sánh")

    results = fasttext_detector.detect_plagiarism_multi(texts)
    return results
