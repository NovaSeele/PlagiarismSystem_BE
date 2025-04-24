from typing import List, Dict, Tuple, Any, Set
import time
import itertools
import numpy as np
from pydantic import BaseModel

from modules.lsa_lda_module import topic_detector
from modules.fasttext_module import fasttext_detector
from modules.bert_module import detector as bert_detector


class DocumentPair:
    """Represents a pair of documents with their indices and similarity score"""

    def __init__(self, doc1_index: int, doc2_index: int, similarity: float = 0.0):
        self.doc1_index = doc1_index
        self.doc2_index = doc2_index
        self.similarity = similarity
        self.similarity_percentage = round(similarity * 100, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the document pair to a dictionary representation"""
        return {
            "doc1_index": self.doc1_index,
            "doc2_index": self.doc2_index,
            "similarity_percentage": self.similarity_percentage,
        }


class DocumentPairDebug(DocumentPair):
    """Extended DocumentPair with debug information for each layer"""

    def __init__(self, doc1_index: int, doc2_index: int, similarity: float = 0.0):
        super().__init__(doc1_index, doc2_index, similarity)
        self.lsa_similarity = 0.0
        self.fasttext_similarity = 0.0
        self.bert_similarity = 0.0
        self.lsa_passed = False
        self.fasttext_passed = False
        self.bert_passed = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert the document pair debug info to a dictionary representation"""
        result = super().to_dict()
        result.update(
            {
                "lsa_similarity": round(self.lsa_similarity * 100, 2),
                "fasttext_similarity": round(self.fasttext_similarity * 100, 2),
                "bert_similarity": round(self.bert_similarity * 100, 2),
                "lsa_passed": self.lsa_passed,
                "fasttext_passed": self.fasttext_passed,
                "bert_passed": self.bert_passed,
            }
        )
        return result


class LayeredPlagiarismDetector:
    """
    A plagiarism detector that uses multiple algorithms in sequence, filtering at each step.

    The process works as follows:
    1. LSA/LDA (Topic Modeling) - Quick but less accurate initial filter
    2. FastText - More accurate semantic analysis for potential matches
    3. BERT - High precision semantic analysis for confirmed matches
    """

    def __init__(self):
        # Thresholds for each layer
        self.lsa_threshold = 0.4  # Lower threshold for initial filter
        self.fasttext_threshold = 0.5  # Medium threshold for second layer
        self.bert_threshold = 0.6  # Higher threshold for final layer

        # Plagiarism final threshold
        self.plagiarism_threshold = 0.75
        self.potential_plagiarism_threshold = 0.60

    def detect_plagiarism(self, texts: List[str]) -> Dict[str, Any]:
        """
        Detect plagiarism across multiple texts using a layered approach.

        Args:
            texts: List of texts to compare

        Returns:
            Dict containing results of plagiarism detection
        """
        start_time = time.time()
        text_count = len(texts)

        print(f"\nBắt đầu phân tích {text_count} văn bản bằng phương pháp phân lớp...")
        print(f"Tạo tất cả các cặp văn bản có thể...")

        # Generate all possible pairs
        all_pairs = list(itertools.combinations(range(text_count), 2))
        total_pairs = len(all_pairs)
        print(f"Đã tạo {total_pairs} cặp văn bản để phân tích")

        # Track which pairs pass each layer
        lsa_filtered_pairs = []
        fasttext_filtered_pairs = []
        final_results = []

        # Layer 1: LSA/LDA Topic Modeling Filter
        print(f"Layer 1: Bắt đầu xử lý {total_pairs} cặp văn bản với LSA/LDA")
        lsa_filtered_pairs = self._apply_lsa_filter(texts, all_pairs)
        lsa_filtered_count = len(lsa_filtered_pairs)
        print(
            f"Layer 1: Hoàn thành - {lsa_filtered_count}/{total_pairs} cặp đã vượt qua bộ lọc LSA/LDA"
        )

        # Layer 2: FastText Filter (only if pairs remain)
        if lsa_filtered_pairs:
            print(
                f"Layer 2: Bắt đầu xử lý {lsa_filtered_count} cặp văn bản với FastText"
            )
            fasttext_filtered_pairs = self._apply_fasttext_filter(
                texts, lsa_filtered_pairs
            )
            fasttext_filtered_count = len(fasttext_filtered_pairs)
            print(
                f"Layer 2: Hoàn thành - {fasttext_filtered_count}/{lsa_filtered_count} cặp đã vượt qua bộ lọc FastText"
            )

        # Layer 3: BERT Analysis (only if pairs remain)
        if fasttext_filtered_pairs:
            print(
                f"Layer 3: Bắt đầu xử lý {len(fasttext_filtered_pairs)} cặp văn bản với BERT"
            )
            final_results = self._apply_bert_analysis(texts, fasttext_filtered_pairs)
            print(
                f"Layer 3: Hoàn thành - {len(final_results)} cặp đã được phân tích bằng BERT"
            )

        # Prepare results
        results = {
            "document_count": text_count,
            "execution_time_seconds": round(time.time() - start_time, 2),
            "potential_plagiarism_pairs": [pair.to_dict() for pair in final_results],
            "is_plagiarized": any(
                pair.similarity >= self.plagiarism_threshold for pair in final_results
            ),
            "potential_plagiarism": any(
                self.potential_plagiarism_threshold
                <= pair.similarity
                < self.plagiarism_threshold
                for pair in final_results
            ),
        }

        print(f"Đã hoàn thành phân tích tất cả các cặp văn bản")
        print(f"Thời gian thực thi: {results['execution_time_seconds']} giây")

        return results

    def detect_plagiarism_debug(self, texts: List[str]) -> Dict[str, Any]:
        """
        Debug version of plagiarism detection that provides detailed information
        about each layer's results.

        Args:
            texts: List of texts to compare

        Returns:
            Dict containing detailed results of plagiarism detection including per-layer data
        """
        start_time = time.time()
        text_count = len(texts)

        print(f"\nBắt đầu phân tích chi tiết {text_count} văn bản...")
        print(f"Tạo tất cả các cặp văn bản có thể...")

        # Generate all possible pairs and create debug objects for each
        all_pairs = list(itertools.combinations(range(text_count), 2))
        all_debug_pairs = {
            (doc1, doc2): DocumentPairDebug(doc1, doc2) for doc1, doc2 in all_pairs
        }
        total_pairs = len(all_pairs)
        print(f"Đã tạo {total_pairs} cặp văn bản để phân tích")

        # Layer 1: LSA/LDA Topic Modeling with debug info
        print(f"Layer 1: Bắt đầu xử lý {total_pairs} cặp văn bản với LSA/LDA")
        print(f"  Layer 1: Đang thực hiện phân tích LSA/LDA...")
        lsa_results = topic_detector.detect_plagiarism_multi(texts)
        print(
            f"  Layer 1: Đã hoàn thành phân tích LSA/LDA, đang cập nhật kết quả cho từng cặp"
        )

        # Process LSA results and update debug pairs
        lsa_passed_pairs = []
        count = 0
        print(
            f"  Layer 1: Đang cập nhật kết quả cho {len(lsa_results['document_pairs'])} cặp văn bản..."
        )

        for pair_result in lsa_results["document_pairs"]:
            count += 1
            if count % 50 == 0 or count == len(lsa_results["document_pairs"]):
                print(
                    f"  Layer 1: Đã xử lý {count}/{len(lsa_results['document_pairs'])} cặp văn bản ({round(count/len(lsa_results['document_pairs'])*100, 1)}%)"
                )

            doc1_idx = pair_result["doc1_index"]
            doc2_idx = pair_result["doc2_index"]
            similarity = pair_result["overall_similarity"]

            # Update debug info
            debug_pair = all_debug_pairs.get(
                (doc1_idx, doc2_idx)
            ) or all_debug_pairs.get((doc2_idx, doc1_idx))
            if debug_pair:
                debug_pair.lsa_similarity = similarity
                debug_pair.lsa_passed = similarity >= self.lsa_threshold

                if debug_pair.lsa_passed:
                    lsa_passed_pairs.append((doc1_idx, doc2_idx))

        print(
            f"Layer 1: Hoàn thành - {len(lsa_passed_pairs)}/{total_pairs} cặp đã vượt qua bộ lọc LSA/LDA"
        )

        # Layer 2: FastText with debug info
        fasttext_passed_pairs = []
        if lsa_passed_pairs:
            print(
                f"Layer 2: Bắt đầu xử lý {len(lsa_passed_pairs)} cặp văn bản với FastText"
            )

            count = 0
            total = len(lsa_passed_pairs)
            for doc1_idx, doc2_idx in lsa_passed_pairs:
                count += 1
                if count % 10 == 0 or count == total:
                    print(
                        f"  Layer 2: Đang xử lý cặp {count}/{total} ({round(count/total*100, 1)}%) - Văn bản {doc1_idx} & {doc2_idx}"
                    )

                text1 = texts[doc1_idx]
                text2 = texts[doc2_idx]

                # Use existing FastText pair comparison
                result = fasttext_detector.detect_plagiarism_pair(text1, text2)
                similarity = result["overall_similarity_percentage"] / 100.0

                # Update debug info
                debug_pair = all_debug_pairs.get(
                    (doc1_idx, doc2_idx)
                ) or all_debug_pairs.get((doc2_idx, doc1_idx))
                if debug_pair:
                    debug_pair.fasttext_similarity = similarity
                    debug_pair.fasttext_passed = similarity >= self.fasttext_threshold

                    if debug_pair.fasttext_passed:
                        fasttext_passed_pairs.append((doc1_idx, doc2_idx))

        print(
            f"Layer 2: Hoàn thành - {len(fasttext_passed_pairs)}/{len(lsa_passed_pairs)} cặp đã vượt qua bộ lọc FastText"
        )

        # Layer 3: BERT Analysis with debug info
        if fasttext_passed_pairs:
            print(
                f"Layer 3: Bắt đầu xử lý {len(fasttext_passed_pairs)} cặp văn bản với BERT"
            )

            count = 0
            total = len(fasttext_passed_pairs)
            for doc1_idx, doc2_idx in fasttext_passed_pairs:
                count += 1
                if count % 5 == 0 or count == total:
                    print(
                        f"  Layer 3: Đang xử lý cặp {count}/{total} ({round(count/total*100, 1)}%) - Văn bản {doc1_idx} & {doc2_idx}"
                    )

                text1 = texts[doc1_idx]
                text2 = texts[doc2_idx]

                # Use existing BERT pair comparison
                result = bert_detector.detect_plagiarism_pair(text1, text2)
                similarity = result["overall_similarity_percentage"] / 100.0

                # Update debug info
                debug_pair = all_debug_pairs.get(
                    (doc1_idx, doc2_idx)
                ) or all_debug_pairs.get((doc2_idx, doc1_idx))
                if debug_pair:
                    debug_pair.bert_similarity = similarity
                    debug_pair.bert_passed = similarity >= self.bert_threshold
                    debug_pair.similarity = (
                        similarity  # Update overall similarity to BERT result
                    )

        # Count BERT passed pairs
        bert_passed_count = sum(
            1 for pair in all_debug_pairs.values() if pair.bert_passed
        )
        print(
            f"Layer 3: Hoàn thành - {bert_passed_count}/{len(fasttext_passed_pairs)} cặp được xác định có khả năng đạo văn"
        )

        # Collect all results for debug output
        # First get pairs that passed at least LSA layer
        passed_any_layer = [
            pair for pair in all_debug_pairs.values() if pair.lsa_passed
        ]

        # Sort by similarity (highest first)
        passed_any_layer.sort(key=lambda x: x.similarity, reverse=True)

        # Prepare detailed results
        results = {
            "document_count": text_count,
            "execution_time_seconds": round(time.time() - start_time, 2),
            "all_processed_pairs": [pair.to_dict() for pair in passed_any_layer],
            "is_plagiarized": any(
                pair.similarity >= self.plagiarism_threshold and pair.bert_passed
                for pair in passed_any_layer
            ),
            "potential_plagiarism": any(
                self.potential_plagiarism_threshold
                <= pair.similarity
                < self.plagiarism_threshold
                and pair.bert_passed
                for pair in passed_any_layer
            ),
            "layer_statistics": {
                "total_pairs": len(all_pairs),
                "lsa_passed_count": len(lsa_passed_pairs),
                "fasttext_passed_count": len(fasttext_passed_pairs),
                "bert_passed_count": bert_passed_count,
            },
        }

        print(f"Đã hoàn thành phân tích chi tiết tất cả các cặp văn bản")
        print(f"Thời gian thực thi: {results['execution_time_seconds']} giây")

        return results

    def _apply_lsa_filter(
        self, texts: List[str], pairs: List[Tuple[int, int]]
    ) -> List[DocumentPair]:
        """Apply LSA/LDA topic modeling as first filter"""
        # Use existing LSA multi-document comparison
        print(f"  Layer 1: Đang thực hiện phân tích LSA/LDA...")
        lsa_results = topic_detector.detect_plagiarism_multi(texts)
        print(f"  Layer 1: Đã hoàn thành phân tích LSA/LDA, đang xử lý kết quả...")

        # Extract document pairs that meet threshold
        filtered_pairs = []

        # Only proceed with pairs that have significant similarity
        if lsa_results["document_pairs"]:
            count = 0
            total = len(lsa_results["document_pairs"])
            for pair_result in lsa_results["document_pairs"]:
                count += 1
                if count % 50 == 0 or count == total:
                    print(
                        f"  Layer 1: Đã xử lý {count}/{total} cặp văn bản ({round(count/total*100, 1)}%)"
                    )

                doc1_idx = pair_result["doc1_index"]
                doc2_idx = pair_result["doc2_index"]
                similarity = pair_result["overall_similarity"]

                if similarity >= self.lsa_threshold:
                    filtered_pairs.append(DocumentPair(doc1_idx, doc2_idx, similarity))

        return filtered_pairs

    def _apply_fasttext_filter(
        self, texts: List[str], doc_pairs: List[DocumentPair]
    ) -> List[DocumentPair]:
        """Apply FastText embedding analysis as second filter"""
        filtered_pairs = []

        # For each pair that passed LSA, apply FastText
        count = 0
        total = len(doc_pairs)
        for pair in doc_pairs:
            count += 1
            if count % 10 == 0 or count == total:
                print(
                    f"  Layer 2: Đang xử lý cặp {count}/{total} ({round(count/total*100, 1)}%) - Văn bản {pair.doc1_index} & {pair.doc2_index}"
                )

            text1 = texts[pair.doc1_index]
            text2 = texts[pair.doc2_index]

            # Use existing FastText pair comparison
            result = fasttext_detector.detect_plagiarism_pair(text1, text2)
            similarity = result["overall_similarity_percentage"] / 100.0

            if similarity >= self.fasttext_threshold:
                # Update similarity with FastText result (more accurate than LSA)
                pair.similarity = similarity
                filtered_pairs.append(pair)

        return filtered_pairs

    def _apply_bert_analysis(
        self, texts: List[str], doc_pairs: List[DocumentPair]
    ) -> List[DocumentPair]:
        """Apply BERT-based semantic analysis as final analysis"""
        final_results = []

        # For each pair that passed FastText, apply BERT
        count = 0
        total = len(doc_pairs)
        for pair in doc_pairs:
            count += 1
            if count % 5 == 0 or count == total:
                print(
                    f"  Layer 3: Đang xử lý cặp {count}/{total} ({round(count/total*100, 1)}%) - Văn bản {pair.doc1_index} & {pair.doc2_index}"
                )

            text1 = texts[pair.doc1_index]
            text2 = texts[pair.doc2_index]

            # Use existing BERT pair comparison
            result = bert_detector.detect_plagiarism_pair(text1, text2)
            similarity = result["overall_similarity_percentage"] / 100.0

            # Always include results from BERT, but update similarity
            pair.similarity = similarity
            final_results.append(pair)

        # Sort by similarity (highest first)
        final_results.sort(key=lambda x: x.similarity, reverse=True)
        return final_results


# Initialize the layered detector
layered_detector = LayeredPlagiarismDetector()


def detect_plagiarism_layered(texts: List[str]) -> Dict[str, Any]:
    """
    Detect plagiarism using a layered approach with multiple algorithms.

    Args:
        texts: List of texts to compare

    Returns:
        Dict containing results of layered plagiarism detection
    """
    if len(texts) < 2:
        raise ValueError("Cần ít nhất 2 văn bản để so sánh")

    print(f"\n=== BẮT ĐẦU PHÂN TÍCH ĐẠO VĂN ({len(texts)} văn bản) ===\n")
    results = layered_detector.detect_plagiarism(texts)
    print(
        f"\n=== HOÀN THÀNH PHÂN TÍCH ĐẠO VĂN - Thời gian: {results['execution_time_seconds']} giây ===\n"
    )
    return results


def detect_plagiarism_debug(texts: List[str]) -> Dict[str, Any]:
    """
    Debug version of the layered plagiarism detection that provides detailed
    information about each layer's results.

    Args:
        texts: List of texts to compare

    Returns:
        Dict containing detailed results including per-layer data
    """
    if len(texts) < 2:
        raise ValueError("Cần ít nhất 2 văn bản để so sánh")

    print(f"\n=== BẮT ĐẦU PHÂN TÍCH CHI TIẾT ĐẠO VĂN ({len(texts)} văn bản) ===\n")
    results = layered_detector.detect_plagiarism_debug(texts)
    print(
        f"\n=== HOÀN THÀNH PHÂN TÍCH CHI TIẾT ĐẠO VĂN - Thời gian: {results['execution_time_seconds']} giây ===\n"
    )
    return results
