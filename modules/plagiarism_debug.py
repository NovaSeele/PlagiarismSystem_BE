from typing import List, Dict, Tuple, Any, Set
import time
import itertools
import numpy as np
from pydantic import BaseModel

from modules.lsa_lda_module import topic_detector
from modules.fasttext_module import fasttext_detector
from modules.bert_module import detector as bert_detector


class DocumentPairDebug:
    """Represents a pair of documents with their indices and similarity scores from each layer"""

    def __init__(self, doc1_index: int, doc2_index: int):
        self.doc1_index = doc1_index
        self.doc2_index = doc2_index

        # Similarity scores from each layer
        self.lsa_similarity = 0.3
        self.fasttext_similarity = 0.4
        self.bert_similarity = 0.5

        # Status flags for each layer (whether they passed the filter)
        self.passed_lsa = False
        self.passed_fasttext = False
        self.final_result = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert the document pair to a dictionary representation with all debug information"""
        return {
            "doc1_index": self.doc1_index,
            "doc2_index": self.doc2_index,
            "lsa_similarity_percentage": round(self.lsa_similarity * 100, 2),
            "passed_lsa_filter": self.passed_lsa,
            "fasttext_similarity_percentage": (
                round(self.fasttext_similarity * 100, 2)
                if self.fasttext_similarity > 0
                else None
            ),
            "passed_fasttext_filter": self.passed_fasttext,
            "bert_similarity_percentage": (
                round(self.bert_similarity * 100, 2)
                if self.bert_similarity > 0
                else None
            ),
            "final_result": self.final_result,
        }


class LayeredPlagiarismDetectorDebug:
    """
    A debug version of the plagiarism detector that shows all document pairs and their progression
    through each layer of the detection process.

    The process works as follows:
    1. LSA/LDA (Topic Modeling) - Quick but less accurate initial filter
    2. FastText - More accurate semantic analysis for potential matches
    3. BERT - High precision semantic analysis for confirmed matches
    """

    def __init__(self):
        # Thresholds for each layer
        self.lsa_threshold = 0.3  # Lower threshold for initial filter
        self.fasttext_threshold = 0.4  # Medium threshold for second layer
        self.bert_threshold = 0.5  # Higher threshold for final layer

    def detect_plagiarism_debug(self, texts: List[str]) -> Dict[str, Any]:
        """
        Detect plagiarism across multiple texts using a layered approach, with detailed debug information.

        Args:
            texts: List of texts to compare

        Returns:
            Dict containing detailed results of plagiarism detection with progression through each layer
        """
        start_time = time.time()
        text_count = len(texts)

        # Generate all possible pairs
        all_pairs = list(itertools.combinations(range(text_count), 2))

        # Create document pair objects for all possible pairs
        document_pairs = [DocumentPairDebug(i, j) for i, j in all_pairs]

        # Track counts for summary
        initial_count = len(document_pairs)
        lsa_passed_count = 0
        fasttext_passed_count = 0
        bert_passed_count = 0

        # Layer 1: LSA/LDA Topic Modeling
        print(f"Layer 1: Processing {initial_count} document pairs with LSA/LDA")
        document_pairs = self._apply_lsa_filter_debug(texts, document_pairs)

        # Count how many passed LSA filter
        lsa_passed_count = sum(1 for pair in document_pairs if pair.passed_lsa)
        print(f"Layer 1: {lsa_passed_count} pairs passed LSA/LDA filter")

        # Layer 2: FastText (apply to all pairs, but only those that passed LSA will proceed)
        print(f"Layer 2: Processing document pairs with FastText")
        document_pairs = self._apply_fasttext_filter_debug(texts, document_pairs)

        # Count how many passed FastText filter
        fasttext_passed_count = sum(
            1 for pair in document_pairs if pair.passed_fasttext
        )
        print(f"Layer 2: {fasttext_passed_count} pairs passed FastText filter")

        # Layer 3: BERT Analysis (apply to all pairs, but only those that passed FastText will be considered as final results)
        print(f"Layer 3: Processing document pairs with BERT")
        document_pairs = self._apply_bert_analysis_debug(texts, document_pairs)

        # Count how many are in the final result
        bert_passed_count = sum(1 for pair in document_pairs if pair.final_result)
        print(
            f"Layer 3: {bert_passed_count} pairs identified as potentially plagiarized"
        )

        # Prepare results
        results = {
            "document_count": text_count,
            "execution_time_seconds": round(time.time() - start_time, 2),
            "summary": {
                "total_pairs": initial_count,
                "lsa_passed_count": lsa_passed_count,
                "fasttext_passed_count": fasttext_passed_count,
                "final_result_count": bert_passed_count,
            },
            "all_document_pairs": [pair.to_dict() for pair in document_pairs],
        }

        return results

    def _apply_lsa_filter_debug(
        self, texts: List[str], document_pairs: List[DocumentPairDebug]
    ) -> List[DocumentPairDebug]:
        """Apply LSA/LDA topic modeling and update all document pairs with results"""
        # Use existing LSA multi-document comparison
        lsa_results = topic_detector.detect_plagiarism_multi(texts)

        # Map for quick access to similarity scores from LSA results
        similarity_map = {}
        if lsa_results["document_pairs"]:
            for pair_result in lsa_results["document_pairs"]:
                doc1_idx = pair_result["doc1_index"]
                doc2_idx = pair_result["doc2_index"]
                similarity = pair_result["overall_similarity"]
                similarity_map[(doc1_idx, doc2_idx)] = similarity

        # Update all document pairs with LSA results
        for pair in document_pairs:
            pair_key = (pair.doc1_index, pair.doc2_index)
            if pair_key in similarity_map:
                pair.lsa_similarity = similarity_map[pair_key]

            # Check if it passes the LSA threshold
            pair.passed_lsa = pair.lsa_similarity >= self.lsa_threshold

        return document_pairs

    def _apply_fasttext_filter_debug(
        self, texts: List[str], document_pairs: List[DocumentPairDebug]
    ) -> List[DocumentPairDebug]:
        """Apply FastText embedding analysis to document pairs"""
        # Only process pairs that passed the LSA filter
        for pair in document_pairs:
            # Only process if passed the previous filter
            if pair.passed_lsa:
                text1 = texts[pair.doc1_index]
                text2 = texts[pair.doc2_index]

                # Use existing FastText pair comparison
                result = fasttext_detector.detect_plagiarism_pair(text1, text2)
                pair.fasttext_similarity = (
                    result["overall_similarity_percentage"] / 100.0
                )

                # Check if it passes the FastText threshold
                pair.passed_fasttext = (
                    pair.fasttext_similarity >= self.fasttext_threshold
                )
            else:
                # Mark as not passed FastText if it didn't pass LSA
                pair.passed_fasttext = False

        return document_pairs

    def _apply_bert_analysis_debug(
        self, texts: List[str], document_pairs: List[DocumentPairDebug]
    ) -> List[DocumentPairDebug]:
        """Apply BERT-based semantic analysis to document pairs"""
        # Process all pairs that passed the FastText filter
        for pair in document_pairs:
            # Only process if passed the previous filter
            if pair.passed_fasttext:
                text1 = texts[pair.doc1_index]
                text2 = texts[pair.doc2_index]

                # Use existing BERT pair comparison
                result = bert_detector.detect_plagiarism_pair(text1, text2)
                pair.bert_similarity = result["overall_similarity_percentage"] / 100.0

                # Final result is determined by BERT analysis for those that passed FastText
                pair.final_result = pair.bert_similarity >= self.bert_threshold

        return document_pairs


# Initialize the debug detector
layered_detector_debug = LayeredPlagiarismDetectorDebug()


def detect_plagiarism_layered_debug(texts: List[str]) -> Dict[str, Any]:
    """
    Detect plagiarism using a layered approach with multiple algorithms, with detailed debug information.

    Args:
        texts: List of texts to compare

    Returns:
        Dict containing detailed results of layered plagiarism detection with progression through each layer
    """
    if len(texts) < 2:
        raise ValueError("Cần ít nhất 2 văn bản để so sánh")

    results = layered_detector_debug.detect_plagiarism_debug(texts)
    return results
