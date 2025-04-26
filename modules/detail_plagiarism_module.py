from typing import List, Dict, Tuple, Any, Set, Optional
import time
import itertools
import numpy as np
from pydantic import BaseModel

from modules.lsa_lda_module import topic_detector
from modules.fasttext_module import fasttext_detector
from modules.bert_module import detector as bert_detector


class PlagiarizedSection:
    """Represents a detected plagiarized section between two documents"""

    def __init__(
        self,
        doc1_content: str,
        doc2_content: str,
        similarity_score: float,
        section_type: str,
        detection_layer: str,
    ):
        # We no longer need special handling for phrase sections since we don't store them
        self.doc1_content = doc1_content
        self.doc2_content = doc2_content
        self.similarity_score = similarity_score
        self.similarity_percentage = round(similarity_score * 100, 2)
        self.section_type = section_type  # "sentence", "paragraph", "phrase"
        self.detection_layer = detection_layer  # "lsa", "fasttext", "bert"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the plagiarized section to a dictionary representation"""
        return {
            "doc1_content": self.doc1_content,
            "doc2_content": self.doc2_content,
            "similarity_score": round(self.similarity_score, 3),
            "similarity_percentage": self.similarity_percentage,
            "section_type": self.section_type,
            "detection_layer": self.detection_layer,
        }


class DetailedDocumentPair:
    """Represents a pair of documents with detailed plagiarism information"""

    def __init__(
        self,
        doc1_filename: str,
        doc2_filename: str,
        doc1_index: int = -1,
        doc2_index: int = -1,
    ):
        self.doc1_filename = doc1_filename
        self.doc2_filename = doc2_filename
        self.doc1_index = doc1_index
        self.doc2_index = doc2_index

        # Full content of both documents
        self.doc1_content = ""
        self.doc2_content = ""

        # Similarity scores from each layer
        self.lsa_similarity = 0.0
        self.fasttext_similarity = 0.0
        self.bert_similarity = 0.0

        # Status flags for each layer
        self.passed_lsa = False
        self.passed_fasttext = False
        self.final_result = False

        # Detailed plagiarized sections from each layer
        self.lsa_sections: List[PlagiarizedSection] = []
        self.fasttext_sections: List[PlagiarizedSection] = []
        self.bert_sections: List[PlagiarizedSection] = []

        # Combined unique sections detected across all layers
        self.all_plagiarized_sections: List[PlagiarizedSection] = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert the document pair to a dictionary with detailed information"""
        return {
            "doc1_filename": self.doc1_filename,
            "doc2_filename": self.doc2_filename,
            "doc1_content": self.doc1_content,
            "doc2_content": self.doc2_content,
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
            "lsa_sections": [section.to_dict() for section in self.lsa_sections],
            "fasttext_sections": [
                section.to_dict() for section in self.fasttext_sections
            ],
            "bert_sections": [section.to_dict() for section in self.bert_sections],
            "all_plagiarized_sections": [
                section.to_dict() for section in self.all_plagiarized_sections
            ],
        }


class DetailedPlagiarismDetector:
    """
    An enhanced plagiarism detector that not only identifies plagiarism through
    three layers (LSA/LDA, FastText, BERT) but also provides detailed information
    about specific plagiarized sections.
    """

    def __init__(self):
        # Thresholds for each layer - same as the basic layered detector
        self.lsa_threshold = 0.3  # Lower threshold for initial filter
        self.fasttext_threshold = 0.4  # Medium threshold for second layer
        self.bert_threshold = 0.5  # Higher threshold for final layer

        # Section-level similarity thresholds (for individual sentences, phrases, etc.)
        self.lsa_section_threshold = 0.5  # Minimum similarity for LSA sections
        self.fasttext_section_threshold = (
            0.6  # Minimum similarity for FastText sections
        )
        self.bert_section_threshold = 0.7  # Minimum similarity for BERT sections

    def detect_plagiarism(self, doc_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect plagiarism across multiple texts using a layered approach, with detailed
        information about specific plagiarized sections.

        Args:
            doc_data: List of document data with "_id", "filename", "content", etc. fields

        Returns:
            Dict containing detailed results of plagiarism detection with specific sections
            identified at each layer
        """
        start_time = time.time()

        # Extract texts and filenames from the document data
        texts = [doc["content"] for doc in doc_data]
        filenames = [doc["filename"] for doc in doc_data]

        text_count = len(texts)

        if text_count < 2:
            raise ValueError("Cần ít nhất 2 văn bản để so sánh")

        print(f"Bắt đầu phân tích chi tiết {text_count} văn bản...")
        print(f"Tạo tất cả các cặp văn bản có thể...")

        # Generate all possible pairs
        all_pairs = list(itertools.combinations(range(text_count), 2))

        # Create detailed document pair objects for all possible pairs with filenames
        document_pairs = [
            DetailedDocumentPair(filenames[i], filenames[j], i, j) for i, j in all_pairs
        ]

        # Store the full document content in each pair
        for pair in document_pairs:
            pair.doc1_content = texts[pair.doc1_index]
            pair.doc2_content = texts[pair.doc2_index]

        # Track counts for summary
        initial_count = len(document_pairs)
        lsa_passed_count = 0
        fasttext_passed_count = 0
        bert_passed_count = 0

        # Layer 1: LSA/LDA Topic Modeling with detailed section extraction
        print(f"Layer 1: Bắt đầu xử lý {initial_count} cặp văn bản với LSA/LDA")
        document_pairs = self._apply_lsa_filter_detailed(texts, document_pairs)

        # Count how many passed LSA filter
        lsa_passed_count = sum(1 for pair in document_pairs if pair.passed_lsa)
        print(
            f"Layer 1: Hoàn thành - {lsa_passed_count}/{initial_count} cặp đã vượt qua bộ lọc LSA/LDA"
        )

        # Layer 2: FastText with detailed section extraction
        print(f"Layer 2: Bắt đầu xử lý cặp văn bản với FastText")
        document_pairs = self._apply_fasttext_filter_detailed(texts, document_pairs)

        # Count how many passed FastText filter
        fasttext_passed_count = sum(
            1 for pair in document_pairs if pair.passed_fasttext
        )
        print(
            f"Layer 2: Hoàn thành - {fasttext_passed_count}/{lsa_passed_count} cặp đã vượt qua bộ lọc FastText"
        )

        # Layer 3: BERT Analysis with detailed section extraction
        print(f"Layer 3: Bắt đầu xử lý cặp văn bản với BERT")
        document_pairs = self._apply_bert_analysis_detailed(texts, document_pairs)

        # Count how many are in the final result
        bert_passed_count = sum(1 for pair in document_pairs if pair.final_result)
        print(
            f"Layer 3: Hoàn thành - {bert_passed_count}/{fasttext_passed_count} cặp được xác định có khả năng đạo văn"
        )

        # Consolidate all plagiarized sections for each document pair
        print(f"Đang tổng hợp kết quả chi tiết...")
        document_pairs = self._consolidate_plagiarized_sections(document_pairs)
        print(f"Đã hoàn thành phân tích chi tiết tất cả các cặp văn bản")

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

    def _apply_lsa_filter_detailed(
        self, texts: List[str], document_pairs: List[DetailedDocumentPair]
    ) -> List[DetailedDocumentPair]:
        """Apply LSA/LDA topic modeling and extract detailed plagiarized sections"""
        # Use existing LSA multi-document comparison
        print(f"  Layer 1: Đang thực hiện phân tích LSA/LDA...")
        lsa_results = topic_detector.detect_plagiarism_multi(texts)
        print(
            f"  Layer 1: Đã hoàn thành phân tích LSA/LDA, đang cập nhật kết quả chi tiết cho từng cặp"
        )

        # Map for quick access to similarity scores and details from LSA results
        similarity_map = {}
        details_map = {}
        if lsa_results["document_pairs"]:
            for pair_result in lsa_results["document_pairs"]:
                doc1_idx = pair_result["doc1_index"]
                doc2_idx = pair_result["doc2_index"]
                similarity = pair_result["overall_similarity"]
                similarity_map[(doc1_idx, doc2_idx)] = similarity

                # Store relevant details for this pair
                details_map[(doc1_idx, doc2_idx)] = {
                    "similar_topics": pair_result.get("similar_topics", []),
                    "similarity_by_topic": pair_result.get("similarity_by_topic", []),
                }

        # Update all document pairs with LSA results
        count = 0
        total = len(document_pairs)
        print(
            f"  Layer 1: Đang cập nhật kết quả chi tiết LSA cho {total} cặp văn bản..."
        )

        for pair in document_pairs:
            count += 1
            if count % 50 == 0 or count == total:
                print(
                    f"  Layer 1: Đã xử lý {count}/{total} cặp văn bản ({round(count/total*100, 1)}%)"
                )

            pair_key = (pair.doc1_index, pair.doc2_index)
            if pair_key in similarity_map:
                pair.lsa_similarity = similarity_map[pair_key]

                # Extract detailed sections from LSA/LDA results
                if pair_key in details_map:
                    # For LSA, we'll use topic information as "sections"
                    topics = details_map[pair_key].get("similar_topics", [])
                    for topic in topics:
                        if (
                            topic.get("similarity_score", 0)
                            >= self.lsa_section_threshold
                        ):
                            # Create a plagiarized section for each significant similar topic
                            section = PlagiarizedSection(
                                doc1_content=f"Topic: {topic.get('topic_words', [])}",
                                doc2_content=f"Topic: {topic.get('topic_words', [])}",
                                similarity_score=topic.get("similarity_score", 0),
                                section_type="topic",
                                detection_layer="lsa",
                            )
                            pair.lsa_sections.append(section)

            # Check if it passes the LSA threshold
            pair.passed_lsa = pair.lsa_similarity >= self.lsa_threshold

            # For pairs that passed LSA but didn't extract specific sections, we'll analyze directly
            if pair.passed_lsa and not pair.lsa_sections:
                # Do a direct pair comparison for more details
                text1 = texts[pair.doc1_index]
                text2 = texts[pair.doc2_index]

                try:
                    # Use direct pair comparison for detailed section extraction
                    direct_result = topic_detector.detect_plagiarism_pair(text1, text2)

                    # Extract similar concepts or sentences
                    for concept in direct_result.get("similar_concepts", []):
                        if (
                            concept.get("similarity_score", 0)
                            >= self.lsa_section_threshold
                        ):
                            section = PlagiarizedSection(
                                doc1_content=concept.get("text1_concept", ""),
                                doc2_content=concept.get("text2_concept", ""),
                                similarity_score=concept.get("similarity_score", 0),
                                section_type="concept",
                                detection_layer="lsa",
                            )
                            pair.lsa_sections.append(section)
                except Exception as e:
                    # If direct comparison fails, continue with what we have
                    print(
                        f"  Layer 1: Warning - Could not extract detailed sections: {str(e)}"
                    )

        return document_pairs

    def _apply_fasttext_filter_detailed(
        self, texts: List[str], document_pairs: List[DetailedDocumentPair]
    ) -> List[DetailedDocumentPair]:
        """Apply FastText embedding analysis and extract detailed plagiarized sections"""
        # Only process pairs that passed the LSA filter
        lsa_passed = [pair for pair in document_pairs if pair.passed_lsa]
        total = len(lsa_passed)

        print(
            f"  Layer 2: Đang xử lý chi tiết {total} cặp văn bản đã vượt qua bộ lọc LSA..."
        )

        count = 0
        for pair in lsa_passed:
            count += 1
            if count % 10 == 0 or count == total:
                print(
                    f"  Layer 2: Đang xử lý cặp {count}/{total} ({round(count/total*100, 1)}%) - Văn bản {pair.doc1_filename} & {pair.doc2_filename}"
                )

            text1 = texts[pair.doc1_index]
            text2 = texts[pair.doc2_index]

            # Use existing FastText pair comparison
            result = fasttext_detector.detect_plagiarism_pair(text1, text2)
            pair.fasttext_similarity = result["overall_similarity_percentage"] / 100.0

            # Extract detailed plagiarized sections from FastText results

            # 1. Extract similar sentence pairs
            for sentence_pair in result.get("sentence_similarity", {}).get(
                "similar_sentence_pairs", []
            ):
                if (
                    sentence_pair.get("similarity_score", 0)
                    >= self.fasttext_section_threshold
                ):
                    section = PlagiarizedSection(
                        doc1_content=sentence_pair.get("text1_sentence", ""),
                        doc2_content=sentence_pair.get("text2_sentence", ""),
                        similarity_score=sentence_pair.get("similarity_score", 0),
                        section_type="sentence",
                        detection_layer="fasttext",
                    )
                    pair.fasttext_sections.append(section)

            # 2. We're skipping storing phrase sections as per optimization requirements
            # Original code for extracting common phrases has been removed

            # Check if it passes the FastText threshold
            pair.passed_fasttext = pair.fasttext_similarity >= self.fasttext_threshold

        # Mark remaining pairs as not passed
        for pair in document_pairs:
            if not pair.passed_lsa:
                pair.passed_fasttext = False

        return document_pairs

    def _apply_bert_analysis_detailed(
        self, texts: List[str], document_pairs: List[DetailedDocumentPair]
    ) -> List[DetailedDocumentPair]:
        """Apply BERT-based semantic analysis and extract detailed plagiarized sections"""
        # Process all pairs that passed the FastText filter
        fasttext_passed = [pair for pair in document_pairs if pair.passed_fasttext]
        total = len(fasttext_passed)

        print(
            f"  Layer 3: Đang xử lý chi tiết {total} cặp văn bản đã vượt qua bộ lọc FastText..."
        )

        count = 0
        for pair in fasttext_passed:
            count += 1
            if count % 5 == 0 or count == total:
                print(
                    f"  Layer 3: Đang xử lý cặp {count}/{total} ({round(count/total*100, 1)}%) - Văn bản {pair.doc1_filename} & {pair.doc2_filename}"
                )

            text1 = texts[pair.doc1_index]
            text2 = texts[pair.doc2_index]

            # Use existing BERT pair comparison
            result = bert_detector.detect_plagiarism_pair(text1, text2)
            pair.bert_similarity = result["overall_similarity_percentage"] / 100.0

            # Extract detailed plagiarized sections from BERT results

            # 1. Extract semantically similar sentence pairs
            for sentence_pair in result.get("semantic_similarity", {}).get(
                "similar_sentence_pairs", []
            ):
                if (
                    sentence_pair.get("similarity_score", 0)
                    >= self.bert_section_threshold
                ):
                    section = PlagiarizedSection(
                        doc1_content=sentence_pair.get("text1_sentence", ""),
                        doc2_content=sentence_pair.get("text2_sentence", ""),
                        similarity_score=sentence_pair.get("similarity_score", 0),
                        section_type="sentence",
                        detection_layer="bert",
                    )
                    pair.bert_sections.append(section)

            # 2. We're skipping storing phrase sections as per optimization requirements
            # Original code for extracting common phrases has been removed

            # Final result is determined by BERT analysis for those that passed FastText
            pair.final_result = pair.bert_similarity >= self.bert_threshold

        return document_pairs

    def _consolidate_plagiarized_sections(
        self, document_pairs: List[DetailedDocumentPair]
    ) -> List[DetailedDocumentPair]:
        """Consolidate all plagiarized sections from all layers into a unified list"""
        for pair in document_pairs:
            # Reset the consolidated list
            pair.all_plagiarized_sections = []

            # Helper function to check if a section is similar to existing ones
            def is_similar_to_existing(section: PlagiarizedSection) -> bool:
                for existing in pair.all_plagiarized_sections:
                    # Since we no longer store phrase sections, we only need to check regular sections
                    if (
                        section.doc1_content is not None
                        and section.doc2_content is not None
                        and existing.doc1_content is not None
                        and existing.doc2_content is not None
                    ):
                        overlap1 = self._get_text_overlap(
                            section.doc1_content, existing.doc1_content
                        )
                        overlap2 = self._get_text_overlap(
                            section.doc2_content, existing.doc2_content
                        )
                        if overlap1 > 0.7 or overlap2 > 0.7:
                            return True
                return False

            # Start with BERT sections (highest quality)
            for section in pair.bert_sections:
                pair.all_plagiarized_sections.append(section)

            # Add FastText sections if not similar to existing ones
            for section in pair.fasttext_sections:
                if not is_similar_to_existing(section):
                    pair.all_plagiarized_sections.append(section)

            # Add LSA sections if not similar to existing ones (topic sections from LSA are different type)
            for section in pair.lsa_sections:
                if section.section_type == "topic" or not is_similar_to_existing(
                    section
                ):
                    pair.all_plagiarized_sections.append(section)

            # Sort sections by similarity score (highest first)
            pair.all_plagiarized_sections.sort(
                key=lambda s: s.similarity_score, reverse=True
            )

        return document_pairs

    def _get_text_overlap(self, text1: str, text2: str) -> float:
        """Calculate simple text overlap ratio to determine if sections are redundant"""
        # Since we're no longer dealing with None values in our optimized code,
        # we can simplify this function. But we'll keep the check just for safety.
        if not text1 or not text2:
            return 0.0

        # Convert to sets of words for comparison
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Avoid division by zero
        if not words1 or not words2:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0


# Initialize the detector
detailed_detector = DetailedPlagiarismDetector()


def detect_plagiarism_detailed_with_metadata(
    doc_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Detect plagiarism using a layered approach with multiple algorithms, processing documents
    with metadata, and providing detailed information about specific plagiarized sections.

    Args:
        doc_data: List of document data with "_id", "filename", "content", etc. fields

    Returns:
        Dict containing detailed results of layered plagiarism detection with specific
        plagiarized sections identified at each layer
    """
    if len(doc_data) < 2:
        raise ValueError("Cần ít nhất 2 văn bản để so sánh")

    print(f"\n=== BẮT ĐẦU PHÂN TÍCH ĐẠO VĂN CHI TIẾT ({len(doc_data)} văn bản) ===\n")
    results = detailed_detector.detect_plagiarism(doc_data)
    print(
        f"\n=== HOÀN THÀNH PHÂN TÍCH ĐẠO VĂN CHI TIẾT - Thời gian: {results['execution_time_seconds']} giây ===\n"
    )
    return results
