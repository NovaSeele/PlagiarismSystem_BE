from typing import List, Dict, Tuple, Any, Set, Optional
import time
import itertools
import numpy as np
import asyncio
import json
from fastapi import WebSocket
from pydantic import BaseModel

from modules.lsa_lda_module import topic_detector
from modules.fasttext_module import fasttext_detector
from modules.bert_module import detector as bert_detector

class DocumentPair:
    """Represents a pair of documents with their filenames and similarity scores from each layer"""

    def __init__(
        self,
        doc1_filename: str,
        doc2_filename: str,
        doc1_index: int = -1,
        doc2_index: int = -1,
    ):
        self.doc1_filename = doc1_filename
        self.doc2_filename = doc2_filename
        self.doc1_index = doc1_index  # Keep index for internal reference
        self.doc2_index = doc2_index  # Keep index for internal reference

        # Similarity scores from each layer
        self.lsa_similarity = 0.0
        self.fasttext_similarity = 0.0
        self.bert_similarity = 0.0

        # Status flags for each layer (whether they passed the filter)
        self.passed_lsa = False
        self.passed_fasttext = False
        self.final_result = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert the document pair to a dictionary representation with all debug information"""
        return {
            "doc1_filename": self.doc1_filename,
            "doc2_filename": self.doc2_filename,
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


class LayeredPlagiarismDetector:
    """
    A plagiarism detector that shows all document pairs and their progression
    through each layer of the detection process.

    The process works as follows:
    1. LSA/LDA (Topic Modeling) - Quick but less accurate initial filter
    2. FastText - More accurate semantic analysis for potential matches
    3. BERT - High precision semantic analysis for confirmed matches
    """

    def __init__(self):
        # Thresholds for each layer
        # LSA threshold (0.3) - First layer filter
        # This is a low threshold used to quickly filter out obviously dissimilar documents.
        self.lsa_threshold = 0.3  # Lower threshold for initial filter

        # FastText threshold (0.4) - Second layer filter
        # This medium threshold applies to document pairs that passed the LSA filter.
        self.fasttext_threshold = 0.4  # Medium threshold for second layer

        # BERT threshold (0.5) - Final layer filter
        # This higher threshold determines which document pairs are included in the final results.
        self.bert_threshold = 0.5  # Higher threshold for final layer

    async def send_progress_update(self, websockets, message):
        """Send progress update to all connected WebSocket clients"""
        if not websockets:
            # Just print to console if no websockets are connected
            print(message)
            return

        if isinstance(websockets, set):
            # Tạo danh sách các websocket cần xóa
            closed_websockets = set()

            for websocket in websockets:
                try:
                    # Kiểm tra trạng thái kết nối trước khi gửi
                    if (
                        hasattr(websocket, "client_state")
                        and websocket.client_state.name == "CONNECTED"
                    ):
                        try:
                            await websocket.send_text(message)
                        except RuntimeError as e:
                            if (
                                "Cannot call 'send' once a close message has been sent"
                                in str(e)
                            ):
                                closed_websockets.add(websocket)
                            else:
                                raise
                    else:
                        closed_websockets.add(websocket)
                except Exception as e:
                    print(f"Error sending to websocket: {e}")
                    # Đánh dấu websocket này để xóa khỏi danh sách
                    closed_websockets.add(websocket)

            # Xóa các websocket đã đóng khỏi tập hợp
            for closed_ws in closed_websockets:
                if closed_ws in websockets:
                    websockets.remove(closed_ws)

        print(message)  # Also print to console for logging

    async def detect_plagiarism(
        self, doc_data: List[Dict[str, Any]], websockets=None
    ) -> Dict[str, Any]:
        """
        Detect plagiarism across multiple texts using a layered approach, with detailed information.

        Args:
            doc_data: List of document data with "_id", "filename", "content", etc. fields
            websockets: Set of active WebSocket connections

        Returns:
            Dict containing detailed results of plagiarism detection with progression through each layer
        """
        start_time = time.time()

        # Extract texts and filenames from the document data
        texts = [doc["content"] for doc in doc_data]
        filenames = [doc["filename"] for doc in doc_data]

        text_count = len(texts)

        if text_count < 2:
            raise ValueError("Cần ít nhất 2 văn bản để so sánh")

        await self.send_progress_update(
            websockets, f"Bắt đầu phân tích {text_count} văn bản..."
        )
        await self.send_progress_update(
            websockets, f"Tạo tất cả các cặp văn bản có thể..."
        )

        # Generate all possible pairs
        all_pairs = list(itertools.combinations(range(text_count), 2))

        # Create document pair objects for all possible pairs with filenames
        document_pairs = [
            DocumentPair(filenames[i], filenames[j], i, j) for i, j in all_pairs
        ]

        # Track counts for summary
        initial_count = len(document_pairs)
        lsa_passed_count = 0
        fasttext_passed_count = 0
        bert_passed_count = 0

        # Layer 1: LSA/LDA Topic Modeling
        await self.send_progress_update(
            websockets,
            f"Layer 1: Bắt đầu xử lý {initial_count} cặp văn bản với LSA/LDA",
        )
        document_pairs = await self._apply_lsa_filter(texts, document_pairs, websockets)

        # Count how many passed LSA filter
        lsa_passed_count = sum(1 for pair in document_pairs if pair.passed_lsa)
        await self.send_progress_update(
            websockets,
            f"Layer 1: Hoàn thành - {lsa_passed_count}/{initial_count} cặp đã vượt qua bộ lọc LSA/LDA",
        )

        # Layer 2: FastText (apply to all pairs, but only those that passed LSA will proceed)
        await self.send_progress_update(
            websockets, f"Layer 2: Bắt đầu xử lý cặp văn bản với FastText"
        )
        document_pairs = await self._apply_fasttext_filter(
            texts, document_pairs, websockets
        )

        # Count how many passed FastText filter
        fasttext_passed_count = sum(
            1 for pair in document_pairs if pair.passed_fasttext
        )
        await self.send_progress_update(
            websockets,
            f"Layer 2: Hoàn thành - {fasttext_passed_count}/{lsa_passed_count} cặp đã vượt qua bộ lọc FastText",
        )

        # Layer 3: BERT Analysis (apply to all pairs, but only those that passed FastText will be considered as final results)
        await self.send_progress_update(
            websockets, f"Layer 3: Bắt đầu xử lý cặp văn bản với BERT"
        )
        document_pairs = await self._apply_bert_analysis(
            texts, document_pairs, websockets
        )

        # Count how many are in the final result
        bert_passed_count = sum(1 for pair in document_pairs if pair.final_result)
        await self.send_progress_update(
            websockets,
            f"Layer 3: Hoàn thành - {bert_passed_count}/{fasttext_passed_count} cặp được xác định có khả năng đạo văn",
        )

        await self.send_progress_update(
            websockets, f"Đã hoàn thành phân tích tất cả các cặp văn bản"
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

    async def _apply_lsa_filter(
        self, texts: List[str], document_pairs: List[DocumentPair], websockets=None
    ) -> List[DocumentPair]:
        """Apply LSA/LDA topic modeling and update all document pairs with results"""
        # Use existing LSA multi-document comparison
        await self.send_progress_update(
            websockets, f"  Layer 1: Đang thực hiện phân tích LSA/LDA..."
        )
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
        count = 0
        total = len(document_pairs)

        for pair in document_pairs:
            count += 1
            if count % 1 == 0 or count == total:
                await self.send_progress_update(
                    websockets,
                    f"  Layer 1: Đã xử lý {count}/{total} cặp văn bản ({round(count/total*100, 1)}%)",
                )

            pair_key = (pair.doc1_index, pair.doc2_index)
            if pair_key in similarity_map:
                pair.lsa_similarity = similarity_map[pair_key]

            # Check if it passes the LSA threshold
            pair.passed_lsa = pair.lsa_similarity >= self.lsa_threshold

        return document_pairs

    async def _apply_fasttext_filter(
        self, texts: List[str], document_pairs: List[DocumentPair], websockets=None
    ) -> List[DocumentPair]:
        """Apply FastText embedding analysis to document pairs"""
        # Only process pairs that passed the LSA filter
        lsa_passed = [pair for pair in document_pairs if pair.passed_lsa]
        total = len(lsa_passed)

        await self.send_progress_update(
            websockets,
            f"  Layer 2: Đang xử lý {total} cặp văn bản đã vượt qua bộ lọc LSA...",
        )

        count = 0
        for pair in lsa_passed:
            count += 1
            if count % 1 == 0 or count == total:
                await self.send_progress_update(
                    websockets,
                    f"  Layer 2: Đang xử lý cặp {count}/{total} ({round(count/total*100, 1)}%)",
                )

            text1 = texts[pair.doc1_index]
            text2 = texts[pair.doc2_index]

            # Use existing FastText pair comparison
            result = fasttext_detector.detect_plagiarism_pair(text1, text2)
            pair.fasttext_similarity = result["overall_similarity_percentage"] / 100.0

            # Check if it passes the FastText threshold
            pair.passed_fasttext = pair.fasttext_similarity >= self.fasttext_threshold

        # Mark remaining pairs as not passed
        for pair in document_pairs:
            if not pair.passed_lsa:
                pair.passed_fasttext = False

        return document_pairs

    async def _apply_bert_analysis(
        self, texts: List[str], document_pairs: List[DocumentPair], websockets=None
    ) -> List[DocumentPair]:
        """Apply BERT-based semantic analysis to document pairs"""
        # Process all pairs that passed the FastText filter
        fasttext_passed = [pair for pair in document_pairs if pair.passed_fasttext]
        total = len(fasttext_passed)

        await self.send_progress_update(
            websockets,
            f"  Layer 3: Đang xử lý {total} cặp văn bản đã vượt qua bộ lọc FastText...",
        )

        count = 0
        for pair in fasttext_passed:
            count += 1
            if count % 1 == 0 or count == total:
                await self.send_progress_update(
                    websockets,
                    f"  Layer 3: Đang xử lý cặp {count}/{total} ({round(count/total*100, 1)}%)",
                )

            text1 = texts[pair.doc1_index]
            text2 = texts[pair.doc2_index]

            # Use existing BERT pair comparison
            result = bert_detector.detect_plagiarism_pair(text1, text2)
            pair.bert_similarity = result["overall_similarity_percentage"] / 100.0

            # Final result is determined by BERT analysis for those that passed FastText
            pair.final_result = pair.bert_similarity >= self.bert_threshold

        return document_pairs


# Initialize the detector
layered_detector = LayeredPlagiarismDetector()


async def detect_plagiarism_layered_with_metadata(
    doc_data: List[Dict[str, Any]], websockets=None
) -> Dict[str, Any]:
    """
    Detect plagiarism using a layered approach with multiple algorithms, processing documents with metadata.

    Args:
        doc_data: List of document data with "_id", "filename", "content", etc. fields
        websockets: Set of active WebSocket connections to send progress updates

    Returns:
        Dict containing detailed results of layered plagiarism detection with progression through each layer
    """
    if len(doc_data) < 2:
        raise ValueError("Cần ít nhất 2 văn bản để so sánh")

    start_message = f"\n=== BẮT ĐẦU PHÂN TÍCH ĐẠO VĂN ({len(doc_data)} văn bản) ===\n"
    await layered_detector.send_progress_update(websockets, start_message)

    results = await layered_detector.detect_plagiarism(doc_data, websockets)

    end_message = f"\n=== HOÀN THÀNH PHÂN TÍCH ĐẠO VĂN - Thời gian: {results['execution_time_seconds']} giây ===\n"
    await layered_detector.send_progress_update(websockets, end_message)

    return results
