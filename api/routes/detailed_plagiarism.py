from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json
import time
import os

from services.detailed_plagiarism import (
    initialize_plagiarism_service,
    check_plagiarism,
    check_multiple_texts,
    check_cross_plagiarism,
)
from services.pdf_metadata import get_all_pdf_metadata


# Define models for request and response
class PlagiarismRequest(BaseModel):
    source_texts: List[str]
    suspect_texts: List[str]
    topic_threshold: Optional[float] = 0.6
    semantic_threshold: Optional[float] = 0.7
    bert_threshold: Optional[float] = 0.5
    fasttext_path: Optional[str] = "cc.en.300.bin.gz"
    bert_model_name: Optional[str] = "jpwahle/longformer-base-plagiarism-detection"


class SingleTextRequest(BaseModel):
    source_texts: List[str]
    suspect_text: str
    topic_threshold: Optional[float] = 0.6
    semantic_threshold: Optional[float] = 0.7
    bert_threshold: Optional[float] = 0.5
    fasttext_path: Optional[str] = "cc.en.300.bin.gz"
    bert_model_name: Optional[str] = "jpwahle/longformer-base-plagiarism-detection"


class CrossPlagiarismRequest(BaseModel):
    topic_threshold: Optional[float] = 0.6
    semantic_threshold: Optional[float] = 0.7
    bert_threshold: Optional[float] = 0.5
    fasttext_path: Optional[str] = "cc.en.300.bin.gz"
    bert_model_name: Optional[str] = "jpwahle/longformer-base-plagiarism-detection"


# Create router
router = APIRouter()

# Store initialized services to avoid reinitializing
service_cache = {}


def get_service_key(source_texts, fasttext_path, bert_model_name):
    """Generate a unique key for the service cache"""
    # Use a hash of the source texts and model paths
    return f"{hash(tuple(source_texts))}-{fasttext_path}-{bert_model_name}"


def get_or_create_service(source_texts, fasttext_path, bert_model_name):
    """Get an existing service or create a new one"""
    key = get_service_key(source_texts, fasttext_path, bert_model_name)

    if key not in service_cache:
        print(f"Initializing new plagiarism service...")

        # Check if FastText model exists and try alternative paths
        if not os.path.exists(fasttext_path) and not os.path.isabs(fasttext_path):
            # Try looking in common directories
            possible_paths = [
                fasttext_path,
                os.path.join("models", fasttext_path),
                os.path.join("data", "models", fasttext_path),
                os.path.join("..", "models", fasttext_path),
                os.path.join(
                    os.path.dirname(__file__), "..", "..", "models", fasttext_path
                ),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    fasttext_path = path
                    print(f"Found FastText model at {fasttext_path}")
                    break

        service_cache[key] = initialize_plagiarism_service(
            source_texts, fasttext_path, bert_model_name
        )
    else:
        print(f"Using cached plagiarism service...")

    return service_cache[key]


@router.post("/detailed_plagiarism")
async def detailed_plagiarism(request: PlagiarismRequest):
    """
    Check multiple suspect texts for plagiarism against source texts

    Args:
        request: PlagiarismRequest containing source and suspect texts

    Returns:
        dict: Plagiarism detection results
    """
    try:
        start_time = time.time()

        # Check if FastText model exists
        fasttext_path = request.fasttext_path
        model_found = os.path.exists(fasttext_path)

        if not model_found and not os.path.isabs(fasttext_path):
            # Try looking in common directories
            possible_paths = [
                fasttext_path,
                os.path.join("models", fasttext_path),
                os.path.join("data", "models", fasttext_path),
                os.path.join("..", "models", fasttext_path),
                os.path.join(
                    os.path.dirname(__file__), "..", "..", "models", fasttext_path
                ),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    fasttext_path = path
                    model_found = True
                    print(f"Found FastText model at {fasttext_path}")
                    break

        if not model_found:
            print(
                f"Warning: FastText model not found at {fasttext_path}. Using WordNet only."
            )

        # Get or create service
        service = get_or_create_service(
            request.source_texts, fasttext_path, request.bert_model_name
        )

        # Check for plagiarism
        results = check_multiple_texts(
            service,
            request.suspect_texts,
            topic_threshold=request.topic_threshold,
            semantic_threshold=request.semantic_threshold,
            bert_threshold=request.bert_threshold,
        )

        # Calculate processing time
        processing_time = time.time() - start_time

        # Return results
        return {
            "status": "success",
            "message": f"Processed {len(request.suspect_texts)} suspect texts in {processing_time:.2f} seconds",
            "processing_time": processing_time,
            "results": results,
            "note": (
                "FastText model not found, using WordNet only for semantic similarity"
                if not model_found
                else ""
            ),
        }

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}\n{error_details}",
        )


@router.post("/check_single_text")
async def check_single_text(request: SingleTextRequest):
    """
    Check a single suspect text for plagiarism against source texts

    Args:
        request: SingleTextRequest containing source texts and a suspect text

    Returns:
        dict: Plagiarism detection results
    """
    try:
        start_time = time.time()

        # Check if FastText model exists
        fasttext_path = request.fasttext_path
        model_found = os.path.exists(fasttext_path)

        if not model_found and not os.path.isabs(fasttext_path):
            # Try looking in common directories
            possible_paths = [
                fasttext_path,
                os.path.join("models", fasttext_path),
                os.path.join("data", "models", fasttext_path),
                os.path.join("..", "models", fasttext_path),
                os.path.join(
                    os.path.dirname(__file__), "..", "..", "models", fasttext_path
                ),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    fasttext_path = path
                    model_found = True
                    print(f"Found FastText model at {fasttext_path}")
                    break

        if not model_found:
            print(
                f"Warning: FastText model not found at {fasttext_path}. Using WordNet only."
            )

        # Get or create service
        service = get_or_create_service(
            request.source_texts, fasttext_path, request.bert_model_name
        )

        # Check for plagiarism
        results = check_plagiarism(
            service,
            request.suspect_text,
            topic_threshold=request.topic_threshold,
            semantic_threshold=request.semantic_threshold,
            bert_threshold=request.bert_threshold,
        )

        # Calculate processing time
        processing_time = time.time() - start_time

        # Return results
        return {
            "status": "success",
            "message": f"Processed suspect text in {processing_time:.2f} seconds",
            "processing_time": processing_time,
            "results": results,
            "note": (
                "FastText model not found, using WordNet only for semantic similarity"
                if not model_found
                else ""
            ),
        }

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}\n{error_details}",
        )


@router.post("/clear_service_cache")
async def clear_service_cache():
    """
    Clear the service cache to free up memory

    Returns:
        dict: Status message
    """
    global service_cache
    cache_size = len(service_cache)
    service_cache = {}

    return {"status": "success", "message": f"Cleared {cache_size} services from cache"}


@router.post("/check_cross_plagiarism")
async def check_all_documents_plagiarism(request: CrossPlagiarismRequest):
    """
    Check for plagiarism across all uploaded PDF documents

    Args:
        request: CrossPlagiarismRequest containing threshold parameters

    Returns:
        dict: Cross-plagiarism detection results
    """
    try:
        start_time = time.time()

        # Get all PDF contents
        all_texts = get_all_pdf_metadata()

        if not all_texts or len(all_texts) < 2:
            return {
                "status": "warning",
                "message": "Need at least 2 documents to check for cross-plagiarism",
                "results": [],
            }

        # Check if FastText model exists
        fasttext_path = request.fasttext_path
        model_found = os.path.exists(fasttext_path)

        if not model_found and not os.path.isabs(fasttext_path):
            # Try looking in common directories
            possible_paths = [
                fasttext_path,
                os.path.join("models", fasttext_path),
                os.path.join("data", "models", fasttext_path),
                os.path.join("..", "models", fasttext_path),
                os.path.join(
                    os.path.dirname(__file__), "..", "..", "models", fasttext_path
                ),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    fasttext_path = path
                    model_found = True
                    print(f"Found FastText model at {fasttext_path}")
                    break

        if not model_found:
            print(
                f"Warning: FastText model not found at {fasttext_path}. Using WordNet only."
            )

        # Check for cross-plagiarism
        results = check_cross_plagiarism(
            all_texts,
            topic_threshold=request.topic_threshold,
            semantic_threshold=request.semantic_threshold,
            bert_threshold=request.bert_threshold,
            fasttext_path=fasttext_path,
            bert_model_name=request.bert_model_name,
        )

        # Calculate processing time
        processing_time = time.time() - start_time

        # Return results
        return {
            "status": "success",
            "message": f"Processed {len(all_texts)} documents for cross-plagiarism in {processing_time:.2f} seconds",
            "processing_time": processing_time,
            "results": results,
            "note": (
                "FastText model not found, using WordNet only for semantic similarity"
                if not model_found
                else ""
            ),
        }

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}\n{error_details}",
        )
