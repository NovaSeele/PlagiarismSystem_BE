from fastapi import APIRouter, HTTPException
from typing import List
import json

# from services.text_rank_keyword_vi import run_textrank

from modules.keyword_classifier import categorize_combined

from schemas.pdf_process import KeywordsResponse, TextRequest

router = APIRouter()

# Load categories and category examples from categories.json
with open("D:/Code/NovaSeelePlagiarismSystem/backend/datasets/categories.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
    categories = data['categories']
    category_examples = data['category_examples']
    stopwords = data['stopwords']

# @router.post("/extract_keywords", response_model=List[str])
# async def extract_keywords_endpoint(request: TextRequest):
#     try:
#         keywords = run_textrank(request.text, stopwords, top_n=10)
#         return keywords  # Return the list directly

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    
# @router.post("/categorize")
# def categorize(input_phrases: List[str]):
#     return categorize_combined(input_phrases, categories, category_examples)

# @router.post("/categorize_keywords")
# async def categorize_keywords_endpoint(request: TextRequest):
#     try:
#         keywords = run_textrank(request.text, stopwords, top_n=10)
#         keyword_categories = categorize_combined(keywords, categories, category_examples)
#         return keyword_categories

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
