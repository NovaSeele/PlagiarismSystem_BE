# Phần 1: Định nghĩa các hàm con
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def categorize_phrases_combined(input_phrases, categories, category_examples=None):
    if category_examples is None:
        # Nếu không có ví dụ, sử dụng chính danh mục làm ví dụ
        category_examples = {cat: [cat] for cat in categories}
    
    # Kết hợp tất cả cụm từ đầu vào thành một văn bản duy nhất
    combined_input = " ".join(input_phrases)
    
    # Tạo corpus cho TF-IDF
    corpus = []
    for cat in categories:
        corpus.extend(category_examples[cat])
    corpus.append(combined_input)
    
    # Tính toán TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Tính toán similarity giữa văn bản kết hợp và mỗi danh mục
    combined_input_idx = len(corpus) - 1
    combined_vector = tfidf_matrix[combined_input_idx]
    
    category_scores = []
    category_start_idx = 0
    
    for cat_idx, cat in enumerate(categories):
        cat_examples_count = len(category_examples[cat])
        category_vectors = tfidf_matrix[category_start_idx:category_start_idx + cat_examples_count]
        category_start_idx += cat_examples_count
        
        # Tính điểm similarity trung bình với các ví dụ trong danh mục
        similarities = cosine_similarity(combined_vector, category_vectors).flatten()
        avg_similarity = np.mean(similarities) * 100  # Chuyển thành phần trăm
        
        if avg_similarity > 0:  # Giữ tất cả các danh mục có mức độ phù hợp > 0%
            category_scores.append((cat, avg_similarity))
    
    # Sắp xếp theo điểm số từ cao đến thấp
    category_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Chuẩn hóa điểm số để tổng là 100%
    if category_scores:
        total_score = sum(score for _, score in category_scores)
        normalized_scores = [(cat, (score / total_score) * 100) 
                          for cat, score in category_scores]
        return normalized_scores
    else:
        return [("Khác", 100)]

def categorize_phrases_frequency(input_phrases, categories):
    """
    Phương pháp phân loại dựa trên tần suất xuất hiện của từ khóa trong các cụm từ
    """
    # Trích xuất tất cả các từ từ input_phrases
    all_words = []
    for phrase in input_phrases:
        words = re.findall(r'\w+', phrase.lower())
        all_words.extend(words)
    
    # Đếm tần suất các từ
    word_freq = Counter(all_words)
    
    # Tính điểm số cho mỗi danh mục
    category_scores = {}
    
    for category in categories:
        # Trích xuất từ khóa từ danh mục
        category_keywords = re.findall(r'\w+', category.lower())
        
        # Tính tổng tần suất của các từ khóa trong danh mục
        score = sum(word_freq.get(keyword, 0) for keyword in category_keywords)
        
        if score > 0:
            category_scores[category] = score
    
    # Chuyển thành danh sách và sắp xếp theo điểm số
    scores_list = [(cat, score) for cat, score in category_scores.items()]
    scores_list.sort(key=lambda x: x[1], reverse=True)
    
    # Chuẩn hóa điểm số
    if scores_list:
        total_score = sum(score for _, score in scores_list)
        normalized_scores = [(cat, (score / total_score) * 100) 
                          for cat, score in scores_list]
        return normalized_scores
    else:
        return [("Khác", 100)]


# def categorize_combined(input_phrases, categories, category_examples=None):
#     """
#     Kết hợp phương pháp TF-IDF và tần suất để có kết quả chính xác hơn
#     """
#     # Kết quả từ phương pháp TF-IDF
#     tfidf_scores = categorize_phrases_combined(input_phrases, categories, category_examples)
    
#     # Kết quả từ phương pháp tần suất
#     freq_scores = categorize_phrases_frequency(input_phrases, categories)
    
#     # Tạo từ điển để kết hợp điểm số
#     combined_scores = {}
    
#     # Trọng số cho mỗi phương pháp (có thể điều chỉnh)
#     tfidf_weight = 0.7
#     freq_weight = 0.3
    
#     # Kết hợp điểm số từ phương pháp TF-IDF
#     for cat, score in tfidf_scores:
#         combined_scores[cat] = score * tfidf_weight
    
#     # Kết hợp điểm số từ phương pháp tần suất
#     for cat, score in freq_scores:
#         if cat in combined_scores:
#             combined_scores[cat] += score * freq_weight
#         else:
#             combined_scores[cat] = score * freq_weight
    
#     # Chuyển thành danh sách và sắp xếp
#     result = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
#     # Chuẩn hóa điểm số
#     total_score = sum(score for _, score in result)
#     if total_score > 0:
#         normalized_result = [(cat, (score / total_score) * 100) for cat, score in result]
#     else:
#         normalized_result = []
    
#     # Lọc kết quả theo điều kiện
#     # filtered_result = [normalized_result[i] for i in range(min(3, len(normalized_result))) if normalized_result[i][1] >= 15]
#     # filtered_result.extend([(cat, score) for cat, score in normalized_result if score > 50 and score >= 15 and (cat, score) not in filtered_result])
    
#     #Kết quả chỉ có category
#     filtered_result = [normalized_result[i][0] for i in range(min(3, len(normalized_result))) if normalized_result[i][1] >= 15]
#     filtered_result.extend([cat for cat, score in normalized_result if score > 50 and score >= 15 and cat not in filtered_result])
    
#     return filtered_result

def categorize_combined(input_phrases, categories, category_examples=None):
    """
    Kết hợp phương pháp TF-IDF và tần suất để có kết quả chính xác hơn
    """
    # Kết quả từ phương pháp TF-IDF
    tfidf_scores = categorize_phrases_combined(input_phrases, categories, category_examples)
    
    # Kết quả từ phương pháp tần suất
    freq_scores = categorize_phrases_frequency(input_phrases, categories)
    
    # Tạo từ điển để kết hợp điểm số
    combined_scores = {}
    
    # Trọng số cho mỗi phương pháp (có thể điều chỉnh)
    tfidf_weight = 0.7
    freq_weight = 0.3
    
    # Kết hợp điểm số từ phương pháp TF-IDF
    for cat, score in tfidf_scores:
        combined_scores[cat] = score * tfidf_weight
    
    # Kết hợp điểm số từ phương pháp tần suất
    for cat, score in freq_scores:
        if cat in combined_scores:
            combined_scores[cat] += score * freq_weight
        else:
            combined_scores[cat] = score * freq_weight
    
    # Chuyển thành danh sách và sắp xếp
    result = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Chuẩn hóa điểm số
    total_score = sum(score for _, score in result)
    if total_score > 0:
        normalized_result = [(cat, (score / total_score) * 100) for cat, score in result]
    else:
        normalized_result = []
    
    # Lọc kết quả theo điều kiện
    filtered_result = [normalized_result[i][0] for i in range(min(3, len(normalized_result))) if normalized_result[i][1] >= 15]
    filtered_result.extend([cat for cat, score in normalized_result if score > 50 and score >= 15 and cat not in filtered_result])
    
    return filtered_result
