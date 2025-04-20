import numpy as np
import networkx as nx
from underthesea import word_tokenize, pos_tag
import re
from itertools import combinations
from collections import Counter

def preprocess_text(text, stopwords):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)
    return [word for word in words if word not in stopwords]

def extract_candidate_phrases(words, min_phrase_length=2, max_phrase_length=3):
    candidate_phrases = []
    for phrase_length in range(min_phrase_length, max_phrase_length + 1):
        for i in range(len(words) - phrase_length + 1):
            candidate_phrases.append(' '.join(words[i:i+phrase_length]))
    return candidate_phrases

def calculate_similarity(phrase1, phrase2):
    words1, words2 = set(phrase1.split()), set(phrase2.split())
    common_words = len(words1.intersection(words2))
    total_words = len(words1.union(words2))
    return common_words / total_words if total_words > 0 else 0

def build_textrank_graph(candidates):
    graph = nx.Graph()
    for candidate in candidates:
        graph.add_node(candidate)
    for i, j in combinations(range(len(candidates)), 2):
        similarity = calculate_similarity(candidates[i], candidates[j])
        if similarity > 0:
            graph.add_edge(candidates[i], candidates[j], weight=similarity)
    return graph

def post_process_keywords(keywords):
    filtered_keywords = []
    for i, keyword1 in enumerate(keywords):
        if not any(keyword1 in keyword2 for j, keyword2 in enumerate(keywords) if i != j):
            filtered_keywords.append(keyword1)
    return filtered_keywords

def extract_keywords(text, stopwords, top_n=10):
    words = preprocess_text(text, stopwords)
    candidate_phrases = extract_candidate_phrases(words)
    graph = build_textrank_graph(candidate_phrases)
    
    if not graph.edges():
        phrase_counts = Counter(candidate_phrases)
        top_keywords = [phrase for phrase, _ in phrase_counts.most_common(top_n)]
    else:
        pagerank_scores = nx.pagerank(graph)
        top_keywords = sorted(pagerank_scores, key=pagerank_scores.get, reverse=True)[:top_n * 2]
    
    filtered_keywords = post_process_keywords(top_keywords)
    return filtered_keywords[:top_n]
