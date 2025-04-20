from modules.text_rank_keyword_vi import extract_keywords

def run_textrank(text, stopwords, top_n=10):
    return extract_keywords(text, stopwords, top_n)