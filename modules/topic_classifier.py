import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from gensim.corpora import Dictionary
from gensim.models import LdaModel

# Download necessary NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


def preprocess_texts(texts):
    """
    Preprocess texts by tokenization and removing stopwords

    Args:
        texts (list): List of input texts

    Returns:
        list: Preprocessed and tokenized texts
    """
    # Download stopwords if not already downloaded
    try:
        stop_words = set(stopwords.words("english"))
    except:
        nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))

    preprocessed = []
    for text in texts:
        # Tokenize and convert to lowercase
        tokens = word_tokenize(text.lower())

        # Remove stopwords and non-alphabetic tokens
        cleaned_tokens = [
            token for token in tokens if token.isalpha() and token not in stop_words
        ]

        preprocessed.append(cleaned_tokens)

    return preprocessed


def train_lsa(preprocessed_texts, source_texts):
    """
    Train Latent Semantic Analysis (LSA) model

    Args:
        preprocessed_texts (list): Preprocessed texts
        source_texts (list): Original source texts

    Returns:
        dict: LSA model components
    """
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words="english")

    # Convert texts to full text for vectorization
    full_texts = [" ".join(tokens) for tokens in preprocessed_texts]

    # Create TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(full_texts)

    # Perform LSA (Truncated SVD)
    lsa = TruncatedSVD(n_components=10, random_state=42)
    lsa_matrix = lsa.fit_transform(tfidf_matrix)

    return {"vectorizer": vectorizer, "lsa": lsa, "lsa_matrix": lsa_matrix}


def train_lda(preprocessed_texts):
    """
    Train Latent Dirichlet Allocation (LDA) model

    Args:
        preprocessed_texts (list): Preprocessed texts

    Returns:
        dict: LDA model and dictionary
    """
    # Create dictionary
    dictionary = Dictionary(preprocessed_texts)

    # Create corpus
    corpus = [dictionary.doc2bow(text) for text in preprocessed_texts]

    # Train LDA model
    lda_model = LdaModel(
        corpus=corpus, id2word=dictionary, num_topics=5, random_state=42
    )

    return {"model": lda_model, "dictionary": dictionary}


def get_lsa_similarity(lsa_model, original_text, suspect_text):
    """
    Calculate similarity using Latent Semantic Analysis

    Args:
        lsa_model (dict): LSA model components
        original_text (str): Original text
        suspect_text (str): Suspect text

    Returns:
        float: Similarity score
    """
    # Get vectorizer and LSA from trained model
    vectorizer = lsa_model["vectorizer"]
    lsa = lsa_model["lsa"]

    # Vectorize texts
    tfidf_matrix = vectorizer.transform([original_text, suspect_text])

    # Transform to LSA space
    lsa_matrix = lsa.transform(tfidf_matrix)

    # Calculate cosine similarity
    return cosine_similarity(lsa_matrix[0:1], lsa_matrix[1:2])[0][0]


def get_lda_similarity(lda_model, original_text, suspect_text):
    """
    Calculate similarity using Latent Dirichlet Allocation

    Args:
        lda_model (dict): LDA model components
        original_text (str): Original text
        suspect_text (str): Suspect text

    Returns:
        float: Similarity score
    """
    # Get LDA model and dictionary
    model = lda_model["model"]
    dictionary = lda_model["dictionary"]

    # Tokenize texts
    original_tokens = word_tokenize(original_text.lower())
    suspect_tokens = word_tokenize(suspect_text.lower())

    # Convert to bag of words
    orig_bow = dictionary.doc2bow(original_tokens)
    susp_bow = dictionary.doc2bow(suspect_tokens)

    # Get topic distributions
    orig_topics = model.get_document_topics(orig_bow)
    susp_topics = model.get_document_topics(susp_bow)

    # Convert to vector representation
    def topics_to_vector(topics, num_topics=5):
        vector = np.zeros(num_topics)
        for topic_id, prob in topics:
            vector[topic_id] = prob
        return vector

    orig_vector = topics_to_vector(orig_topics)
    susp_vector = topics_to_vector(susp_topics)

    # Calculate cosine similarity
    return np.dot(orig_vector, susp_vector) / (
        np.linalg.norm(orig_vector) * np.linalg.norm(susp_vector)
    )


def extract_topics(lda_model, text):
    """
    Extract the main topic from a text

    Args:
        lda_model (dict): LDA model components
        text (str): Input text

    Returns:
        str: Main topic description
    """
    # Get LDA model and dictionary
    model = lda_model["model"]
    dictionary = lda_model["dictionary"]

    # Tokenize text
    tokens = word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]

    # Convert to bag of words
    bow = dictionary.doc2bow(tokens)

    # Get topic distribution
    topics = model.get_document_topics(bow)

    # Find the dominant topic
    if not topics:
        return "No clear topic"

    dominant_topic = max(topics, key=lambda x: x[1])
    topic_id, topic_prob = dominant_topic

    # Get the top words for this topic
    top_words = [word for word, _ in model.show_topic(topic_id, topn=5)]

    # Return topic description
    return f"Topic {topic_id}: {', '.join(top_words)} ({topic_prob:.2f})"


def initialize_topic_classifier(source_texts):
    """
    Initialize the topic classifier with source texts

    Args:
        source_texts (list): List of source texts

    Returns:
        dict: Initialized topic classifier models
    """
    # Preprocess texts
    preprocessed_texts = preprocess_texts(source_texts)

    # Train LSA model
    lsa_model = train_lsa(preprocessed_texts, source_texts)

    # Train LDA model
    lda_model = train_lda(preprocessed_texts)

    return {
        "preprocessed_texts": preprocessed_texts,
        "lsa_model": lsa_model,
        "lda_model": lda_model,
        "source_texts": source_texts,
    }


def check_topic_similarity(topic_classifier, source_text, suspect_text):
    """
    Check topic similarity between source and suspect text

    Args:
        topic_classifier (dict): Topic classifier models
        source_text (str): Source text
        suspect_text (str): Suspect text

    Returns:
        dict: Topic similarity results
    """
    # Get LSA similarity
    lsa_similarity = get_lsa_similarity(
        topic_classifier["lsa_model"], source_text, suspect_text
    )

    # Get LDA similarity
    lda_similarity = get_lda_similarity(
        topic_classifier["lda_model"], source_text, suspect_text
    )

    # Extract topics
    source_topic = extract_topics(topic_classifier["lda_model"], source_text)
    suspect_topic = extract_topics(topic_classifier["lda_model"], suspect_text)

    # Calculate average similarity
    avg_similarity = np.mean([lsa_similarity, lda_similarity])

    return {
        "lsa_similarity": lsa_similarity,
        "lda_similarity": lda_similarity,
        "source_topic": source_topic,
        "suspect_topic": suspect_topic,
        "average_similarity": avg_similarity,
    }
