import nltk
import numpy as np
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import itertools
import gensim
from gensim.models import FastText, KeyedVectors

# Download necessary NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Global variable to store the loaded model
GLOBAL_FASTTEXT_MODEL = None

def preprocess_text(text):
    """
    Preprocess a single text

    Args:
        text (str): Input text

    Returns:
        list: Preprocessed tokens
    """
    # Download stopwords if not already downloaded
    try:
        stop_words = set(stopwords.words("english"))
    except:
        nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))

    # Tokenize and convert to lowercase
    tokens = word_tokenize(text.lower())

    # Remove stopwords and non-alphabetic tokens
    cleaned_tokens = [
        token for token in tokens if token.isalpha() and token not in stop_words
    ]

    return cleaned_tokens


def load_fasttext(model_path):
    """
    Load pre-trained FastText model

    Args:
        model_path (str): Path to the pre-trained model

    Returns:
        FastText model or None if loading fails
    """
    global GLOBAL_FASTTEXT_MODEL
    
    # If model is already loaded, return it
    if GLOBAL_FASTTEXT_MODEL is not None:
        print("Using already loaded FastText model.")
        return GLOBAL_FASTTEXT_MODEL
        
    print(f"Loading FastText model from {model_path}...")

    # Check if file exists
    if not os.path.exists(model_path):
        print(f"FastText model file not found: {model_path}")
        print("Using WordNet similarity only.")
        return None

    # Try different paths if the model is a common name
    if not os.path.exists(model_path) and not os.path.isabs(model_path):
        # Try looking in common directories
        possible_paths = [
            model_path,
            os.path.join("models", model_path),
            os.path.join("data", "models", model_path),
            os.path.join("..", "models", model_path),
            os.path.join(os.path.dirname(__file__), "..", "..", "models", model_path),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"Found model at {model_path}")
                break

    # Load the model with multiple attempts
    try:
        # First try the method that worked in hybrid_semantic.py
        print("Attempting to load with FastText.load_fasttext_format...")
        model = FastText.load_fasttext_format(model_path)
        print("FastText model loaded successfully using load_fasttext_format.")
        GLOBAL_FASTTEXT_MODEL = model
        return model
    except Exception as e:
        print(f"First loading attempt failed: {e}")
        try:
            # Try loading with gensim's load_word2vec_format
            print("Attempting to load with KeyedVectors.load_word2vec_format...")
            model = KeyedVectors.load_word2vec_format(
                model_path,
                binary=True,
                no_header=True,  # Add this parameter for binary files without headers
            )
            print("FastText model loaded successfully.")
            GLOBAL_FASTTEXT_MODEL = model
            return model
        except Exception as e2:
            print(f"Second loading attempt failed: {e2}")
            try:
                # Try without no_header parameter
                model = KeyedVectors.load_word2vec_format(model_path, binary=True)
                print("FastText model loaded successfully with standard parameters.")
                GLOBAL_FASTTEXT_MODEL = model
                return model
            except Exception as e3:
                print(f"Third loading attempt failed: {e3}")
                try:
                    # Try loading as a gzipped file
                    import gzip

                    if model_path.endswith(".gz"):
                        with gzip.open(model_path, "rb") as f:
                            model = KeyedVectors.load_word2vec_format(f, binary=True)
                        print("FastText model loaded successfully from gzipped file.")
                        GLOBAL_FASTTEXT_MODEL = model
                        return model
                except Exception as e4:
                    print(f"Fourth loading attempt failed: {e4}")
                    print("All loading attempts failed. Using WordNet similarity only.")
                    return None


def get_word_embedding_similarity(model, text1, text2):
    """
    Calculate similarity between texts using word embeddings

    Args:
        model: Word embedding model (FastText)
        text1 (str): First text
        text2 (str): Second text

    Returns:
        float: Similarity score
    """
    # If model is None, return 0 similarity
    if model is None:
        return 0.0

    # Preprocess texts
    tokens1 = preprocess_text(text1)
    tokens2 = preprocess_text(text2)

    # If either text has no tokens after preprocessing, return 0
    if not tokens1 or not tokens2:
        return 0.0

    # Get embeddings for each word
    embeddings1 = []
    embeddings2 = []

    for token in tokens1:
        try:
            # Handle different model types (FastText vs KeyedVectors)
            if isinstance(model, FastText):
                embeddings1.append(model.wv[token])
            else:
                embeddings1.append(model[token])
        except KeyError:
            # Skip words not in vocabulary
            continue
        except Exception as e:
            print(f"Error getting embedding for token '{token}': {str(e)}")
            continue

    for token in tokens2:
        try:
            # Handle different model types (FastText vs KeyedVectors)
            if isinstance(model, FastText):
                embeddings2.append(model.wv[token])
            else:
                embeddings2.append(model[token])
        except KeyError:
            # Skip words not in vocabulary
            continue
        except Exception as e:
            print(f"Error getting embedding for token '{token}': {str(e)}")
            continue

    # If no embeddings found, return 0
    if not embeddings1 or not embeddings2:
        return 0.0

    # Calculate average embeddings
    avg_embedding1 = np.mean(embeddings1, axis=0)
    avg_embedding2 = np.mean(embeddings2, axis=0)

    # Calculate cosine similarity
    similarity = np.dot(avg_embedding1, avg_embedding2) / (
        np.linalg.norm(avg_embedding1) * np.linalg.norm(avg_embedding2)
    )

    # Convert numpy float to Python float
    return float(similarity)


def get_max_wordnet_similarity(text1, text2):
    """
    Calculate maximum WordNet similarity between texts

    Args:
        text1 (str): First text
        text2 (str): Second text

    Returns:
        float: Maximum similarity score
    """
    # Preprocess texts
    tokens1 = preprocess_text(text1)
    tokens2 = preprocess_text(text2)

    # If either text has no tokens, return 0
    if not tokens1 or not tokens2:
        return 0.0

    # Calculate similarities between all pairs of words
    max_similarities = []

    for token1 in tokens1:
        token_similarities = []

        for token2 in tokens2:
            # Get all synsets for both tokens
            synsets1 = wn.synsets(token1)
            synsets2 = wn.synsets(token2)

            # If either token has no synsets, skip
            if not synsets1 or not synsets2:
                continue

            # Calculate similarities between all synset pairs
            synset_similarities = []

            for s1, s2 in itertools.product(synsets1, synsets2):
                try:
                    similarity = s1.path_similarity(s2)
                    if similarity is not None:
                        synset_similarities.append(similarity)
                except:
                    continue

            # Get maximum similarity between synsets
            if synset_similarities:
                token_similarities.append(max(synset_similarities))

        # Get maximum similarity for current token
        if token_similarities:
            max_similarities.append(max(token_similarities))

    # Return average of maximum similarities
    if max_similarities:
        return float(np.mean(max_similarities))
    else:
        return 0.0


def initialize_semantic_detector(source_texts, fasttext_path="d:/Code/NovaSeelePlagiarismSystem/backend/models/cc.en.300.bin.gz"):
    """
    Initialize the semantic detector with source texts and FastText model

    Args:
        source_texts (list): List of source texts
        fasttext_path (str): Path to pre-trained FastText model

    Returns:
        dict: Initialized semantic detector
    """
    # Load FastText model
    fasttext_model = load_fasttext(fasttext_path)

    return {"fasttext_model": fasttext_model, "source_texts": source_texts}


def check_semantic_similarity(semantic_detector, original_text, suspect_text):
    """
    Check semantic similarity between source and suspect text

    Args:
        semantic_detector (dict): Semantic detector components
        original_text (str): Original text
        suspect_text (str): Suspect text

    Returns:
        dict: Semantic similarity results
    """
    # Get FastText similarity if model is available
    fasttext_model = semantic_detector["fasttext_model"]
    fasttext_similarity = get_word_embedding_similarity(
        fasttext_model, original_text, suspect_text
    )

    # Get WordNet similarity
    wordnet_similarity = get_max_wordnet_similarity(original_text, suspect_text)

    # Calculate average similarity
    avg_similarity = float(np.mean([fasttext_similarity, wordnet_similarity]))

    return {
        "fasttext_similarity": float(fasttext_similarity),
        "wordnet_similarity": float(wordnet_similarity),
        "average_similarity": avg_similarity,
    }


# Function to preload the model at application startup
def preload_fasttext_model(
    model_path="d:/Code/NovaSeelePlagiarismSystem/backend/models/cc.en.300.bin.gz",
):
    """
    Preload the FastText model at application startup
    
    Args:
        model_path (str): Path to the FastText model
        
    Returns:
        The loaded model or None if loading fails
    """
    global GLOBAL_FASTTEXT_MODEL

    if GLOBAL_FASTTEXT_MODEL is None:
        GLOBAL_FASTTEXT_MODEL = load_fasttext(model_path)

    return GLOBAL_FASTTEXT_MODEL
