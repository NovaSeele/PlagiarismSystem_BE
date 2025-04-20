import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from nltk.tokenize import word_tokenize


def initialize_bert_detector(
    source_texts, model_name="jpwahle/longformer-base-plagiarism-detection"
):
    """
    Initialize BERT-based plagiarism detector

    Args:
        source_texts (list): List of source texts
        model_name (str): Name of the pre-trained model

    Returns:
        dict: Initialized BERT detector
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"BERT model loaded successfully and running on {device}")

    # Create BERT detector dictionary
    bert_detector = {
        "model_name": model_name,
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "source_texts": source_texts,
    }

    # Generate embeddings for source texts
    source_embeddings = generate_bert_embeddings(bert_detector, source_texts)

    # Create FAISS index
    embedding_dim = source_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(source_embeddings.astype(np.float32))

    print(f"FAISS index created with {len(source_texts)} source documents")

    # Update the bert_detector dictionary with embeddings and index
    bert_detector.update({"source_embeddings": source_embeddings, "faiss_index": index})

    return bert_detector


def generate_bert_embeddings(bert_detector, texts):
    """
    Generate embeddings for texts using the BERT model

    Args:
        bert_detector (dict): BERT detector components
        texts (list): List of texts

    Returns:
        numpy.ndarray: Text embeddings
    """
    embeddings = []

    for text in texts:
        # Get embedding for a single text
        embedding = get_bert_embedding(bert_detector, text)
        embeddings.append(embedding)

    return np.array(embeddings)


def get_bert_embedding(bert_detector, text):
    """
    Get embedding for a single text using BERT

    Args:
        bert_detector (dict): BERT detector components
        text (str): Input text

    Returns:
        numpy.ndarray: Text embedding
    """
    # Tokenize the text
    inputs = bert_detector["tokenizer"](
        text, return_tensors="pt", truncation=True, max_length=512, padding=True
    )
    inputs = {k: v.to(bert_detector["device"]) for k, v in inputs.items()}

    # Get model output
    with torch.no_grad():
        outputs = bert_detector["model"](**inputs, output_hidden_states=True)

    # Use the last hidden state of the [CLS] token as the embedding
    last_hidden_state = outputs.hidden_states[-1]
    cls_embedding = last_hidden_state[:, 0, :].cpu().numpy()[0]

    return cls_embedding


def get_bert_similarity(bert_detector, original_text, suspect_text):
    """
    Calculate similarity between texts using BERT embeddings

    Args:
        bert_detector (dict): BERT detector components
        original_text (str): Original text
        suspect_text (str): Suspect text

    Returns:
        float: Similarity score
    """
    # Generate embeddings
    original_embedding = get_bert_embedding(bert_detector, original_text)
    suspect_embedding = get_bert_embedding(bert_detector, suspect_text)

    # Calculate cosine similarity
    similarity = cosine_similarity([original_embedding], [suspect_embedding])[0][0]

    return similarity


def get_bert_plagiarism_probability(bert_detector, original_text, suspect_text):
    """
    Get plagiarism probability using the finetuned BERT model

    Args:
        bert_detector (dict): BERT detector components
        original_text (str): Original text
        suspect_text (str): Suspect text

    Returns:
        float: Plagiarism probability
    """
    # Prepare input for the model - use paired input format
    inputs = bert_detector["tokenizer"](
        original_text,
        suspect_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    inputs = {k: v.to(bert_detector["device"]) for k, v in inputs.items()}

    # Get model prediction
    with torch.no_grad():
        outputs = bert_detector["model"](**inputs)

    # Get probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    # Return probability of plagiarism (assuming index 1 is the plagiarism class)
    return probs[0, 1].item()


def find_similar_segments(source_text, suspect_text, min_length=5):
    """
    Find similar text segments between source and suspect texts

    Args:
        source_text (str): Source text
        suspect_text (str): Suspect text
        min_length (int): Minimum length of segments to consider

    Returns:
        list: List of similar text segments
    """
    # Tokenize texts
    source_tokens = word_tokenize(source_text.lower())
    suspect_tokens = word_tokenize(suspect_text.lower())

    similar_segments = []

    # Simple n-gram matching
    for n in range(min_length, min(15, len(source_tokens), len(suspect_tokens))):
        # Generate n-grams for source
        source_ngrams = [
            " ".join(source_tokens[i : i + n])
            for i in range(len(source_tokens) - n + 1)
        ]
        # Generate n-grams for suspect
        suspect_ngrams = [
            " ".join(suspect_tokens[i : i + n])
            for i in range(len(suspect_tokens) - n + 1)
        ]

        # Find matching n-grams
        for s_ngram in source_ngrams:
            if s_ngram in suspect_ngrams and len(s_ngram.split()) >= min_length:
                similar_segments.append(s_ngram)

    # Remove duplicates and sort by length (longest first)
    unique_segments = list(set(similar_segments))
    unique_segments.sort(key=len, reverse=True)

    # Return top 5 segments or fewer if less available
    return unique_segments[:5]


def convert_numpy_to_python(obj):
    """
    Convert numpy types to Python native types for JSON serialization

    Args:
        obj: Object that might contain numpy types

    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(
        obj,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
        return complex(obj)
    return obj


def check_bert_similarity(
    bert_detector, original_text, suspect_text, bert_threshold=0.5
):
    """
    Check similarity using BERT model

    Args:
        bert_detector (dict): BERT detector components
        original_text (str): Original text
        suspect_text (str): Suspect text
        bert_threshold (float): Threshold for plagiarism detection

    Returns:
        dict: BERT similarity results
    """
    # Generate embedding for suspect text
    suspect_embedding = get_bert_embedding(bert_detector, suspect_text)

    # Convert to the format expected by FAISS
    query_vector = np.array([suspect_embedding]).astype(np.float32)

    # Search the index for top matches
    k = min(3, len(bert_detector["source_texts"]))
    distances, indices = bert_detector["faiss_index"].search(query_vector, k)

    # Convert L2 distances to similarities
    faiss_similarities = 1.0 / (1.0 + distances[0])

    # Get BERT plagiarism probability
    bert_prob = get_bert_plagiarism_probability(
        bert_detector, original_text, suspect_text
    )

    # Get BERT similarity
    bert_similarity = get_bert_similarity(bert_detector, original_text, suspect_text)

    # Find similar text segments
    similar_segments = find_similar_segments(original_text, suspect_text)

    # Check if source is in top FAISS results
    source_index = (
        bert_detector["source_texts"].index(original_text)
        if original_text in bert_detector["source_texts"]
        else -1
    )
    is_in_top_faiss = source_index in indices[0] if source_index >= 0 else False

    # Check if exact match
    is_exact_match = original_text.strip() == suspect_text.strip()

    # Determine if plagiarized
    is_plagiarized = bert_prob >= bert_threshold or is_exact_match

    # If texts are very similar (high BERT similarity), consider as potential plagiarism
    potential_plagiarism = bert_similarity > 0.95 and not is_plagiarized

    # Convert numpy types to Python native types
    return {
        "bert_plagiarism_probability": float(bert_prob),
        "bert_similarity": float(bert_similarity),
        "is_in_top_faiss": bool(is_in_top_faiss),
        "similar_segments": similar_segments,
        "is_plagiarized": bool(is_plagiarized),
        "potential_plagiarism": bool(potential_plagiarism),
        "is_exact_match": bool(is_exact_match),
        "faiss_similarities": [float(sim) for sim in faiss_similarities],
        "faiss_indices": [int(idx) for idx in indices[0]],
    }
