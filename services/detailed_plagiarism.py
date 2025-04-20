import time
import numpy as np
from modules.topic_classifier import (
    initialize_topic_classifier,
    check_topic_similarity,
)
from modules.fasttext import initialize_semantic_detector, check_semantic_similarity
from modules.finetuned_bert import initialize_bert_detector, check_bert_similarity


# Add this function to ensure all results are serializable
def ensure_serializable(obj):
    """
    Ensure all values in a dictionary or list are serializable

    Args:
        obj: Object to make serializable

    Returns:
        Serializable object
    """
    if isinstance(obj, dict):
        return {k: ensure_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_serializable(item) for item in obj]
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
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class DetailedPlagiarismService:
    def __init__(
        self,
        source_texts,
        fasttext_path="cc.en.300.bin.gz",
        bert_model_name="jpwahle/longformer-base-plagiarism-detection",
    ):
        """
        Initialize combined plagiarism detection system

        Args:
            source_texts (list): List of source texts for reference
            fasttext_path (str): Path to pre-trained FastText model
            bert_model_name (str): Name of the pre-trained BERT model
        """
        print("Initializing Detailed Plagiarism Service...")
        self.source_texts = source_texts

        # Initialize LSA/LDA component
        print("Initializing LSA/LDA component...")
        self.topic_classifier = initialize_topic_classifier(source_texts)

        # Skip initializing Semantic component since we're skipping step 2
        print("Skipping Semantic component initialization (step 2 is disabled)...")
        self.semantic_detector = None

        # Initialize BERT component
        print("Initializing BERT component...")
        self.bert_detector = initialize_bert_detector(source_texts, bert_model_name)

        print("Detailed Plagiarism Service initialized successfully!")

    def detect_plagiarism(
        self,
        suspect_text,
        topic_threshold=0.6,
        semantic_threshold=0.7,
        bert_threshold=0.5,
    ):
        """
        Detect plagiarism using the combined approach

        Args:
            suspect_text (str): Text to check for plagiarism
            topic_threshold (float): Similarity threshold for topic filtering
            semantic_threshold (float): Similarity threshold for semantic methods
            bert_threshold (float): Probability threshold for BERT verification

        Returns:
            dict: Plagiarism detection results
        """
        results = []
        topic_filtered_pairs = []

        # Step 1: Use LSA/LDA to filter by topic similarity
        print("\nSTEP 1: LSA/LDA TOPIC FILTERING")

        for i, source_text in enumerate(self.source_texts):
            # Get topic similarity using LSA/LDA
            topic_result = check_topic_similarity(
                self.topic_classifier, source_text, suspect_text
            )
            topic_similarity = topic_result["average_similarity"]

            print(f"\nSource {i+1}: {source_text[:50]}...")
            print(f"LSA Similarity: {topic_result['lsa_similarity']:.4f}")
            print(f"LDA Similarity: {topic_result['lda_similarity']:.4f}")
            print(f"Source Topic: {topic_result['source_topic']}")
            print(f"Suspect Topic: {topic_result['suspect_topic']}")
            print(f"Average Topic Similarity: {topic_similarity:.4f}")

            # Store all results regardless of threshold
            pair = {
                "source_index": i,
                "source_text": source_text,
                "topic_similarity": float(topic_similarity),
                "lsa_similarity": float(topic_result["lsa_similarity"]),
                "lda_similarity": float(topic_result["lda_similarity"]),
                "source_topic": topic_result["source_topic"],
                "suspect_topic": topic_result["suspect_topic"],
                "suspect_text": suspect_text,
                "is_plagiarized": False,  # Default value
                "potential_plagiarism": False,  # Default value
                "is_exact_match": source_text.strip() == suspect_text.strip(),
                "passed_topic_filter": False,  # Default value
            }

            # If topic similarity is above threshold, add to filtered pairs
            if topic_similarity >= topic_threshold or pair["is_exact_match"]:
                pair["passed_topic_filter"] = True
                topic_filtered_pairs.append(pair)
                print(f"PASSED topic filtering with similarity {topic_similarity:.4f}")
            else:
                print(f"FAILED topic filtering with similarity {topic_similarity:.4f}")
                # Still add to results but mark as not passing topic filter
                results.append(pair)

        # If no pairs passed topic filtering, return the best match
        if not topic_filtered_pairs and self.source_texts:
            # Find the pair with highest topic similarity
            best_match = max(results, key=lambda x: x["topic_similarity"])
            best_match["note"] = "No text passed topic filtering, showing best match"
            return [ensure_serializable(best_match)]

        # Step 2: Skip Semantic Similarity Filtering (commented out for performance)
        """
        print("\nSTEP 2: SEMANTIC SIMILARITY FILTERING")
        semantic_filtered_pairs = []

        for pair in topic_filtered_pairs:
            source_text = pair["source_text"]

            # Get semantic similarity
            semantic_result = check_semantic_similarity(
                self.semantic_detector, source_text, suspect_text
            )
            semantic_similarity = semantic_result["average_similarity"]

            print(f"\nSource {pair['source_index']+1}: {source_text[:50]}...")
            print(f"FastText Similarity: {semantic_result['fasttext_similarity']:.4f}")
            print(f"WordNet Similarity: {semantic_result['wordnet_similarity']:.4f}")
            print(f"Average Semantic Similarity: {semantic_similarity:.4f}")

            # Update pair with semantic results
            pair.update(
                {
                    "semantic_similarity": float(semantic_similarity),
                    "fasttext_similarity": float(
                        semantic_result["fasttext_similarity"]
                    ),
                    "wordnet_similarity": float(semantic_result["wordnet_similarity"]),
                    "passed_semantic_filter": False,  # Default value
                }
            )

            # If semantic similarity is above threshold, add to filtered pairs
            if semantic_similarity >= semantic_threshold or pair["is_exact_match"]:
                pair["passed_semantic_filter"] = True
                semantic_filtered_pairs.append(pair)
                print(f"PASSED semantic filtering with similarity {semantic_similarity:.4f}")
            else:
                print(f"FAILED semantic filtering with similarity {semantic_similarity:.4f}")
                # Still add to results but mark as not passing semantic filter
                results.append(pair)

        # If no pairs passed semantic filtering, return the best match from topic filtering
        if not semantic_filtered_pairs and topic_filtered_pairs:
            # Find the pair with highest semantic similarity
            best_match = max(topic_filtered_pairs, key=lambda x: x.get("semantic_similarity", 0))
            best_match["note"] = "No text passed semantic filtering, showing best match from topic filtering"
            return [ensure_serializable(best_match)]
        """

        # Skip Step 2 and use topic_filtered_pairs directly for Step 3
        semantic_filtered_pairs = topic_filtered_pairs

        # Add placeholder semantic similarity values to maintain compatibility
        for pair in semantic_filtered_pairs:
            pair.update(
                {
                    "semantic_similarity": 0.0,
                    "fasttext_similarity": 0.0,
                    "wordnet_similarity": 0.0,
                    "passed_semantic_filter": True,  # Mark as passed to continue to next step
                }
            )

        # Step 3: Use BERT for final verification
        print("\nSTEP 3: BERT VERIFICATION")

        for pair in semantic_filtered_pairs:
            source_text = pair["source_text"]

            # Get BERT similarity
            bert_result = check_bert_similarity(
                self.bert_detector, source_text, suspect_text
            )
            bert_probability = bert_result["bert_plagiarism_probability"]

            print(f"\nSource {pair['source_index']+1}: {source_text[:50]}...")
            print(f"BERT Plagiarism Probability: {bert_probability:.4f}")

            # Update pair with BERT results
            pair.update(
                {
                    "bert_probability": float(bert_probability),
                    "passed_bert_verification": False,  # Default value
                }
            )

            # If BERT probability is above threshold, mark as plagiarized
            if bert_probability >= bert_threshold or pair["is_exact_match"]:
                pair["passed_bert_verification"] = True
                pair["is_plagiarized"] = True
                print(
                    f"PASSED BERT verification with probability {bert_probability:.4f}"
                )
            elif (
                bert_probability >= bert_threshold * 0.7
            ):  # Lower threshold for potential plagiarism
                pair["potential_plagiarism"] = True
                print(
                    f"POTENTIAL plagiarism detected with probability {bert_probability:.4f}"
                )
            else:
                print(
                    f"FAILED BERT verification with probability {bert_probability:.4f}"
                )

            # Add to final results
            results.append(pair)

        # Sort results by plagiarism status and similarity scores
        sorted_results = sorted(
            results,
            key=lambda x: (
                x["is_plagiarized"],
                x["potential_plagiarism"],
                x.get("bert_probability", 0),
                x.get("semantic_similarity", 0),
                x["topic_similarity"],
            ),
            reverse=True,
        )

        return ensure_serializable(sorted_results)

    def check_multiple_suspects(
        self,
        suspect_texts,
        topic_threshold=0.6,
        semantic_threshold=0.7,
        bert_threshold=0.5,
    ):
        """
        Check multiple suspect texts for plagiarism

        Args:
            suspect_texts (list): List of suspect texts
            topic_threshold (float): Threshold for topic similarity filtering
            semantic_threshold (float): Threshold for semantic similarity filtering
            bert_threshold (float): Threshold for BERT verification

        Returns:
            list: Plagiarism detection results for all suspect texts
        """
        all_results = []

        for i, suspect_text in enumerate(suspect_texts):
            print(f"\n{'='*80}")
            print(f"CHECKING SUSPECT TEXT {i+1}/{len(suspect_texts)}")
            print(f"{'='*80}")
            print(f"Text: {suspect_text[:100]}...")

            # Detect plagiarism for current suspect text
            results = self.detect_plagiarism(
                suspect_text,
                topic_threshold=topic_threshold,
                semantic_threshold=semantic_threshold,
                bert_threshold=bert_threshold,
            )

            # Add suspect index to results
            for result in results:
                result["suspect_index"] = i

            # Add to all results
            all_results.extend(results)

        # Ensure all results are serializable
        return ensure_serializable(all_results)


def initialize_plagiarism_service(
    source_texts,
    fasttext_path="cc.en.300.bin.gz",
    bert_model_name="jpwahle/longformer-base-plagiarism-detection",
):
    """
    Initialize the plagiarism detection service

    Args:
        source_texts (list): List of source texts
        fasttext_path (str): Path to pre-trained FastText model
        bert_model_name (str): Name of the pre-trained BERT model

    Returns:
        DetailedPlagiarismService: Initialized plagiarism service
    """
    return DetailedPlagiarismService(source_texts, fasttext_path, bert_model_name)


def check_plagiarism(
    service,
    suspect_text,
    topic_threshold=0.6,
    semantic_threshold=0.7,
    bert_threshold=0.5,
):
    """
    Check a single suspect text for plagiarism

    Args:
        service (DetailedPlagiarismService): Plagiarism detection service
        suspect_text (str): Suspect text to check
        topic_threshold (float): Threshold for topic similarity filtering
        semantic_threshold (float): Threshold for semantic similarity filtering
        bert_threshold (float): Threshold for BERT verification

    Returns:
        list: Plagiarism detection results
    """
    return service.detect_plagiarism(
        suspect_text,
        topic_threshold=topic_threshold,
        semantic_threshold=semantic_threshold,
        bert_threshold=bert_threshold,
    )


def check_multiple_texts(
    service,
    suspect_texts,
    topic_threshold=0.6,
    semantic_threshold=0.7,
    bert_threshold=0.5,
):
    """
    Check multiple suspect texts for plagiarism

    Args:
        service (DetailedPlagiarismService): Plagiarism detection service
        suspect_texts (list): List of suspect texts
        topic_threshold (float): Threshold for topic similarity filtering
        semantic_threshold (float): Threshold for semantic similarity filtering
        bert_threshold (float): Threshold for BERT verification

    Returns:
        list: Plagiarism detection results for all suspect texts
    """
    return service.check_multiple_suspects(
        suspect_texts,
        topic_threshold=topic_threshold,
        semantic_threshold=semantic_threshold,
        bert_threshold=bert_threshold,
    )


def check_cross_plagiarism(
    texts,
    topic_threshold=0.6,
    semantic_threshold=0.7,
    bert_threshold=0.5,
    fasttext_path="cc.en.300.bin.gz",
    bert_model_name="jpwahle/longformer-base-plagiarism-detection",
):
    """
    Check for plagiarism across all texts by comparing each text with all others,
    ensuring each pair is only checked once.

    Args:
        texts (list): List of texts to check against each other
        topic_threshold (float): Threshold for topic similarity filtering
        semantic_threshold (float): Threshold for semantic similarity filtering
        bert_threshold (float): Threshold for BERT verification
        fasttext_path (str): Path to pre-trained FastText model (not used since step 2 is skipped)
        bert_model_name (str): Name of the pre-trained BERT model

    Returns:
        list: Cross-plagiarism detection results
    """
    if not texts or len(texts) < 2:
        return []

    all_results = []
    topic_filtered_pairs = []

    # Initialize a single service with all texts
    print("Initializing plagiarism service with all texts...")
    service = DetailedPlagiarismService(texts, fasttext_path, bert_model_name)

    # Step 1: Perform LSA/LDA topic filtering on all pairs
    print("\nSTEP 1: LSA/LDA TOPIC FILTERING FOR ALL PAIRS")

    # Check each pair only once (i checks against j where j > i)
    for i in range(len(texts) - 1):
        for j in range(i + 1, len(texts)):
            print(f"\n{'='*80}")
            print(f"CHECKING TEXT PAIR ({i+1},{j+1}) OUT OF {len(texts)} TEXTS")
            print(f"{'='*80}")

            source_text = texts[j]
            suspect_text = texts[i]

            # Get topic similarity using LSA/LDA
            topic_result = check_topic_similarity(
                service.topic_classifier, source_text, suspect_text
            )
            topic_similarity = topic_result["average_similarity"]

            print(f"LSA Similarity: {topic_result['lsa_similarity']:.4f}")
            print(f"LDA Similarity: {topic_result['lda_similarity']:.4f}")
            print(f"Average Topic Similarity: {topic_similarity:.4f}")

            # Store pair information
            pair = {
                "source_index": j,
                "suspect_index": i,
                "source_text": source_text,
                "suspect_text": suspect_text,
                "topic_similarity": float(topic_similarity),
                "lsa_similarity": float(topic_result["lsa_similarity"]),
                "lda_similarity": float(topic_result["lda_similarity"]),
                "source_topic": topic_result["source_topic"],
                "suspect_topic": topic_result["suspect_topic"],
                "is_plagiarized": False,  # Default value
                "potential_plagiarism": False,  # Default value
                "is_exact_match": source_text.strip() == suspect_text.strip(),
                "passed_topic_filter": False,  # Default value
                "text_index": i,
                "compared_with_index": j,
            }

            # If topic similarity is above threshold, add to filtered pairs
            if topic_similarity >= topic_threshold or pair["is_exact_match"]:
                pair["passed_topic_filter"] = True
                topic_filtered_pairs.append(pair)
                print(f"PASSED topic filtering with similarity {topic_similarity:.4f}")
            else:
                print(f"FAILED topic filtering with similarity {topic_similarity:.4f}")
                # Still add to results but mark as not passing topic filter
                all_results.append(pair)

    # If no pairs passed topic filtering, return the best match
    if not topic_filtered_pairs and len(texts) >= 2:
        # Find the pair with highest topic similarity
        best_match = max(all_results, key=lambda x: x["topic_similarity"])
        best_match["note"] = "No text pairs passed topic filtering, showing best match"
        return [ensure_serializable(best_match)]

    # Add placeholder semantic similarity values to maintain compatibility
    for pair in topic_filtered_pairs:
        pair.update(
            {
                "semantic_similarity": 0.0,
                "fasttext_similarity": 0.0,
                "wordnet_similarity": 0.0,
                "passed_semantic_filter": True,  # Mark as passed to continue to next step
            }
        )

    # Step 3: Use BERT for final verification on filtered pairs
    print("\nSTEP 3: BERT VERIFICATION FOR FILTERED PAIRS")

    for pair in topic_filtered_pairs:
        source_text = pair["source_text"]
        suspect_text = pair["suspect_text"]

        # Get BERT similarity
        bert_result = check_bert_similarity(
            service.bert_detector, source_text, suspect_text
        )
        bert_probability = bert_result["bert_plagiarism_probability"]

        print(f"\nChecking pair ({pair['suspect_index']+1},{pair['source_index']+1})")
        print(f"BERT Plagiarism Probability: {bert_probability:.4f}")

        # Update pair with BERT results
        pair.update(
            {
                "bert_probability": float(bert_probability),
                "passed_bert_verification": False,  # Default value
            }
        )

        # If BERT probability is above threshold, mark as plagiarized
        if bert_probability >= bert_threshold or pair["is_exact_match"]:
            pair["passed_bert_verification"] = True
            pair["is_plagiarized"] = True
            print(f"PASSED BERT verification with probability {bert_probability:.4f}")
        elif (
            bert_probability >= bert_threshold * 0.7
        ):  # Lower threshold for potential plagiarism
            pair["potential_plagiarism"] = True
            print(
                f"POTENTIAL plagiarism detected with probability {bert_probability:.4f}"
            )
        else:
            print(f"FAILED BERT verification with probability {bert_probability:.4f}")

        # Add to final results
        all_results.append(pair)

    # Sort results by plagiarism status and similarity scores
    sorted_results = sorted(
        all_results,
        key=lambda x: (
            x["is_plagiarized"],
            x["potential_plagiarism"],
            x.get("bert_probability", 0),
            x.get("semantic_similarity", 0),
            x["topic_similarity"],
        ),
        reverse=True,
    )

    # Ensure all results are serializable
    return ensure_serializable(sorted_results)
