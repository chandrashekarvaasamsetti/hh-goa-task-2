import json
import re
import faiss
from sentence_transformers import SentenceTransformer


INDEX_PATH = "data/index/telugu_prototype.faiss"
META_PATH = "data/index/telugu_prototype.json"


print("Loading index...")
index = faiss.read_index(INDEX_PATH)


print("Loading metadata...")
with open(META_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


print("Loading embedding model...")
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def clean_text(text):
    """
    Preserve Unicode text correctly.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    return text.strip()


def extract_terms(text):
    """
    Extract meaningful Telugu and English terms.
    """
    text = clean_text(text).lower()

    words = re.findall(
        r"[\u0C00-\u0C7F]+|[A-Za-z]+",
        text
    )

    stop_words = {
        "ఏమిటి",
        "ఏమిటి?",
        "ఎప్పుడు",
        "ఎక్కడ",
        "ఎలా",
        "ఎందుకు",
        "ఏది",
        "ఏ",
        "లో",
        "పై",
        "కు",
        "ని",
        "తో",
        "గురించి",
        "అని",
        "ఒక",
        "the",
        "is",
        "are",
        "was",
        "were",
        "what",
        "when",
        "where",
        "how",
        "why",
        "which",
        "a",
        "an",
        "of",
        "in",
        "on",
        "to",
        "for"
    }

    return [
        word
        for word in words
        if len(word) >= 2 and word not in stop_words
    ]


def search(query, top_k=10):

    query = clean_text(query)

    if not query:
        return []

    # ---------------------------------
    # Create query embedding
    # ---------------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    # Retrieve the complete candidate pool.
    # Reranking happens after FAISS retrieval.
    similarities, ids = index.search(
        query_embedding,
        len(data)
    )

    # ---------------------------------
    # Detect definition question
    # ---------------------------------

    is_definition_question = (
        "అంటే ఏమిటి" in query
        or "అంటే ఏమిటి?" in query
        or "ఏమిటి" in query
    )

    # ---------------------------------
    # Extract meaningful query terms
    # ---------------------------------

    query_terms = extract_terms(query)

    candidates = []

    for rank in range(len(ids[0])):

        idx = int(ids[0][rank])

        similarity = float(
            similarities[0][rank]
        )

        # Safety check
        if idx < 0 or idx >= len(data):
            continue

        item = data[idx]

        text = clean_text(
            item.get("text", "")
        )

        # ---------------------------------
        # Lexical overlap
        # ---------------------------------

        text_terms = set(
            extract_terms(text)
        )

        matching_terms = [
            term
            for term in query_terms
            if term in text_terms
        ]

        lexical_bonus = 0.0

        if matching_terms:
            overlap_ratio = (
                len(matching_terms)
                / max(len(query_terms), 1)
            )

            lexical_bonus = min(
                overlap_ratio * 0.10,
                0.10
            )

        # ---------------------------------
        # Definition bonus
        # ---------------------------------

        definition_bonus = 0.0

        if is_definition_question:

            definition_patterns = [
                "is defined as",
                "refers to",
                "means",
                "is a",
                "is an",
                "definition"
            ]

            lower_text = text.lower()

            for pattern in definition_patterns:

                if pattern in lower_text:
                    definition_bonus = 0.05
                    break

        # ---------------------------------
        # Selected passage bonus
        # ---------------------------------

        selected_bonus = (
            0.05
            if item.get("is_selected", 0) == 1
            else 0.0
        )

        # ---------------------------------
        # Small rank bonus
        # ---------------------------------
        #
        # Preserve FAISS's semantic ranking
        # instead of allowing secondary signals
        # to completely dominate it.
        #

        rank_bonus = max(
            0.0,
            0.02 * (1.0 - rank / max(len(data), 1))
        )

        # ---------------------------------
        # Final score
        # ---------------------------------

        final_score = (
            similarity
            + lexical_bonus
            + definition_bonus
            + selected_bonus
            + rank_bonus
        )

        candidates.append({
            "final_score": final_score,
            "similarity": similarity,
            "selected": item.get(
                "is_selected",
                0
            ),
            "text": text
        })

    # ---------------------------------
    # Sort by final score
    # ---------------------------------

    candidates.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    return candidates[:top_k]


# ---------------------------------
# Interactive search
# ---------------------------------

if __name__ == "__main__":

    query = input(
        "\nEnter Telugu query: "
    )

    results = search(
        query,
        top_k=10
    )

    print(
        "\n========== RERANKED SEARCH RESULTS =========="
    )

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n--- Result {i} ---"
        )

        print(
            f"Final Score: "
            f"{result['final_score']:.4f}"
        )

        print(
            f"Similarity: "
            f"{result['similarity']:.4f}"
        )

        print(
            f"Selected: "
            f"{result['selected']}"
        )

        print(
            f"Text: "
            f"{result['text']}"
        )