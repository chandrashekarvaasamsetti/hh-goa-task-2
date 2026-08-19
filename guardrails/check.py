import re
from typing import List, Dict


SIMILARITY_THRESHOLD = 0.65


# ---------------------------------------------------------
# Unsafe / inappropriate query detection
# ---------------------------------------------------------

UNSAFE_PATTERNS = [
    # Violence / physical harm
    r"\bkill\b",
    r"\bmurder\b",
    r"\bbomb\b",
    r"\bweapon\b",
    r"\bexplosive\b",

    # Cyber abuse
    r"\bhack\b",
    r"\bhacking\b",
    r"\bmalware\b",
    r"\bransomware\b",
    r"\bpassword\s*steal\b",

    # Explicit sexual content
    r"\bporn\b",
    r"\bsexual\s+exploitation\b",

    # Telugu examples
    r"చంపడం",
    r"బాంబు తయారు",
    r"హ్యాక్ చేయడం",
    r"మాల్వేర్",
    r"పాస్‌వర్డ్ దొంగిల",
]


def is_unsafe_query(query: str) -> bool:
    """
    Detect clearly unsafe or inappropriate requests.

    This is intentionally conservative:
    normal educational questions should not be blocked
    merely because they contain a sensitive word.
    """

    text = query.strip().lower()

    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, text):
            return True

    return False


# ---------------------------------------------------------
# Query term extraction
# ---------------------------------------------------------

def extract_query_terms(query: str) -> list[str]:
    """
    Extract meaningful Telugu/English words from the query.
    """

    phrases = [
        "అంటే ఏమిటి?",
        "అంటే ఏమిటి",
        "ఏమిటి?",
        "ఏమిటి",
        "ఎవరు?",
        "ఎవరు",
        "ఎప్పుడు?",
        "ఎప్పుడు",
        "ఎక్కడ?",
        "ఎక్కడ",
        "ఎలా?",
        "ఎలా"
    ]

    text = query.strip()

    for phrase in phrases:
        text = text.replace(phrase, " ")

    words = re.findall(
        r"[\u0C00-\u0C7F]+|[A-Za-z]+",
        text
    )

    return [
        word
        for word in words
        if len(word) >= 2
    ]


# ---------------------------------------------------------
# Retrieval guardrail
# ---------------------------------------------------------

def check_retrieval(
    query: str,
    results: List[Dict]
) -> Dict:
    """
    Validate retrieved context using:

    1. Unsafe-query detection
    2. Retrieval availability
    3. Semantic similarity
    4. Query-term relevance
    """

    query = query.strip()

    # -----------------------------------------------------
    # Safety check
    # -----------------------------------------------------

    if is_unsafe_query(query):
        return {
            "allowed": False,
            "reason": "unsafe_query",
            "best_similarity": 0.0,
            "matching_terms": []
        }

    # -----------------------------------------------------
    # Retrieval availability
    # -----------------------------------------------------

    if not results:
        return {
            "allowed": False,
            "reason": "no_results",
            "best_similarity": 0.0,
            "matching_terms": []
        }

    valid_results = [
        result
        for result in results
        if result.get("similarity") is not None
    ]

    if not valid_results:
        return {
            "allowed": False,
            "reason": "no_similarity_score",
            "best_similarity": 0.0,
            "matching_terms": []
        }

    # -----------------------------------------------------
    # Best similarity
    # -----------------------------------------------------

    best_result = max(
        valid_results,
        key=lambda x: float(
            x.get("similarity", 0.0)
        )
    )

    best_similarity = float(
        best_result.get("similarity", 0.0)
    )

    # -----------------------------------------------------
    # Query terms
    # -----------------------------------------------------

    query_terms = extract_query_terms(query)

    matching_terms = []

    for result in valid_results:

        text = str(
            result.get("text", "")
        ).lower()

        for term in query_terms:

            if term.lower() in text:
                if term not in matching_terms:
                    matching_terms.append(term)

    # -----------------------------------------------------
    # Similarity guardrail
    # -----------------------------------------------------

    if best_similarity < SIMILARITY_THRESHOLD:
        return {
            "allowed": False,
            "reason": "low_similarity",
            "best_similarity": best_similarity,
            "matching_terms": matching_terms
        }

    # -----------------------------------------------------
    # Query-term relevance guardrail
    # -----------------------------------------------------

    if query_terms and not matching_terms:
        return {
            "allowed": False,
            "reason": "no_query_term_match",
            "best_similarity": best_similarity,
            "matching_terms": matching_terms
        }

    # -----------------------------------------------------
    # Relevant grounded context found
    # -----------------------------------------------------

    return {
        "allowed": True,
        "reason": "relevant_context_found",
        "best_similarity": best_similarity,
        "matching_terms": matching_terms
    }


# ---------------------------------------------------------
# Safe response
# ---------------------------------------------------------

def guardrail_response(check: Dict) -> str:
    """
    Return a safe response based on the guardrail reason.
    """

    reason = check.get("reason")

    if reason == "unsafe_query":
        return (
            "క్షమించండి, ఈ రకమైన అభ్యర్థనకు నేను సహాయం చేయలేను."
        )

    if reason == "no_results":
        return (
            "క్షమించండి, మీ ప్రశ్నకు సంబంధిత సమాచారం "
            "అందుబాటులో లేదు."
        )

    if reason == "low_similarity":
        return (
            "క్షమించండి, అందుబాటులో ఉన్న సమాచారంలో "
            "మీ ప్రశ్నకు సరిపడిన సమాచారం లేదు."
        )

    if reason == "no_query_term_match":
        return (
            "క్షమించండి, అందుబాటులో ఉన్న సమాచారంలో "
            "మీ ప్రశ్నకు సరిపడిన సమాచారం లేదు."
        )

    return (
        "క్షమించండి, మీ ప్రశ్నకు సమాధానం ఇవ్వడానికి "
        "తగిన సమాచారం అందుబాటులో లేదు."
    )