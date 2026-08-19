from guardrails.check import (
    is_unsafe_query,
    extract_query_terms,
    check_retrieval,
)


def test_unsafe_query_is_blocked():
    result = check_retrieval(
        "How to hack a computer",
        [
            {
                "similarity": 0.90,
                "text": "Computer security information."
            }
        ]
    )

    assert result["allowed"] is False
    assert result["reason"] == "unsafe_query"


def test_empty_results_are_blocked():
    result = check_retrieval(
        "Who is Krishna?",
        []
    )

    assert result["allowed"] is False
    assert result["reason"] == "no_results"


def test_low_similarity_is_blocked():
    result = check_retrieval(
        "Who is Krishna?",
        [
            {
                "similarity": 0.40,
                "text": "Information about Krishna."
            }
        ]
    )

    assert result["allowed"] is False
    assert result["reason"] == "low_similarity"


def test_relevant_context_is_allowed():
    result = check_retrieval(
        "Krishna",
        [
            {
                "similarity": 0.86,
                "text": "Krishna is a central figure in this knowledge source."
            }
        ]
    )

    assert result["allowed"] is True
    assert result["reason"] == "relevant_context_found"


def test_query_term_extraction():
    terms = extract_query_terms("Who is Krishna?")

    assert "Krishna" in terms


def test_missing_query_term_is_blocked():
    result = check_retrieval(
        "Krishna",
        [
            {
                "similarity": 0.90,
                "text": "This text is about a completely different topic."
            }
        ]
    )

    assert result["allowed"] is False
    assert result["reason"] == "no_query_term_match"