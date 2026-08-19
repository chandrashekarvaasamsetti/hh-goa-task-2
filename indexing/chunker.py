import re
from typing import List, Dict


def clean_text(text: str) -> str:
    """Clean extra whitespace from text."""
    return re.sub(r"\s+", " ", text).strip()


def passage_chunk(text: str) -> List[str]:
    """Strategy 1: Keep the complete passage as one chunk."""
    text = clean_text(text)

    if not text:
        return []

    return [text]


def sentence_chunk(
    text: str,
    max_chars: int = 1200
) -> List[str]:
    """
    Strategy 2: Sentence-aware chunking.
    Keeps sentences together until the chunk reaches max_chars.
    """
    text = clean_text(text)

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?।])\s+", text)

    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()

    if current:
        chunks.append(current.strip())

    return chunks


def fixed_overlap_chunk(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150
) -> List[str]:
    """
    Strategy 3: Fixed-size character chunks with overlap.
    """
    text = clean_text(text)

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def adaptive_chunk(text: str) -> List[str]:
    """
    Strategy 4: Adaptive/hybrid chunking.

    Short passages stay intact.
    Medium passages use sentence-aware chunks.
    Very long passages use smaller sentence-aware chunks.
    """
    text = clean_text(text)

    if not text:
        return []

    length = len(text)

    if length <= 800:
        return passage_chunk(text)

    if length <= 4000:
        return sentence_chunk(text, max_chars=1200)

    return sentence_chunk(text, max_chars=800)


def create_chunks(
    text: str,
    query_id: int,
    passage_id: int,
    source_lang: str,
    target_lang: str,
    query_type: str,
    is_selected: int
) -> List[Dict]:

    strategies = {
        "passage": passage_chunk(text),
        "sentence": sentence_chunk(text),
        "fixed_overlap": fixed_overlap_chunk(text),
        "adaptive": adaptive_chunk(text),
    }

    results = []

    for strategy_name, chunks in strategies.items():

        for chunk_id, chunk in enumerate(chunks):

            results.append({
                "query_id": query_id,
                "passage_id": passage_id,
                "chunk_id": chunk_id,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "query_type": query_type,
                "is_selected": is_selected,
                "chunk_strategy": strategy_name,
                "text": chunk
            })

    return results


if __name__ == "__main__":

    sample = (
        "కార్పొరేషన్ అనేది ఒక సంస్థగా వ్యవహరించడానికి "
        "మరియు చట్టంలో గుర్తింపు పొందిన వ్యక్తుల సమూహం. "
        "ఇది తన సభ్యుల నుండి ప్రత్యేకమైన చట్టపరమైన గుర్తింపును కలిగి ఉంటుంది."
    )

    chunks = create_chunks(
        text=sample,
        query_id=1,
        passage_id=1,
        source_lang="tel",
        target_lang="tel",
        query_type="definition",
        is_selected=1
    )

    for item in chunks:
        print("\n---", item["chunk_strategy"], "---")
        print(item["text"])