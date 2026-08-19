import os
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from indexing.chunker import adaptive_chunk


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

DATA_PATH = "data/validation/telval.parquet"
INDEX_DIR = "data/index"

MAX_PASSAGES = 2000


def load_passages():
    """Load the prototype passages and apply adaptive chunking."""

    df = pd.read_parquet(
        DATA_PATH,
        columns=[
            "query",
            "query_id",
            "query_type",
            "source_lang",
            "target_lang",
            "passages"
        ]
    )

    records = []
    passage_count = 0

    for _, row in df.iterrows():

        passages = row["passages"]

        translated = passages["Translated_passages"]
        selected = passages["is_selected"]

        for passage_id, text in enumerate(translated):

            if not text or not text.strip():
                continue

            if passage_count >= MAX_PASSAGES:
                return records

            chunks = adaptive_chunk(text)

            for chunk_id, chunk in enumerate(chunks):

                records.append({
                    "query_id": int(row["query_id"]),
                    "query_type": row["query_type"],
                    "source_lang": row["source_lang"],
                    "target_lang": row["target_lang"],
                    "passage_id": passage_id,
                    "chunk_id": chunk_id,
                    "chunk_strategy": "adaptive",
                    "is_selected": int(selected[passage_id]),
                    "text": chunk.strip()
                })

            passage_count += 1

    return records


def build_index(records):

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    texts = [
        record["text"]
        for record in records
    ]

    print(
        "Embedding",
        len(texts),
        "adaptive chunks..."
    )

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    os.makedirs(
        INDEX_DIR,
        exist_ok=True
    )

    index_path = os.path.join(
        INDEX_DIR,
        "telugu_prototype.faiss"
    )

    metadata_path = os.path.join(
        INDEX_DIR,
        "telugu_prototype.json"
    )

    faiss.write_index(
        index,
        index_path
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            records,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("================================")
    print("ADAPTIVE INDEX CREATED")
    print("================================")
    print("Vectors:", index.ntotal)
    print("Dimension:", dimension)
    print("Strategy: adaptive")
    print("Index:", index_path)
    print("Metadata:", metadata_path)


if __name__ == "__main__":

    records = load_passages()

    print(
        "Adaptive chunks loaded:",
        len(records)
    )

    build_index(records)