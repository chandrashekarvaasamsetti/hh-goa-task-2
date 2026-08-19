# HH GOA Telugu RAG Assistant

A Telugu Voice and Knowledge Retrieval System using Retrieval-Augmented Generation (RAG), FAISS semantic search, Gemini, retrieval guardrails, and Telugu speech-to-text.

## 1. Project Overview

HH GOA Telugu RAG Assistant accepts text and voice queries and retrieves relevant information from the indexed knowledge base.

The system provides:

- Text-based queries
- Telugu voice input
- Semantic retrieval
- Retrieval guardrails
- Grounded RAG answer generation
- Retrieved supporting information
- Telugu-language responses

## 2. Architecture

User Query
    |
    v
Frontend
    |
    v
FastAPI Backend
    |
    v
FAISS Retrieval
    |
    v
Retrieval Guardrails
    |
    v
Relevant Context
    |
    v
Gemini RAG Generation
    |
    v
Answer + Supporting Information
    |
    v
Frontend

Voice Flow:

Telugu Voice
    |
    v
Sarvam Speech-to-Text
    |
    v
Telugu Text Query
    |
    v
RAG Pipeline
    |
    v
Answer

## 3. Main Components

### Semantic Retrieval

FAISS is used for vector similarity search.

Embedding model:

sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

The retrieval system also applies reranking using similarity, lexical overlap, definition matching, selected-passage information, and rank signals.

### RAG Generation

Gemini generates the final response using retrieved context.

The generator is designed to:

- Use retrieved context
- Avoid unsupported facts
- Avoid inventing information
- Follow the user's language
- Provide concise answers

### Retrieval Guardrails

The guardrail layer checks:

- Unsafe queries
- Empty retrieval results
- Missing similarity scores
- Low similarity
- Query-term relevance

Similarity threshold:

0.65

If the retrieved information is insufficient or irrelevant, the system rejects the response instead of generating unsupported information.

### Telugu Speech-to-Text

Telugu speech is converted to text using Sarvam Saaras v3.

Language:

te-IN

### Frontend

The web interface provides:

- Text input
- Voice input
- Search
- Answer display
- Supporting retrieved information

## 4. Technologies

- Python
- FastAPI
- Uvicorn
- FAISS
- Sentence Transformers
- Google Gemini API
- Sarvam AI
- HTML
- CSS
- JavaScript
- Pytest

## 5. Project Structure

HH GOA TASK-2/

    backend/
        main.py

    frontend/
        index.html

    guardrails/
        check.py

    indexing/
        chunker.py

    rag/
        generator.py

    retrieval/
        search.py

    data/
        index/
            telugu_prototype.faiss
            telugu_prototype.json

    tests/
        test_guardrails.py

    benchmark_latency.py
    speech_to_text.py
    test_request.json
    requirements.txt
    README.md
    .env

## 6. Environment Variables

Create a .env file in the project root.

Required variables:

GEMINI_API_KEY=your_gemini_api_key
SARVAM_API_KEY=your_sarvam_api_key

Do not commit API keys or other secrets.

## 7. Installation

Create the virtual environment:

    python -m venv venv

Activate it on Windows:

    .\venv\Scripts\Activate.ps1

Install dependencies:

    pip install -r requirements.txt

## 8. Run Backend

From the project root:

    .\venv\Scripts\python.exe -m uvicorn backend.main:app --reload

Backend:

    http://127.0.0.1:8000

FastAPI documentation:

    http://127.0.0.1:8000/docs

## 9. Run Frontend

Open another terminal:

    .\venv\Scripts\python.exe -m http.server 5500 --directory frontend

Frontend:

    http://127.0.0.1:5500

The frontend uses:

    http://127.0.0.1:8000/search
    http://127.0.0.1:8000/transcribe

## 10. API Endpoints

### GET /

Checks whether the backend is running.

### POST /search

Accepts a text query and returns:

- Query
- Generated answer
- Retrieved results
- Guardrail information

Example:

    {
        "query": "Example question"
    }

### POST /transcribe

Accepts an audio file and converts Telugu speech into text.

## 11. Testing

Automated guardrail tests are implemented using Pytest.

Run:

    .\venv\Scripts\python.exe -m pytest -v

Verified result:

    6 passed

The tests cover:

1. Unsafe query blocking
2. Empty retrieval handling
3. Low similarity blocking
4. Relevant context acceptance
5. Query-term extraction
6. Query-term mismatch blocking

## 12. Retrieval Latency

The retrieval benchmark was executed using 10 queries.

Results:

    Queries: 10
    Minimum: 60.94 ms
    Average: 77.38 ms
    P50: 77.90 ms
    P70: 81.75 ms
    P100: 87.68 ms

Average retrieval latency:

    77.38 ms

## 13. Functional Verification

The following components have been tested:

- Backend startup
- FastAPI Swagger interface
- Search API
- Semantic retrieval
- Retrieval guardrails
- RAG generation
- Frontend text search
- Telugu speech-to-text
- Supporting information display
- Automated guardrail tests
- Retrieval latency benchmark

## 14. Safety and Grounding

The system uses retrieval guardrails to reduce unsupported answers.

When retrieved information is insufficient or irrelevant, the system can reject the query instead of generating an unsupported answer.

## 15. Limitations

Answer quality depends on the information contained in the indexed knowledge base.

If the knowledge base does not contain sufficient relevant information, the guardrail may reject the query.

This behavior is intentional because the system is designed to prioritize grounded responses.

## 16. Project Status

The project currently includes:

- Retrieval pipeline
- Reranking
- RAG generation
- Retrieval guardrails
- FastAPI backend
- Telugu speech-to-text
- Web frontend
- Automated tests
- Latency benchmarking
- Requirements documentation
- Project README