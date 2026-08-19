from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)

from retrieval.search import search
from rag.generator import generate_answer
from guardrails.check import check_retrieval, guardrail_response
from speech_to_text import transcribe_audio


app = FastAPI(
    title="HH GOA Telugu RAG Assistant"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {
        "message": "HH GOA Telugu RAG Assistant is running"
    }


@app.post("/search")
def search_api(request: QueryRequest):

    query = request.query.strip()

    if not query:
        return {
            "query": query,
            "answer": "దయచేసి ఒక ప్రశ్నను నమోదు చేయండి.",
            "results": []
        }

    # -----------------------------------------
    # STEP 1: Retrieve relevant contexts
    # -----------------------------------------

    results = search(
        query,
        top_k=5
    )

    # -----------------------------------------
    # STEP 2: Guardrail check
    # -----------------------------------------

    guardrail = check_retrieval(
        query,
        results
    )

    # If retrieval is too weak, do NOT call Gemini
    if not guardrail["allowed"]:
        return {
            "query": query,
            "answer": guardrail_response(guardrail),
            "results": results,
            "guardrail": guardrail
        }

    # -----------------------------------------
    # STEP 3: Extract retrieved contexts
    # -----------------------------------------

    contexts = [
        result["text"]
        for result in results
        if result.get("text")
    ]

    # -----------------------------------------
    # STEP 4: Generate grounded answer
    # -----------------------------------------

    answer = generate_answer(
        query=query,
        contexts=contexts
    )

    # -----------------------------------------
    # STEP 5: Return answer + evidence
    # -----------------------------------------

    return {
        "query": query,
        "answer": answer,
        "results": results,
        "guardrail": guardrail
    }


# -----------------------------------------
# VOICE / SPEECH-TO-TEXT ENDPOINT
# -----------------------------------------

@app.post("/transcribe")
async def transcribe_api(
    file: UploadFile = File(...)
):

    # Use a fixed temporary filename to avoid
    # unsafe filenames being used on the server.
    audio_path = "temp_voice_input.webm"

    try:

        contents = await file.read()

        with open(audio_path, "wb") as f:
            f.write(contents)

        transcript = transcribe_audio(
            audio_path
        )

        return {
            "transcript": transcript
        }

    except Exception as e:

        return {
            "error": str(e)
        }

    finally:

        if os.path.exists(audio_path):
            os.remove(audio_path)