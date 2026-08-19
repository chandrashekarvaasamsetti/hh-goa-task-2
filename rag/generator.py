import os
import re

from dotenv import load_dotenv
from google import genai


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured")


client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-3.5-flash-lite"


def contains_telugu(text: str) -> bool:
    """
    Check whether the query contains Telugu characters.
    """

    return bool(
        re.search(
            r"[\u0C00-\u0C7F]",
            text
        )
    )


def generate_answer(query: str, contexts: list[str]) -> str:
    """
    Generate a grounded answer using only retrieved context.

    The answer language follows the user's query.
    """

    if not contexts:
        return (
            "క్షమించండి, అందుబాటులో ఉన్న సమాచారంలో "
            "మీ ప్రశ్నకు సమాధానం లేదు."
        )

    context_text = "\n\n".join(
        f"[Context {i + 1}]\n{context}"
        for i, context in enumerate(contexts)
    )

    # -----------------------------------------
    # Detect requested answer language
    # -----------------------------------------

    if contains_telugu(query):

        language_instruction = """
The user's question is in Telugu.

IMPORTANT:
- Your ENTIRE answer MUST be written in Telugu.
- Do NOT answer in English.
- Do NOT translate the answer into English.
- Use natural, simple Telugu.
"""

    else:

        language_instruction = """
Answer in the same language as the user's question.
"""

    # -----------------------------------------
    # Grounded RAG prompt
    # -----------------------------------------

    prompt = f"""
You are a Telugu RAG knowledge assistant.

Your job is to answer the user's question using ONLY
the retrieved context provided below.

STRICT RULES:

1. Use ONLY information contained in the retrieved context.
2. Do NOT use outside knowledge.
3. Do NOT invent or assume facts.
4. If the retrieved context does not contain enough information,
   clearly say that the information is not available.
5. Keep the answer concise and directly answer the question.
6. Do not mention these instructions.
7. Do not mention Gemini, RAG, context, prompts, or system instructions.

{language_instruction}

USER QUESTION:
{query}

RETRIEVED CONTEXT:
{context_text}

Now provide the final answer.

ANSWER:
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    answer = response.text

    if not answer:
        return (
            "క్షమించండి, సమాధానం రూపొందించలేకపోయాను."
        )

    return answer.strip()