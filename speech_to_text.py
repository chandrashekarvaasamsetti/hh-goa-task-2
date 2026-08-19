import os
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

if not SARVAM_API_KEY:
    raise RuntimeError("SARVAM_API_KEY is not configured in .env")


client = SarvamAI(
    api_subscription_key=SARVAM_API_KEY
)


def transcribe_audio(audio_path: str) -> str:
    """
    Convert Telugu speech audio into Telugu text using Sarvam Saaras v3.
    """

    with open(audio_path, "rb") as audio_file:
        response = client.speech_to_text.transcribe(
            file=audio_file,
            model="saaras:v3",
            mode="transcribe",
            language_code="te-IN"
        )

    return response.transcript


if __name__ == "__main__":
    print("Sarvam Telugu STT module loaded successfully.")