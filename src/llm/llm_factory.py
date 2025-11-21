import os

from dotenv import load_dotenv

from .gemini_client import GeminiClient
from .openai_client import OpenAIClient

def create_llm(provider: str, model: str, temperature=0.7):
    load_dotenv()
    if provider == "gemini":
        return GeminiClient(
            model=model,
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature
        )

    if provider == "openai":
        return OpenAIClient(
            model=model,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=temperature
        )

    raise ValueError("Unknown provider: " + provider)
