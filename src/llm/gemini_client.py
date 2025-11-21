import google.generativeai as genai
from .base_client import BaseLLMClient
import time

class GeminiClient(BaseLLMClient):
    def __init__(self, model, api_key, temperature=0.7):
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)
        self.temperature = temperature
        self.max_retries=3

    def convert(self, messages):
        """Convert universal messages → Gemini format."""
        parts = []
        for m in messages:
            parts.append({
                "role": m["role"],
                "parts": [{"text": m["content"]}]
            })
        return parts

    def chat(self, messages):
        retries = 0
        messages = self.convert(messages)
        # Gemini expects a single list of message dicts
        while retries < self.max_retries:
            try:
                response = self.client.generate_content(
                    messages,
                    generation_config={"temperature": self.temperature}
                )

                return response.text

            except Exception as e:
                print(f"[LLM ERROR] {e}")
                retries += 1
                time.sleep(3)

        raise RuntimeError("LLM failed after retries")
