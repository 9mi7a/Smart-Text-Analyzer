import os
import time
import google.generativeai as genai
from typing import List, Dict, Optional
from dotenv import load_dotenv
load_dotenv()
class LLMClient:
    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        temperature: float = 0.7
    ):
        self.model = model
        self.api_key = os.getenv("GEMINI_API_KEY")


        self.max_retries = max_retries
        self.temperature = temperature

        # Configure the Gemini client
        genai.configure(api_key=self.api_key)

        # Load the model
        self.client = genai.GenerativeModel(self.model)

    # ==========
    # MAIN METHOD
    # ==========
    def chat(self, messages: List[Dict]) -> str:
        """
        messages = [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]
        """
        retries = 0

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
