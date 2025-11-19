import os
import time
import openai
from typing import List, Dict, Optional

class LLMClient:
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        temperature: float = 0.7
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_retries = max_retries
        self.temperature = temperature

        openai.api_key = self.api_key

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

        while retries < self.max_retries:
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature
                )

                return response.choices[0].message["content"]

            except Exception as e:
                print(f"[LLM ERROR] {e}")
                retries += 1
                time.sleep(1)

        raise RuntimeError("LLM failed after retries")
