import openai
from openai import OpenAI
from .base_client import BaseLLMClient
import time 
class OpenAIClient(BaseLLMClient):
    def __init__(self, model, api_key, temperature=0.7):
        self.max_retries = 3
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def chat(self, messages) :
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
                time.sleep(3)

        raise RuntimeError("LLM failed after retries")
