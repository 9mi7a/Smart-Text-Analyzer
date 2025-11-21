import json
import re
from pathlib import Path

from .llm_client import LLMClient

BASE_DIR = Path(__file__).resolve().parent.parent  # → src/
PROMPTS_DIR = BASE_DIR / "prompts"                 # → src/prompts/


def load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    return path.read_text(encoding="utf-8")


def wrap(content: str, role: str):
    """Convert OpenAI-style content into Gemini parts format."""
    return {
        "role": role,
        "parts": [{"text": content}]
    }


class TextAnalyzerChain:
    def __init__(self, model="gemini-2.5-flash-lite"):
        self.client = LLMClient(model=model)

        # Preload prompts
        self.system_prompt = load_prompt("system.txt")
        self.core_prompt = load_prompt("analyze_core.txt")
        self.bias_prompt = load_prompt("analyze_bias.txt")

    def run_core_analysis(self, text: str) -> dict:
        messages = [
            wrap(self.system_prompt, "user"),
            wrap(self.core_prompt, "user"),
            wrap(f"TEXT TO ANALYZE:\n{text}", "user")
        ]

        response = self.client.chat(messages)
        return self.safe_parse_json(response)

    def run_bias_analysis(self, text: str) -> dict:
        messages = [
            wrap(self.system_prompt, "user"),
            wrap(self.bias_prompt, "user"),
            wrap(f"TEXT TO ANALYZE:\n{text}", "user")
        ]
        response = self.client.chat(messages)
        return self.safe_parse_json(response)

    def safe_parse_json(self, raw_output: str) -> dict:

        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output.strip(), flags=re.IGNORECASE)

        try:
            parsed = json.loads(cleaned)
            return parsed
        except json.JSONDecodeError:
            return {"error": "Invalid JSON returned by LLM", "raw_output": raw_output}

    def analyze(self, text: str) ->dict:
        core = self.run_core_analysis(text)
        bias = self.run_bias_analysis(text)
        print("Core:")
        print(core)
        print("Bias:")
        print(bias)
        return core | bias
    def output(self,text):

        output = self.analyze(text)
        print ("output ",output)
        print("-----------------------------------")
        print(json.dumps(output, indent=2))
