from abc import ABC, abstractmethod
from typing import List, Dict

class BaseLLMClient(ABC):

    @abstractmethod
    def chat(self, messages: List[Dict]) -> str:
        pass
