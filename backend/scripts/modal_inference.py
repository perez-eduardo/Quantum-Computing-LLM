"""Modal API inference for Quantum Computing LLM."""

import requests
from typing import Optional


class ModalInference:
    def __init__(self, url: str, timeout: int = 300):
        self.url = url
        self.timeout = timeout
    
    def generate(self, context: str, question: str) -> str:
        """Call Modal API to generate answer."""
        response = requests.post(
            self.url,
            json={"context": context, "question": question},
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        return data.get("answer", "")
