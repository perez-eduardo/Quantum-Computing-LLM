"""Groq API inference for Quantum Computing LLM."""

from groq import Groq

SYSTEM_PROMPT = """You are a quantum computing assistant for beginners. 
Answer using the provided context. Keep explanations simple and accessible.
Do not use complex math or equations. Be concise but thorough."""


class GroqInference:
    def __init__(self, api_key: str, model: str, temperature: float = 0.2, max_tokens: int = 300):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(self, context: str, question: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content.strip()
