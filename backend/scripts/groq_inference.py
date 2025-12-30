"""
Groq API inference for Quantum Computing LLM.
Uses Llama 3.3 70B Versatile for fast responses.
"""

from groq import Groq

SYSTEM_PROMPT = """You are a quantum computing assistant for beginners. 
Answer using the provided context. Keep explanations simple and accessible.
Do not use complex math or equations. Be concise but thorough."""


class GroqInference:
    """Inference wrapper for Groq API (Llama 3.3 70B)."""
    
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 300
    ):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(self, context: str, question: str) -> str:
        """Generate answer using Groq API."""
        user_content = f"Context:\n{context}\n\nQuestion: {question}"
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content
    
    def extract_answer(self, generated_text: str) -> str:
        """Extract answer from generated text."""
        return generated_text.strip()
