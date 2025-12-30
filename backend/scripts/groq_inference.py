"""
Groq API inference for Quantum Computing LLM.
Uses Llama 3.3 70B Versatile for fast responses.

Usage:
    from groq_inference import GroqInference
    
    inferencer = GroqInference()
    answer = inferencer.generate(context, question)
"""

import os
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

from base_inference import BaseLLM

# Load environment variables
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Groq settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# System prompt for quantum computing assistant
SYSTEM_PROMPT = """You are a quantum computing assistant for beginners. 
Answer using the provided context. Keep explanations simple and accessible.
Do not use complex math or equations. Be concise but thorough."""


class GroqInference(BaseLLM):
    """Inference wrapper for Groq API (Llama 3.3 70B)."""
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        temperature: float = 0.2,
        max_tokens: int = 300
    ):
        """
        Initialize Groq inference.
        
        Args:
            api_key: Groq API key (uses env var if None)
            model: Model name (default: llama-3.3-70b-versatile)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.api_key = api_key or GROQ_API_KEY
        self.model = model or GROQ_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found. Set it in .env or pass api_key parameter.")
        
        self.client = Groq(api_key=self.api_key)
        print(f"Groq client initialized with model: {self.model}")
    
    @property
    def name(self) -> str:
        return "groq"
    
    def generate(self, context: str, question: str) -> str:
        """
        Generate answer using Groq API.
        
        Args:
            context: Retrieved Q&A pairs (e.g., "Q: ... A: ... Q: ... A: ...")
            question: User's question
        
        Returns:
            Generated answer text
        """
        # Format context for readability
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
        """
        Extract answer from generated text.
        
        Groq returns clean text, no extraction needed.
        
        Args:
            generated_text: Output from generate()
        
        Returns:
            Clean answer string
        """
        return generated_text.strip()


def test_groq_inference():
    """Test Groq inference with sample prompt."""
    print("=" * 60)
    print("GROQ INFERENCE TEST")
    print("=" * 60)
    
    try:
        inferencer = GroqInference()
    except ValueError as e:
        print(f"ERROR: {e}")
        print("Set GROQ_API_KEY in .env to test")
        return
    
    # Test with context
    context = "Q: What is superposition? A: Superposition allows a qubit to exist in multiple states simultaneously until measured."
    question = "What is a qubit?"
    
    print(f"\nContext: {context[:80]}...")
    print(f"Question: {question}")
    print("\nGenerating...")
    
    generated = inferencer.generate(context, question)
    answer = inferencer.extract_answer(generated)
    
    print(f"\nAnswer: {answer}")
    print(f"LLM used: {inferencer.name}")


if __name__ == "__main__":
    test_groq_inference()
