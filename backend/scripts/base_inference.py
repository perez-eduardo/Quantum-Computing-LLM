"""
Abstract base class for LLM inference.
Provides common interface for custom model and Groq API.
"""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""
    
    @abstractmethod
    def generate(self, context: str, question: str) -> str:
        """
        Generate answer given RAG context and question.
        
        Args:
            context: Retrieved Q&A pairs formatted as context string
            question: User's question
        
        Returns:
            Generated text (may need extraction)
        """
        pass
    
    @abstractmethod
    def extract_answer(self, generated_text: str) -> str:
        """
        Extract clean answer from generated text.
        
        Args:
            generated_text: Raw output from generate()
        
        Returns:
            Clean answer string
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return identifier for this LLM (e.g., 'custom', 'groq')."""
        pass
