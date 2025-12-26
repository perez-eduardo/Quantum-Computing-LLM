"""
RAG Pipeline for Quantum Computing Assistant.
Combines retrieval and inference for end-to-end Q&A.

Usage:
    from pipeline import QuantumRAGPipeline
    
    pipeline = QuantumRAGPipeline()
    response = pipeline.query("What is a qubit?")
    
    print(response["answer"])
    print(response["suggested_questions"])
"""

from typing import List, Dict, Optional
from difflib import SequenceMatcher

from retrieval import Retriever
from inference import QuantumInference


class QuantumRAGPipeline:
    """End-to-end RAG pipeline for quantum computing Q&A."""
    
    def __init__(
        self,
        model_path: str = None,
        tokenizer_path: str = None,
        device: str = None
    ):
        """
        Initialize pipeline.
        
        Args:
            model_path: Path to model checkpoint
            tokenizer_path: Path to tokenizer JSON
            device: 'cuda' or 'cpu'
        """
        print("Initializing RAG pipeline...")
        
        # Initialize components
        self.retriever = Retriever()
        self.inferencer = QuantumInference(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            device=device
        )
        
        print("Pipeline ready!")
    
    def query(
        self,
        question: str,
        top_k_retrieval: int = 5,
        top_k_context: int = 3,
        max_new_tokens: int = 150,
        temperature: float = 0.7
    ) -> Dict:
        """
        Answer a question using RAG.
        
        Args:
            question: User's question
            top_k_retrieval: Number of Q&A pairs to retrieve
            top_k_context: Number of pairs to use as context
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Dict with keys:
                - answer: Generated answer text
                - sources: Retrieved Q&A pairs used as context
                - suggested_questions: Follow-up question suggestions
        """
        # Step 1: Retrieve relevant Q&A pairs
        retrieved = self.retriever.search(question, top_k=top_k_retrieval)
        
        if not retrieved:
            return {
                "answer": "I don't have enough information to answer that question.",
                "sources": [],
                "suggested_questions": []
            }
        
        # Step 2: Build context from top results
        context_pairs = retrieved[:top_k_context]
        context = self._build_context(context_pairs)
        
        # Step 3: Build prompt
        prompt = f"Context: {context} Question: {question} Answer:"
        
        # Step 4: Generate answer
        generated = self.inferencer.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        answer = self.inferencer.extract_answer(generated)
        
        # Step 5: Get suggested questions
        suggested = self._get_suggestions(question, retrieved)
        
        # Step 6: Format sources
        sources = [
            {
                "question": r["question"],
                "answer": r["answer"][:200] + "..." if len(r["answer"]) > 200 else r["answer"],
                "source": r["source"],
                "similarity": round(r["similarity"], 4)
            }
            for r in context_pairs
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "suggested_questions": suggested
        }
    
    def _build_context(self, pairs: List[Dict]) -> str:
        """
        Build context string from Q&A pairs.
        
        Format: Q: [q1] A: [a1] Q: [q2] A: [a2] ...
        """
        context_parts = []
        
        for p in pairs:
            q = p["question"]
            a = p["answer"]
            
            # Truncate long answers to fit context window
            if len(a) > 300:
                a = a[:300] + "..."
            
            context_parts.append(f"Q: {q} A: {a}")
        
        return " ".join(context_parts)
    
    def _get_suggestions(
        self,
        original_question: str,
        retrieved: List[Dict],
        max_suggestions: int = 3
    ) -> List[str]:
        """
        Get suggested follow-up questions from retrieved pairs.
        
        Filters out questions too similar to the original.
        """
        suggestions = []
        original_lower = original_question.lower()
        
        for r in retrieved:
            q = r["question"]
            
            if not q:
                continue
            
            # Skip if too similar to original
            similarity = self._text_similarity(original_lower, q.lower())
            if similarity > 0.6:
                continue
            
            # Skip if already in suggestions (dedup)
            is_duplicate = False
            for s in suggestions:
                if self._text_similarity(s.lower(), q.lower()) > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                suggestions.append(q)
            
            if len(suggestions) >= max_suggestions:
                break
        
        return suggestions
    
    def _text_similarity(self, a: str, b: str) -> float:
        """Calculate text similarity ratio (0-1)."""
        return SequenceMatcher(None, a, b).ratio()


def test_pipeline():
    """Test the full RAG pipeline."""
    print("=" * 60)
    print("RAG PIPELINE TEST")
    print("=" * 60)
    
    pipeline = QuantumRAGPipeline()
    
    test_questions = [
        "What is a qubit?",
        "How does quantum entanglement work?",
        "What is Shor's algorithm used for?",
        "Why is quantum error correction important?",
        "What is superposition?"
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print("=" * 60)
        
        response = pipeline.query(question)
        
        print(f"\nANSWER:")
        print(f"  {response['answer'][:400]}")
        
        print(f"\nSOURCES ({len(response['sources'])}):")
        for i, src in enumerate(response['sources'], 1):
            print(f"  [{i}] {src['source']} (sim={src['similarity']})")
            print(f"      Q: {src['question'][:60]}...")
        
        print(f"\nSUGGESTED QUESTIONS:")
        for i, sq in enumerate(response['suggested_questions'], 1):
            print(f"  {i}. {sq}")


if __name__ == "__main__":
    test_pipeline()
