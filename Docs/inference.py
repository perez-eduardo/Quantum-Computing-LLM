"""
Inference module for Quantum Computing LLM.
Loads model and tokenizer, generates text from prompts.

Usage:
    from inference import QuantumInference
    
    inferencer = QuantumInference()
    response = inferencer.generate("Context: ... Question: What is a qubit? Answer:")
"""

import os
import sys
from pathlib import Path

import torch

# Add training/scripts to path for model import
PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAINING_SCRIPTS = PROJECT_ROOT / "training" / "scripts"
sys.path.insert(0, str(TRAINING_SCRIPTS))

from model import QuantumLLM

# Default paths (relative to project root)
DEFAULT_MODEL_PATH = PROJECT_ROOT / "training" / "model" / "final_model.pt"
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "training" / "tokenizer" / "tokenizer.json"


class QuantumInference:
    """Inference wrapper for Quantum Computing LLM."""
    
    def __init__(
        self,
        model_path: str = None,
        tokenizer_path: str = None,
        device: str = None
    ):
        """
        Initialize inference module.
        
        Args:
            model_path: Path to model checkpoint (.pt file)
            tokenizer_path: Path to tokenizer JSON file
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else DEFAULT_TOKENIZER_PATH
        
        # Device selection
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
        self._load_tokenizer()
    
    def _load_model(self):
        """Load model from checkpoint."""
        print(f"Loading model: {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = QuantumLLM.load(str(self.model_path), self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded: {self.model.n_params:,} parameters")
    
    def _load_tokenizer(self):
        """Load tokenizer from JSON file."""
        from tokenizers import Tokenizer
        
        print(f"Loading tokenizer: {self.tokenizer_path}")
        
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {self.tokenizer_path}")
        
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        print(f"Tokenizer loaded: {self.tokenizer.get_vocab_size()} tokens")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.2,
        top_k: int = 30
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt (should include Context: and Question:)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
        
        Returns:
            Generated text (full output including prompt)
        """
        # Tokenize prompt
        tokens = self.tokenizer.encode(prompt).ids
        x = torch.tensor([tokens], device=self.device)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
        
        # Decode
        generated = self.tokenizer.decode(output[0].tolist())
        return generated
    
    def extract_answer(self, generated_text: str) -> str:
        """
        Extract just the answer portion from generated text.
        
        Prompt format:
        Context: Q: ... A: ... Q: ... A: ... Question: [user question] Answer:
        
        Model generates:
        [answer] Question: [new Q] Answer: [new A] ...
        
        We want ONLY the first answer (response to user's question).
        
        Args:
            generated_text: Full generated text including prompt
        
        Returns:
            Just the first answer portion
        """
        question_marker = "Question:"
        answer_marker = "Answer:"
        
        # Find the FIRST "Question:" (user's question from the prompt)
        # Context uses "Q:" format, user question uses "Question:" format
        question_idx = generated_text.find(question_marker)
        
        if question_idx == -1:
            # No question marker found, try to find any answer
            answer_idx = generated_text.find(answer_marker)
            if answer_idx == -1:
                return generated_text.strip()
            answer = generated_text[answer_idx + len(answer_marker):].strip()
        else:
            # Find the FIRST "Answer:" after the user's question
            search_start = question_idx + len(question_marker)
            answer_idx = generated_text.find(answer_marker, search_start)
            
            if answer_idx == -1:
                return generated_text[search_start:].strip()
            
            answer = generated_text[answer_idx + len(answer_marker):].strip()
        
        # Stop at markers indicating model is generating new Q&A
        stop_markers = [
            "Question:",  # Model starting new question
            "Q:",         # Alternate question format  
            "Context:",   # Model hallucinating new context
            "\n\n",       # Double newline often indicates new section
        ]
        
        for stop in stop_markers:
            if stop in answer:
                answer = answer[:answer.find(stop)].strip()
        
        return answer


def test_inference():
    """Test inference with sample prompts."""
    print("=" * 60)
    print("INFERENCE TEST")
    print("=" * 60)
    
    inferencer = QuantumInference()
    
    # Test with context prompt
    test_prompts = [
        {
            "context": "Q: What is superposition? A: Superposition allows a qubit to exist in multiple states simultaneously until measured.",
            "question": "What is a qubit?"
        },
        {
            "context": "Q: What is a quantum gate? A: A quantum gate is a basic operation that manipulates qubits.",
            "question": "What is a quantum circuit?"
        }
    ]
    
    for i, test in enumerate(test_prompts, 1):
        prompt = f"Context: {test['context']} Question: {test['question']} Answer:"
        
        print(f"\n--- Test {i} ---")
        print(f"Question: {test['question']}")
        
        generated = inferencer.generate(prompt)
        answer = inferencer.extract_answer(generated)
        
        print(f"Answer: {answer[:300]}")


if __name__ == "__main__":
    test_inference()
