"""
Custom model inference for Quantum Computing LLM.
Loads 125.8M parameter model and generates text from prompts.

Usage:
    from inference import QuantumInference
    
    inferencer = QuantumInference()
    answer = inferencer.generate(context, question)
"""

import sys
from pathlib import Path

import torch

from base_inference import BaseLLM

# Add training/scripts to path for model import
PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAINING_SCRIPTS = PROJECT_ROOT / "training" / "scripts"
sys.path.insert(0, str(TRAINING_SCRIPTS))

from model import QuantumLLM

# Default paths (relative to project root)
DEFAULT_MODEL_PATH = PROJECT_ROOT / "training" / "model" / "final_model.pt"
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "training" / "tokenizer" / "tokenizer.json"


class QuantumInference(BaseLLM):
    """Inference wrapper for custom 125.8M parameter Quantum Computing LLM."""
    
    def __init__(
        self,
        model_path: str = None,
        tokenizer_path: str = None,
        device: str = None,
        temperature: float = 0.2,
        top_k: int = 30,
        max_new_tokens: int = 150
    ):
        """
        Initialize inference module.
        
        Args:
            model_path: Path to model checkpoint (.pt file)
            tokenizer_path: Path to tokenizer JSON file
            device: 'cuda' or 'cpu' (auto-detected if None)
            temperature: Sampling temperature (lower = more focused)
            top_k: Top-k sampling parameter
            max_new_tokens: Maximum tokens to generate
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else DEFAULT_TOKENIZER_PATH
        self.temperature = temperature
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        
        # Device selection
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
        self._load_tokenizer()
    
    @property
    def name(self) -> str:
        return "custom"
    
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
    
    def generate(self, context: str, question: str) -> str:
        """
        Generate answer given RAG context and question.
        
        Args:
            context: Retrieved Q&A pairs (e.g., "Q: ... A: ... Q: ... A: ...")
            question: User's question
        
        Returns:
            Full generated text including prompt
        """
        # Build prompt in format model was trained on
        prompt = f"Context: {context} Question: {question} Answer:"
        
        # Tokenize
        tokens = self.tokenizer.encode(prompt).ids
        x = torch.tensor([tokens], device=self.device)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                x,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k
            )
        
        # Decode
        generated = self.tokenizer.decode(output[0].tolist())
        return generated
    
    def extract_answer(self, generated_text: str) -> str:
        """
        Extract just the answer portion from generated text.
        
        The model generates:
        Context: ... Question: [user Q] Answer: [answer] Question: [new Q] Answer: [junk]...
        
        We want ONLY the first answer (response to user's question).
        
        Args:
            generated_text: Full generated text including prompt
        
        Returns:
            Just the first answer portion
        """
        question_marker = "Question:"
        answer_marker = "Answer:"
        
        # Find the FIRST "Question:" (user's question from the prompt)
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
    
    # Test with context
    context = "Q: What is superposition? A: Superposition allows a qubit to exist in multiple states simultaneously until measured."
    question = "What is a qubit?"
    
    print(f"\nContext: {context[:80]}...")
    print(f"Question: {question}")
    
    generated = inferencer.generate(context, question)
    answer = inferencer.extract_answer(generated)
    
    print(f"Answer: {answer[:300]}")
    print(f"LLM used: {inferencer.name}")


if __name__ == "__main__":
    test_inference()
