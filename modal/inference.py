"""
Modal deployment for Quantum Computing LLM.
Deploy: modal deploy inference.py
Test:   modal run inference.py
"""

import modal

# Configuration
MODEL_DIR = "/vol/model"
GPU_TYPE = "T4"
IDLE_TIMEOUT = 300  # 5 minutes

# Modal App
app = modal.App("quantum-llm")

# Container Image
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.0",
        "tokenizers==0.22.1",
        "fastapi",
    )
)

# Volume for Model Weights
model_volume = modal.Volume.from_name("quantum-model-volume", create_if_missing=True)


# =============================================================================
# MODEL ARCHITECTURE (from training/scripts/model.py)
# =============================================================================

def get_model_classes():
    """Returns all model classes needed for QuantumLLM."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math

    class RMSNorm(nn.Module):
        """Root Mean Square Layer Normalization"""
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
            return (x.float() * norm).type_as(x) * self.weight

    class RotaryEmbedding(nn.Module):
        """Rotary Position Embedding (RoPE)"""
        def __init__(self, dim, max_seq_len=2048):
            super().__init__()
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("inv_freq", inv_freq)
            self.max_seq_len = max_seq_len
            self._build_cache(max_seq_len)

        def _build_cache(self, seq_len):
            t = torch.arange(seq_len, device=self.inv_freq.device)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

        def forward(self, x, seq_len):
            if seq_len > self.max_seq_len:
                self._build_cache(seq_len)
            return (
                self.cos_cached[:, :, :seq_len, :],
                self.sin_cached[:, :, :seq_len, :]
            )

    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin):
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    class Attention(nn.Module):
        """Multi-head attention with RoPE"""
        def __init__(self, config):
            super().__init__()
            self.n_heads = config["n_heads"]
            self.d_model = config["d_model"]
            self.head_dim = self.d_model // self.n_heads

            self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)

            self.rotary = RotaryEmbedding(self.head_dim, config["max_seq_len"])
            self.dropout = nn.Dropout(config["dropout"])

        def forward(self, x, mask=None):
            B, T, C = x.shape

            q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

            cos, sin = self.rotary(x, T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if mask is not None:
                attn = attn.masked_fill(mask == 0, float('-inf'))

            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return self.o_proj(out)

    class SwiGLU(nn.Module):
        """SwiGLU activation for feed-forward network"""
        def __init__(self, config):
            super().__init__()
            self.w1 = nn.Linear(config["d_model"], config["d_ff"], bias=False)
            self.w2 = nn.Linear(config["d_ff"], config["d_model"], bias=False)
            self.w3 = nn.Linear(config["d_model"], config["d_ff"], bias=False)
            self.dropout = nn.Dropout(config["dropout"])

        def forward(self, x):
            return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

    class TransformerBlock(nn.Module):
        """Single transformer block"""
        def __init__(self, config):
            super().__init__()
            self.attn_norm = RMSNorm(config["d_model"])
            self.attn = Attention(config)
            self.ff_norm = RMSNorm(config["d_model"])
            self.ff = SwiGLU(config)

        def forward(self, x, mask=None):
            x = x + self.attn(self.attn_norm(x), mask)
            x = x + self.ff(self.ff_norm(x))
            return x

    class QuantumLLM(nn.Module):
        """140M Parameter Quantum Computing Language Model"""
        def __init__(self, config):
            super().__init__()
            self.config = config

            self.tok_emb = nn.Embedding(config["vocab_size"], config["d_model"])
            self.dropout = nn.Dropout(config["dropout"])

            self.blocks = nn.ModuleList([
                TransformerBlock(config) for _ in range(config["n_layers"])
            ])

            self.norm = RMSNorm(config["d_model"])
            self.lm_head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)

            # Weight tying
            self.lm_head.weight = self.tok_emb.weight

        def forward(self, idx, targets=None):
            B, T = idx.shape

            x = self.tok_emb(idx)
            x = self.dropout(x)

            mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)

            for block in self.blocks:
                x = block(x, mask)

            x = self.norm(x)
            logits = self.lm_head(x)

            loss = None
            if targets is not None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=self.config["pad_token_id"]
                )

            return logits, loss

        def generate(self, idx, max_new_tokens, temperature=0.2, top_k=30):
            """Generate text autoregressively"""
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config["max_seq_len"]:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)

            return idx

        @classmethod
        def load(cls, path, device='cpu'):
            """Load model from checkpoint (config is inside .pt file)"""
            import torch
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            model = cls(checkpoint['config'])
            model.load_state_dict(checkpoint['model_state_dict'])
            return model

    return QuantumLLM


# =============================================================================
# MODAL INFERENCE CLASS
# =============================================================================

@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes={MODEL_DIR: model_volume},
    scaledown_window=IDLE_TIMEOUT,
    timeout=120,
)
class QuantumInference:
    @modal.enter()
    def load_model(self):
        """Load model and tokenizer when container starts."""
        import torch
        from tokenizers import Tokenizer
        import os

        print("Loading model...")

        # Get model class
        QuantumLLM = get_model_classes()

        # Load model (config is inside the .pt file)
        model_path = os.path.join(MODEL_DIR, "final_model.pt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = QuantumLLM.load(model_path, self.device)
        self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        print(f"Model loaded on {self.device}")
        print(f"Config: {self.model.config}")

    @modal.method()
    def generate(self, context: str, question: str) -> str:
        """Generate answer given context and question."""
        import torch

        # Build prompt (same format as training)
        prompt = f"Context: {context} Question: {question} Answer:"

        # Tokenize
        encoding = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoding.ids], device=self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=150,
                temperature=0.2,
                top_k=30
            )

        # Decode
        generated_text = self.tokenizer.decode(output_ids[0].tolist())

        # Extract answer (use find, not rfind, to get FIRST answer)
        answer = self.extract_answer(generated_text)
        return answer

    def extract_answer(self, text: str) -> str:
        """Extract answer from generated text."""
        marker = "Answer:"
        idx = text.find(marker)
        if idx != -1:
            answer = text[idx + len(marker):].strip()
            # Stop at next section or double newline
            for stop in ["\n\nContext:", "\n\nQuestion:", "\n\n"]:
                stop_idx = answer.find(stop)
                if stop_idx != -1:
                    answer = answer[:stop_idx]
            return answer.strip()
        return text.strip()


# =============================================================================
# WEB ENDPOINTS
# =============================================================================

@app.function(image=image, timeout=120)
@modal.fastapi_endpoint(method="POST")
def query(request: dict):
    """
    HTTP endpoint for inference.

    Request:
    {
        "context": "Retrieved context from RAG...",
        "question": "What is a qubit?"
    }

    Response:
    {
        "answer": "A qubit is...",
        "model": "quantum-llm-140m",
        "response_time_ms": 1234
    }
    """
    import time

    context = request.get("context", "")
    question = request.get("question", "")

    if not question:
        return {"error": "Question is required"}, 400

    start_time = time.time()

    inference = QuantumInference()
    answer = inference.generate.remote(context, question)

    elapsed_ms = int((time.time() - start_time) * 1000)

    return {
        "answer": answer,
        "model": "quantum-llm-140m",
        "response_time_ms": elapsed_ms
    }


@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health():
    """Health check endpoint."""
    return {"status": "ok", "model": "quantum-llm-140m"}


# =============================================================================
# LOCAL TESTING
# =============================================================================

@app.local_entrypoint()
def main():
    """Test the model locally."""
    context = "Q: What is superposition? A: Superposition allows a qubit to exist in multiple states simultaneously until measured. Q: What are quantum states? A: Quantum states describe the complete information about a quantum system."
    question = "What is a qubit?"

    inference = QuantumInference()
    answer = inference.generate.remote(context, question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")