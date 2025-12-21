"""
Quantum Computing LLM - Custom Transformer Model
1.2M parameter decoder-only GPT-style architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """Multi-head attention with RoPE"""
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, cos, sin, mask=None):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class FeedForward(nn.Module):
    """SwiGLU feed-forward network"""
    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        # SwiGLU uses 2/3 of the hidden dim for gate and up projections
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm"""
    def __init__(self, dim: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, dropout)
        self.ff_norm = RMSNorm(dim)
        self.ff = FeedForward(dim, ff_dim, dropout)

    def forward(self, x, cos, sin, mask=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        x = x + self.ff(self.ff_norm(x))
        return x


class QuantumLLM(nn.Module):
    """
    Quantum Computing LLM
    
    Architecture (1.2M parameters):
    - Embedding: 64 dim, 16K vocab
    - 4 transformer blocks
    - 4 attention heads (16 dim per head)
    - 256 FFN hidden dim
    - RoPE positional encoding
    - RMSNorm + SwiGLU
    """
    def __init__(
        self,
        vocab_size: int = 16000,
        dim: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        max_seq_len: int = 512,
        ff_dim: int = 256,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        
        # Token embedding
        self.tok_emb = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(dim // n_heads, max_seq_len)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        
        # Output
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        # Weight tying
        self.output.weight = self.tok_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        self._num_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        
        # Token embeddings
        x = self.tok_emb(input_ids)
        
        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        
        # Get rotary embeddings
        cos, sin = self.rotary_emb(T)
        
        # Transformer blocks
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        
        # Output
        x = self.norm(x)
        logits = self.output(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.pad_token_id
            )
        
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int = 1,
    ):
        """Generate tokens autoregressively"""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            
            # Forward pass
            outputs = self(idx_cond)
            logits = outputs["logits"][:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop at EOS
            if next_token.item() == eos_token_id:
                break
        
        return input_ids

    def num_parameters(self):
        return self._num_params


def create_model(config=None):
    """Create model with default or custom config"""
    default_config = {
        "vocab_size": 16000,
        "dim": 64,
        "n_layers": 4,
        "n_heads": 4,
        "max_seq_len": 512,
        "ff_dim": 256,
        "dropout": 0.1,
        "pad_token_id": 0,
    }
    if config:
        default_config.update(config)
    
    model = QuantumLLM(**default_config)
    print(f"Created QuantumLLM with {model.num_parameters():,} parameters")
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_model()
    
    # Test forward pass
    x = torch.randint(0, 16000, (2, 128))
    out = model(x, labels=x)
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {out['logits'].shape}")
    print(f"Loss: {out['loss'].item():.4f}")
    
    # Test generation
    prompt = torch.randint(0, 16000, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")
