"""
Quantum Computing LLM - 50M Parameter Transformer
Two-phase training: books pretrain -> context Q&A fine-tune
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json

# ============================================================
# 50M PARAMETER CONFIGURATION
# ============================================================
# Target: ~50M parameters
# Formula: params ≈ 2 * d_model * n_layers * (4*d_model + d_model) + vocab*d_model
# With d_model=768, n_layers=12, vocab=16384: ~50M params

DEFAULT_CONFIG = {
    "vocab_size": 16384,
    "d_model": 768,
    "n_heads": 12,
    "n_layers": 12,
    "d_ff": 3072,  # 4 * d_model
    "max_seq_len": 1024,
    "dropout": 0.1,
    "pad_token_id": 0,
}


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
    """50M Parameter Quantum Computing Language Model"""
    def __init__(self, config=None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG

        self.tok_emb = nn.Embedding(self.config["vocab_size"], self.config["d_model"])
        self.dropout = nn.Dropout(self.config["dropout"])

        self.blocks = nn.ModuleList([
            TransformerBlock(self.config) for _ in range(self.config["n_layers"])
        ])

        self.norm = RMSNorm(self.config["d_model"])
        self.lm_head = nn.Linear(self.config["d_model"], self.config["vocab_size"], bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {self.n_params:,} ({self.n_params/1e6:.1f}M)")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Embedding
        x = self.tok_emb(idx)
        x = self.dropout(x)

        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.config["pad_token_id"])

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
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

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
        }, path)

    @classmethod
    def load(cls, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Test model creation
    model = QuantumLLM(DEFAULT_CONFIG)
    
    # Test forward pass
    x = torch.randint(0, 1000, (2, 128))
    logits, loss = model(x, x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
