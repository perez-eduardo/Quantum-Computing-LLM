# Modal Deployment Plan - Quantum Computing LLM

**Status:** ✅ COMPLETE
**Last Updated:** December 30, 2025

---

## Overview

The custom 140M parameter QuantumLLM is deployed on Modal for serverless GPU inference. This provides fast inference (~5-10s) compared to CPU (~50-80s) while staying within budget using Modal's $30/month free tier.

---

## Live Endpoints

| Endpoint | URL |
|----------|-----|
| Query (POST) | https://perez-eduardo--quantum-llm-query.modal.run |
| Health (GET) | https://perez-eduardo--quantum-llm-health.modal.run |
| Dashboard | https://modal.com/apps/perez-eduardo/main/deployed/quantum-llm |

---

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Architecture | QuantumLLM (custom transformer) |
| Total Parameters | 140,004,480 (140M) |
| File Size | 510MB |
| vocab_size | 16384 |
| d_model | 768 |
| n_heads | 12 |
| n_layers | 12 |
| d_ff | 3072 |
| max_seq_len | 1024 |
| dropout | 0.1 |
| pad_token_id | 0 |

**Important:** Model config is stored INSIDE the .pt file. Use `QuantumLLM.load(path, device)` to load both weights and config.

---

## Modal Configuration

| Setting | Value |
|---------|-------|
| GPU | T4 (16GB VRAM) |
| Python | 3.10 |
| Image | debian_slim |
| Idle timeout | 300s (5 min) |
| Request timeout | 120s |
| Volume | quantum-model-volume |

### Dependencies

```python
.pip_install(
    "torch==2.1.0",
    "tokenizers==0.22.1",  # Must match training version!
    "fastapi",
)
```

---

## Volume Contents

```
quantum-model-volume/
├── final_model.pt     # 486 MB
└── tokenizer.json     # 1.1 MB
```

---

## API Reference

### POST /query

Generate an answer using the custom model.

**Request:**
```json
{
  "context": "Q: What is superposition? A: Superposition allows...",
  "question": "What is a qubit?"
}
```

**Response:**
```json
{
  "answer": "A qubit, short for quantum bit, is the basic unit...",
  "model": "quantum-llm-140m",
  "response_time_ms": 5234
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model": "quantum-llm-140m"
}
```

---

## Files

### modal/inference.py

Contains:
- Full QuantumLLM architecture (RMSNorm, RoPE, SwiGLU, Attention, TransformerBlock)
- QuantumInference class with @modal.enter() for model loading
- Web endpoints: /query (POST), /health (GET)
- Generation params: temperature=0.2, top_k=30
- Answer extraction logic

### modal/test_local.py

Local testing script to verify model works before deploying.

---

## Deployment Commands

```bash
# Setup (one-time)
cd modal
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install modal
modal setup

# Create volume and upload files (one-time)
modal volume create quantum-model-volume
modal volume put quantum-model-volume ../training/model/final_model.pt final_model.pt
modal volume put quantum-model-volume ../training/tokenizer/tokenizer.json tokenizer.json

# Deploy
modal deploy inference.py

# View logs
modal app logs quantum-llm

# Stop app
modal app stop quantum-llm

# List volume contents
modal volume ls quantum-model-volume
```

---

## Testing

### PowerShell

```powershell
# Health check
Invoke-WebRequest -Uri "https://perez-eduardo--quantum-llm-health.modal.run"

# Query
Invoke-WebRequest -Uri "https://perez-eduardo--quantum-llm-query.modal.run" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"context": "Q: What is superposition? A: Superposition allows a qubit to exist in multiple states simultaneously.", "question": "What is a qubit?"}'
```

### Bash/curl

```bash
# Health check
curl https://perez-eduardo--quantum-llm-health.modal.run

# Query
curl -X POST https://perez-eduardo--quantum-llm-query.modal.run \
  -H "Content-Type: application/json" \
  -d '{"context": "...", "question": "What is a qubit?"}'
```

---

## Performance

| Metric | Value |
|--------|-------|
| Cold start | 3-4 seconds |
| Warm inference | 5-10 seconds |
| VRAM usage | ~560MB |

---

## Cost Estimate

| Usage | Cost |
|-------|------|
| T4 GPU | $0.000164/sec |
| 100 queries/day × 10s | ~$0.16/day |
| Monthly (100 queries/day) | ~$5/month |
| Free tier | $30/month |

Well within free tier for demo usage.

---

## Troubleshooting

### Tokenizer error: "data did not match any variant"

Tokenizers version mismatch. Ensure Modal uses `tokenizers==0.22.1` (same as training).

### Timeout errors

- Check logs: `modal app logs quantum-llm`
- Cold start can take 3-4s, inference 5-10s
- Total should be under 120s timeout

### Cached image not updating

```bash
modal app stop quantum-llm
modal deploy inference.py
```

Watch for "Building image" in output.

---

## Integration with Railway Backend

To enable dual-mode (Groq + Custom) in the Railway backend:

1. Add `MODAL_URL` env var in Railway
2. Create `modal_inference.py` in backend/scripts/
3. Update `main.py` to accept `model` parameter
4. Frontend sends `model: "groq"` or `model: "custom"`

---

*Document version: 3.0*
