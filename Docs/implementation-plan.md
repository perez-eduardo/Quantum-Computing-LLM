# Quantum Computing Assistant - Implementation Plan

**Last Updated:** December 31, 2025

**Related Documents:**
- Design Document: `quantum-computing-assistant-design.md`
- Infrastructure Planning: `initial-exploratory-brainstorming.md`
- Model Investigation Report: `model_investigation_report.md`
- Modal Deployment Plan: `modal-deployment-plan.md`
- Groq Integration Plan: `groq-integration-plan.md`

---

## Current Status

**Phase 1:** Training Pipeline - ✅ COMPLETE (v5 trained, 100% pass rate)
**Phase 2:** RAG System - ✅ COMPLETE (100% retrieval accuracy)
**Phase 3:** Backend - ✅ COMPLETE (FastAPI, dual-mode Groq + Modal)
**Phase 4:** Frontend - ✅ COMPLETE (Flask + Jinja, model selector, health checks)
**Phase 5:** Deployment - ✅ COMPLETE (Railway, live)
**Phase 6:** Custom Model Hosting - ✅ COMPLETE (Modal T4 GPU, live)

**Live URLs:**
- Frontend: https://quantum-computing-llm.up.railway.app
- Backend: https://quantum-computing-llm-backend.up.railway.app
- API Docs: https://quantum-computing-llm-backend.up.railway.app/docs
- Modal Query: https://perez-eduardo--quantum-llm-query.modal.run
- Modal Health: https://perez-eduardo--quantum-llm-health.modal.run

---

## Architecture Overview

### Dual-Mode System

| Component | Host | Purpose |
|-----------|------|---------|
| Frontend | Railway | User interface, model selector |
| Backend + RAG | Railway | API, retrieval, Groq inference |
| Custom Model | Modal (T4 GPU) | Demo inference (140M transformer) |

### Request Flow

```
User Question → Railway Frontend → Railway Backend → Voyage AI embed → Neon vector search
                                                                              ↓
                                                         model="groq"   → Groq API (~2-3s)
                                                         model="custom" → Modal API (~35-60s)
```

---

## Custom Model (Modal)

### Model Specs

| Parameter | Value |
|-----------|-------|
| Architecture | QuantumLLM (custom transformer) |
| Parameters | 140,004,480 (140M) |
| File size | 510MB |
| vocab_size | 16384 |
| d_model | 768 |
| n_heads | 12 |
| n_layers | 12 |
| d_ff | 3072 |
| max_seq_len | 1024 |
| Tokenizer | BPE (tokenizers==0.22.1) |
| Best temp | 0.2 |
| Best top_k | 30 |

### Modal Deployment

| Setting | Value |
|---------|-------|
| GPU | T4 (16GB VRAM) |
| Free tier | $30/month |
| Cold start | 3-4 seconds |
| Inference | ~35-60 seconds total |
| Idle timeout | 5 minutes |
| Volume | quantum-model-volume |
| Query endpoint | https://perez-eduardo--quantum-llm-query.modal.run |
| Health endpoint | https://perez-eduardo--quantum-llm-health.modal.run |

### Modal Files

```
modal/
├── inference.py      # Modal app with QuantumLLM architecture
├── test_local.py     # Local testing script
└── venv/             # Local venv for Modal CLI
```

### Modal API

**POST /query**
```json
Request:  {"context": "...", "question": "What is a qubit?"}
Response: {"answer": "...", "model": "quantum-llm-140m", "response_time_ms": 5000}
```

**GET /health**
```json
Response: {"status": "ok", "model": "quantum-llm-140m"}
```

---

## Project Structure

```
Quantum-Computing-LLM/
├── Docs/
│   ├── implementation-plan.md
│   ├── modal-deployment-plan.md
│   ├── groq-integration-plan.md
│   ├── initial-exploratory-brainstorming.md
│   ├── model_investigation_report.md
│   └── quantum-computing-assistant-design.md
│
├── training/
│   ├── model/
│   │   ├── final_model.pt        # 140M params (510MB)
│   │   └── config.json
│   ├── tokenizer/
│   │   └── tokenizer.json        # 16K vocab BPE
│   └── scripts/
│       ├── model.py              # QuantumLLM architecture
│       ├── dataset.py
│       ├── train.py
│       └── evaluate.py
│
├── backend/                      # Railway
│   ├── Dockerfile
│   ├── Procfile
│   ├── requirements.txt
│   ├── scripts/
│   │   ├── retrieval.py
│   │   ├── base_inference.py
│   │   ├── groq_inference.py
│   │   └── modal_inference.py
│   └── app/
│       ├── __init__.py
│       ├── config.py
│       └── main.py
│
├── frontend/                     # Railway
│   ├── Dockerfile                # gunicorn --timeout 300
│   ├── Procfile
│   ├── app.py
│   ├── requirements.txt
│   ├── static/
│   │   ├── css/style.css
│   │   ├── js/main.js
│   │   └── images/
│   └── templates/
│       └── index.html
│
├── modal/                        # Modal deployment
│   ├── inference.py
│   ├── test_local.py
│   └── venv/
│
└── data/
    ├── raw/
    └── processed/
```

---

## Frontend Features

### Model Selector
- Dropdown in header (top-left)
- Options: Groq (default), Custom
- Toast notification when Custom selected
- Displays response time expectations

### Health Check System
- Initial startup check (waits for backend)
- Pre-query health check on every request
- Wake-up indicator for cold starts
- Specific error message after 10 failed attempts

### Loading Indicators
- Groq: "Thinking..."
- Custom: Pipeline animation with stages:
  - "Embedding your question with Voyage AI"
  - "Searching knowledge base (28,071 Q&A pairs)"
  - "Retrieved relevant context from Neon database"
  - "Ranking results by relevance"
  - "Building prompt with context"
  - "Loading 140M parameter model"
  - "Generating response"

### Error Handling
- Retry button on failed queries
- Timeout handling (30s Groq, 180s Custom)
- Backend unavailable detection

### Modals
- Models modal: Explains Groq vs Custom
- About modal: Project details, tech stack, contact links

---

## Completed Tasks

### Phase 6: Modal Deployment ✅
- [x] Create Modal account
- [x] Write inference.py with full model architecture
- [x] Create Modal volume (quantum-model-volume)
- [x] Upload final_model.pt (510MB) to volume
- [x] Upload tokenizer.json to volume
- [x] Fix tokenizers version (0.22.1)
- [x] Deploy to Modal
- [x] Test health endpoint
- [x] Test query endpoint

### Frontend Integration ✅
- [x] Add model selector dropdown (Groq vs Custom)
- [x] Update JS to send model choice to backend
- [x] Increase timeout for custom model requests (300s)
- [x] Update About modal with custom model info
- [x] Add health check before every query
- [x] Add retry button for failed queries
- [x] Add cold start detection and wake-up indicator

### Backend Integration ✅
- [x] Add modal_inference.py
- [x] Update main.py for dual mode
- [x] Add MODAL_URL env var
- [x] Deploy to Railway with Groq API key

---

## Key Findings

1. **Model config inside .pt file.** QuantumLLM.load() extracts config automatically.

2. **Tokenizers version must match.** Training used 0.22.1, Modal must use same.

3. **Modal cold start ~3-4s.** Acceptable for demo purposes.

4. **T4 GPU inference ~35-60s total.** Much faster than Railway CPU (~50-80s).

5. **Modal $30/mo free tier.** Sufficient for demo usage.

6. **Health check before every query.** Prevents 502 errors from sleeping servers.

7. **Dockerfile timeout matters.** Must match Procfile (300s for custom model).

---

## Cost Summary

| Service | Monthly Cost |
|---------|--------------|
| Railway (backend + frontend) | ~$5 |
| Neon PostgreSQL | Free tier |
| Voyage AI | Free tier |
| Groq API | Free tier |
| Modal (custom model) | Free tier ($30 credit) |
| **Total** | **~$5/month** |

---

*Document version: 27.0*
