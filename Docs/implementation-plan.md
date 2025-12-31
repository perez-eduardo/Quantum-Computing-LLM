# Quantum Computing Assistant - Implementation Plan

**Last Updated:** December 30, 2025

**Related Documents:**
- Design Document: `quantum-computing-assistant-design.md`
- Infrastructure Planning: `initial-exploratory-brainstorming.md`
- Model Investigation Report: `model_investigation_report.md`
- Modal Deployment Plan: `modal-deployment-plan.md`

---

## Current Status

**Phase 1:** Training Pipeline - ✅ COMPLETE (v5 trained, 100% pass rate)
**Phase 2:** RAG System - ✅ COMPLETE (100% retrieval accuracy)
**Phase 3:** Backend - ✅ COMPLETE (FastAPI, Groq-only)
**Phase 4:** Frontend - ✅ COMPLETE (Flask + Jinja)
**Phase 5:** Deployment - ✅ COMPLETE (Railway, live)
**Phase 6:** Custom Model Hosting - ✅ COMPLETE (Modal, live)

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
| Frontend | Railway | User interface |
| Backend + RAG | Railway | API, retrieval, Groq inference |
| Custom Model | Modal | Demo inference (140M transformer) |

### Request Flow

```
User Question → Railway Backend → Voyage AI embed → Neon vector search → Build prompt
                                                                        ↓
                                                    mode=groq   → Groq API (~1-2s)
                                                    mode=custom → Modal API (~5-10s GPU)
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
| Inference | ~5-10 seconds |
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
│   └── ...
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
├── backend/                      # Railway (Groq-only)
│   ├── Dockerfile
│   ├── Procfile
│   ├── requirements.txt
│   ├── scripts/
│   │   ├── retrieval.py
│   │   └── groq_inference.py
│   └── app/
│       ├── __init__.py
│       ├── config.py
│       └── main.py
│
├── frontend/                     # Railway
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

---

## Pending Tasks

### Frontend Integration (Local Development)

| Task | Priority |
|------|----------|
| Add model selector dropdown (Groq vs Custom) | High |
| Update JS to send model choice to backend | High |
| Increase timeout for custom model requests | High |
| Update About modal with custom model info | Medium |

### Backend Integration

| Task | Priority |
|------|----------|
| Add modal_inference.py | Medium |
| Update main.py for dual mode | Medium |
| Add MODAL_URL env var | Medium |

---

## Key Findings

1. **Model config inside .pt file.** QuantumLLM.load() extracts config automatically.

2. **Tokenizers version must match.** Training used 0.22.1, Modal must use same.

3. **Modal cold start ~3-4s.** Acceptable for demo purposes.

4. **T4 GPU inference ~5-10s.** Much faster than CPU (~50-80s).

5. **Modal $30/mo free tier.** Sufficient for demo usage.

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

*Document version: 26.0*
