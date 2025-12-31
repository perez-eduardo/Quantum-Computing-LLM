# Groq Integration Implementation Plan

**Date:** December 27, 2025
**Last Updated:** December 31, 2025
**Status:** ✅ COMPLETE
**Purpose:** Add Groq API as a fast LLM option alongside custom model

---

## Overview

### Goal
Add Llama 3.3 70B Versatile via Groq API as a fast (~2-3s) alternative to the custom 140M model (~35-60s on Modal).

### Model Selection
| Setting | Value |
|---------|-------|
| Provider | Groq (Free tier) |
| Model | llama-3.3-70b-versatile |
| Speed | ~2-3s response time |
| Cost | $0.59/M input, $0.79/M output |
| Rate Limits | ~6,000 TPM, ~30 RPM |

---

## Architecture

### Dual-Mode System

| Mode | LLM | Host | Speed |
|------|-----|------|-------|
| Production | Groq API (Llama 3.3 70B) | Groq Cloud | ~2-3s |
| Demo | Custom 140M | Modal (T4 GPU) | ~35-60s |

### Design Pattern
Abstract Base Class with separate implementations per LLM.

### File Structure
```
backend/scripts/
├── retrieval.py          # Retriever class (Voyage AI + Neon)
├── base_inference.py     # BaseLLM abstract class
├── groq_inference.py     # GroqInference(BaseLLM)
└── modal_inference.py    # ModalInference (calls Modal API)
```

### Class Diagram
```
              BaseLLM (ABC)
              ├── generate(context, question) -> str
              ├── extract_answer(text) -> str
              └── name property -> str
                    │
        ┌───────────┴───────────┐
        │                       │
GroqInference            ModalInference
(Groq API)               (Modal API)
name="groq"              name="custom"
```

---

## API Reference

### Base URL
```
Local:      http://localhost:8000
Production: https://quantum-computing-llm-backend.up.railway.app
```

---

### POST /query

Send a question and receive an answer.

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "question": "What is a qubit?",
  "model": "groq"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `question` | string | Yes | - | The question to ask |
| `model` | string | No | `"groq"` | `"groq"` = Groq API (~2-3s), `"custom"` = Modal API (~35-60s) |

**Response (200 OK):**
```json
{
  "answer": "A qubit, short for quantum bit, is the basic unit of information in quantum computing...",
  "sources": [
    {
      "question": "is a qubit just a tiny particle?",
      "source": "claude",
      "similarity": 0.7602
    },
    {
      "question": "What is a qubit?",
      "source": "claude",
      "similarity": 0.7377
    },
    {
      "question": "What is a quantum bit fundamentally?",
      "source": "claude",
      "similarity": 0.7346
    }
  ],
  "response_time_ms": 2500,
  "suggested_question": "can you explain qubits in simple terms?",
  "model_used": "groq"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | The generated answer |
| `sources` | array | Top 3 retrieved Q&A pairs from RAG |
| `sources[].question` | string | The retrieved question |
| `sources[].source` | string | Data source: `"claude"`, `"stackexchange"`, or `"cot"` |
| `sources[].similarity` | float | Cosine similarity score (0-1) |
| `response_time_ms` | integer | Total response time in milliseconds |
| `suggested_question` | string or null | Suggested follow-up question |
| `model_used` | string | Which LLM was used: `"groq"` or `"custom"` |

**Error Response (500):**
```json
{
  "detail": "Error message here"
}
```

---

### GET /health

Check API status and LLM availability.

**Response (200 OK):**
```json
{
  "status": "ok",
  "groq_available": true,
  "modal_available": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always `"ok"` if API is running |
| `groq_available` | boolean | Whether Groq API key is configured |
| `modal_available` | boolean | Whether Modal URL is configured |

---

## Implementation Details

### 1. BaseLLM Abstract Class (`base_inference.py`) ✅

```python
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Return LLM identifier."""
        pass
    
    @abstractmethod
    def generate(self, context: str, question: str) -> str:
        """Generate answer given RAG context and question."""
        pass
    
    @abstractmethod
    def extract_answer(self, generated_text: str) -> str:
        """Extract clean answer from generated text."""
        pass
```

---

### 2. GroqInference (`groq_inference.py`) ✅

```python
class GroqInference(BaseLLM):
    @property
    def name(self) -> str:
        return "groq"
    
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_MODEL_NAME
    
    def generate(self, context: str, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a quantum computing assistant for beginners..."
            },
            {
                "role": "user", 
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=300
        )
        return response.choices[0].message.content
    
    def extract_answer(self, generated_text: str) -> str:
        return generated_text.strip()
```

---

### 3. ModalInference (`modal_inference.py`) ✅

```python
class ModalInference(BaseLLM):
    @property
    def name(self) -> str:
        return "custom"
    
    def __init__(self):
        self.modal_url = MODAL_URL
    
    def generate(self, context: str, question: str) -> str:
        response = requests.post(
            self.modal_url,
            json={"context": context, "question": question},
            timeout=120
        )
        return response.json()["answer"]
    
    def extract_answer(self, generated_text: str) -> str:
        return generated_text.strip()
```

---

### 4. Config (`config.py`) ✅

```python
# Groq settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.2
GROQ_MAX_TOKENS = 300

# Modal settings
MODAL_URL = os.getenv("MODAL_URL", "https://perez-eduardo--quantum-llm-query.modal.run")
```

---

### 5. Dependencies (`backend/requirements.txt`) ✅

```
groq>=0.11.0
```

Note: Version 0.4.2 had compatibility issues with httpx proxies.

---

### 6. Environment Variables ✅

**Local (.env):**
```
VOYAGE_API_KEY=your_voyage_key
DATABASE_URL=your_neon_url
GROQ_API_KEY=your_groq_key
MODAL_URL=https://perez-eduardo--quantum-llm-query.modal.run
```

**Railway:**
- `GROQ_API_KEY` ✅ Configured
- `MODAL_URL` ✅ Configured

---

## Test Results ✅

### Groq Response (20 queries tested)
All passed with ~2-3s average response time.

### Test Queries Run
All 20 quantum computing questions passed:
- What is a qubit?
- What is superposition?
- What is quantum entanglement?
- What is a quantum gate?
- What is the Hadamard gate?
- What is quantum decoherence?
- What is a quantum circuit?
- What is Shor's algorithm?
- What is Grover's algorithm?
- What is quantum error correction?
- What is the Bloch sphere?
- What is quantum teleportation?
- What is a CNOT gate?
- What is quantum supremacy?
- What is a quantum register?
- What is quantum interference?
- What is quantum parallelism?
- What is a universal quantum gate set?
- What is measurement in quantum computing?
- How do quantum computers differ from classical computers?

---

## Implementation Checklist ✅

| Step | Task | Status |
|------|------|--------|
| 1 | Create abstract base class | ✅ Done |
| 2 | Create Groq inference class | ✅ Done |
| 3 | Create Modal inference class | ✅ Done |
| 4 | Add Groq config | ✅ Done |
| 5 | Add Modal config | ✅ Done |
| 6 | Update API request/response models | ✅ Done |
| 7 | Update `/query` endpoint logic | ✅ Done |
| 8 | Add groq dependency | ✅ Done |
| 9 | Add GROQ_API_KEY to Railway | ✅ Done |
| 10 | Add MODAL_URL to Railway | ✅ Done |
| 11 | Deploy | ✅ Done |
| 12 | Add frontend model selector | ✅ Done |

---

## Frontend Integration ✅

The frontend now includes:
- Model selector dropdown (Groq vs Custom)
- Sends `model: "groq"` or `model: "custom"` to backend
- Different timeout handling (30s for Groq, 180s for Custom)
- Toast notification when Custom is selected
- Pipeline animation for Custom model loading

---

*Document version: 3.0*
