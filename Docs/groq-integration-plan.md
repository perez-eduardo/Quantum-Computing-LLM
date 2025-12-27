# Groq Integration Implementation Plan

**Date:** December 27, 2025  
**Status:** ✅ COMPLETE  
**Purpose:** Add Groq API as a fast LLM option alongside custom model

---

## Overview

### Goal
Add Llama 3.3 70B Versatile via Groq API as a fast (~1-2s) alternative to the custom 125.8M model (~50-80s).

### Model Selection
| Setting | Value |
|---------|-------|
| Provider | Groq (Free tier) |
| Model | llama-3.3-70b-versatile |
| Speed | ~725ms response time |
| Cost | $0.59/M input, $0.79/M output |
| Rate Limits | ~6,000 TPM, ~30 RPM |

---

## Architecture

### Design Pattern
Abstract Base Class with separate implementations per LLM.

### File Structure
```
backend/scripts/
├── retrieval.py          # unchanged
├── base_inference.py     # BaseLLM abstract class
├── inference.py          # QuantumInference(BaseLLM)
└── groq_inference.py     # GroqInference(BaseLLM)
```

Note: `pipeline.py` was deleted (unused).

### Class Diagram
```
              BaseLLM (ABC)
              ├── generate(context, question) -> str
              ├── extract_answer(text) -> str
              └── name property -> str
                    │
        ┌───────────┴───────────┐
        │                       │
QuantumInference          GroqInference
(custom 125.8M)           (Groq API)
name="custom"             name="groq"
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
  "use_groq": true
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `question` | string | Yes | - | The question to ask |
| `use_groq` | boolean | No | `false` | `true` = Groq API (~725ms), `false` = custom model (~50-80s) |

**Response (200 OK):**
```json
{
  "answer": "A qubit, short for quantum bit, is the basic unit of information in quantum computing. It's special because, unlike a classical bit that can only be 0 or 1, a qubit can be 0, 1, or both at the same time. This is possible due to a property called superposition, which gives quantum computers their potential power.",
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
  "response_time_ms": 725,
  "model_loaded_fresh": false,
  "suggested_question": "can you explain qubits in simple terms?",
  "llm_used": "groq"
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
| `model_loaded_fresh` | boolean | `true` if custom model was just loaded (cold start) |
| `suggested_question` | string or null | Suggested follow-up question |
| `llm_used` | string | Which LLM was used: `"groq"` or `"custom"` |

**Error Response (500):**
```json
{
  "detail": "Error message here"
}
```

**Example (JavaScript):**
```javascript
const response = await fetch('http://localhost:8000/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: 'What is a qubit?',
    use_groq: true
  })
});

const data = await response.json();
console.log(data.answer);
console.log(data.llm_used);  // "groq"
```

**Example (PowerShell):**
```powershell
Invoke-WebRequest -Uri http://localhost:8000/query -Method POST -ContentType "application/json" -Body '{"question": "What is a qubit?", "use_groq": true}'
```

---

### GET /health

Check API status and LLM availability.

**Response (200 OK):**
```json
{
  "status": "ok",
  "model_loaded": false,
  "idle_seconds": null,
  "groq_available": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always `"ok"` if API is running |
| `model_loaded` | boolean | Whether custom model is currently in memory |
| `idle_seconds` | float or null | Seconds since last custom model use (null if not loaded) |
| `groq_available` | boolean | Whether Groq API key is configured |

**Example (JavaScript):**
```javascript
const response = await fetch('http://localhost:8000/health');
const data = await response.json();

if (data.groq_available) {
  // Can use use_groq: true
}
```

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

### 2. Updated QuantumInference (`inference.py`) ✅

Changes:
- Inherits from `BaseLLM`
- `generate()` accepts `context` and `question` separately
- Builds flat prompt internally: `f"Context: {context} Question: {question} Answer:"`
- `name` property returns `"custom"`
- Keeps `extract_answer()` logic (parsing "Answer:" and stopping at markers)

---

### 3. GroqInference (`groq_inference.py`) ✅

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

### 4. Config Updates (`config.py`) ✅

```python
# Groq settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.2
GROQ_MAX_TOKENS = 300
```

GROQ_API_KEY is optional (only validated when Groq mode is used).

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
```

**Railway:**
Add `GROQ_API_KEY` to backend service environment variables.

---

## Test Results (December 27, 2025) ✅

### Docker Local Testing

```powershell
docker build -f backend/Dockerfile -t quantum-backend .
docker run -p 8000:8000 --env-file .env -e PORT=8000 quantum-backend
```

### Health Check
```json
{
  "status": "ok",
  "model_loaded": false,
  "idle_seconds": null,
  "groq_available": true
}
```

### Groq Response (20 queries tested)
All passed with ~725ms average response time.

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
- What are the applications of quantum computing?

---

## Implementation Checklist ✅

| Step | Task | Status |
|------|------|--------|
| 1 | Create abstract base class | ✅ Done |
| 2 | Update custom inference to inherit | ✅ Done |
| 3 | Create Groq inference class | ✅ Done |
| 4 | Add Groq config | ✅ Done |
| 5 | Update API request/response models | ✅ Done |
| 6 | Update `/query` endpoint logic | ✅ Done |
| 7 | Add groq dependency | ✅ Done |
| 8 | Add GROQ_API_KEY to `.env` | ✅ Done |
| 9 | Test locally with Docker | ✅ Done |
| 10 | Update Railway env vars | ⬜ Pending |
| 11 | Deploy | ⬜ Pending |

---

## Future Considerations

1. **Frontend toggle:** Add UI switch to select mode
2. **Default mode config:** Make default mode configurable via env var
3. **Rate limiting:** Handle Groq 429 errors gracefully
4. **Fallback:** If Groq fails, optionally fall back to custom model

---

*Document version: 2.1*
*Last updated: December 27, 2025*
