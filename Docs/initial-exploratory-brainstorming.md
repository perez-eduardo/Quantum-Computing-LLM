# Initial Exploratory Brainstorming: Next LLM Project Stack

**Date:** December 20, 2025 (Updated December 22, 2025)
**Project:** Quantum Computing LLM  
**Purpose:** Portfolio demonstration piece  
**Expected Traffic:** Minimal (recruiters, students)

---

## Problem Statement

The Philippine Legal Assistant project revealed infrastructure limitations:

1. **Groq free tier:** Shared limits across all users caused rate limiting
2. **Render free tier:** 512MB RAM cannot handle local Sentence Transformers model

Need a stack that is:
- Cost-optimized (minimal monthly spend)
- Reliable (no shared rate limits)
- Fast (minimal cold starts)

---

## Project Architecture

This project combines two components:

1. **Custom Transformer** (trained from scratch) for portfolio demonstration
2. **RAG System** to supplement the small model's limited knowledge

The custom transformer shows recruiters you understand ML internals. RAG compensates for the small model's limitations by retrieving relevant documents.

---

## Final Stack

| Component | Provider | Cost | Notes |
|-----------|----------|------|-------|
| **Frontend + Backend** | Railway (Hobby) | $5/month | Monorepo, always on |
| **LLM (Custom)** | Your trained transformer | $0 | ~1.2M params, one-time training cost |
| **Embeddings** | Voyage AI | $0 | 200M free tokens |
| **Database** | Neon (free) | $0 | PostgreSQL + pgvector |
| **Training Compute** | Oregon State HPC | $0 | H100 GPUs available |

**Total: ~$5/month** (after one-time training)

---

## Custom Transformer Training

### Purpose

- Demonstrate understanding of transformer architecture
- Show ability to build a complete training pipeline
- Create a "fully yours" component for the portfolio
- Combined with RAG for actual usefulness

### Architecture

| Parameter | Value |
|-----------|-------|
| Type | Decoder-only (GPT-style) |
| Size | ~1.2M parameters |
| Layers | 4 |
| Attention heads | 4 |
| Embedding dimension | 64 |
| Vocabulary | 16,000 (custom BPE) |
| Context length | 512 tokens |

### Training Data

> **⚠️ MAJOR REVISION (December 22, 2025)**
> 
> ChatGPT synthetic Q&A data has been **abandoned**. After extensive cleaning:
> - 83% was boilerplate phrases
> - 59% was templated repetitive questions (only numbers changed)
> - After all cleaning: 85,643 → 4,808 rows (94% garbage)
> 
> **Decision:** Replace ChatGPT data with Claude-generated Q&A.

#### Current Clean Data (Before Claude Q&A)

| Source | Count | Status |
|--------|-------|--------|
| ~~ChatGPT Synthetic Q&A~~ | ~~85,643 pairs~~ | ❌ ABANDONED (94% garbage) |
| Stack Exchange Q&A | 8,858 pairs | ✅ Cleaned |
| 5 Books (3x upsampled) | 11,493 chunks | ✅ Ready |
| **Current Total** | **20,351** | Needs Q&A supplement |

#### Claude Q&A Generation Plan

| Parameter | Value |
|-----------|-------|
| Target | 3,000 beginner Q&A pairs |
| Source material | 5 quantum computing textbooks |
| Method | Semi-automated via Claude.ai chat |
| Sessions | ~20 sessions (150 Q&A each) |
| Difficulty | Beginner level |
| Cost | $0 (Pro subscription) |

**Book Sources for Q&A Generation:**

| Book | Text File | Priority |
|------|-----------|----------|
| Quantum Computing for Everyone (Bernhardt) | bernhardt.txt | High (beginner) |
| Quantum Computing Explained for Beginners | beginners.txt | High (beginner) |
| Introduction to Classical and Quantum Computing (Wong) | wong.txt | Medium |
| Quantum Computing: An Applied Approach (Hidary) | hidary.txt | Low (foundational only) |
| Quantum Computation and Quantum Information (Nielsen & Chuang) | nielsen_chuang.txt | Low (foundational only) |

**Process:**
1. Upload book PDF to Claude.ai chat
2. Use prompt template to generate 150 Q&A pairs per session
3. Copy output to text file
4. Run formatting script to combine into CSV
5. Repeat for all books/chapters

#### Final Dataset (After Claude Q&A)

| Source | Count | Percent |
|--------|-------|---------|
| Claude Q&A (planned) | 3,000 | 12.8% |
| Stack Exchange Q&A | 8,858 | 37.9% |
| Books (3x upsampled) | 11,493 | 49.2% |
| **Total** | **23,351** | 100% |

### Compute: Oregon State University HPC

**Access confirmed.** Connection details:

| Field | Value |
|-------|-------|
| Host | `submit-b.hpc.engr.oregonstate.edu` |
| Username | `pereze4` |
| Scheduler | SLURM |
| Home Storage | 25 GB |
| HPC Share | 1.5 TB quota |

**Connect from Windows PowerShell:**
```powershell
ssh pereze4@submit-b.hpc.engr.oregonstate.edu
```

### Available GPU Partitions

| Partition | GPUs per Node | GPU Type | RAM | Time Limit |
|-----------|---------------|----------|-----|------------|
| **dgxh** | 16 | H100-40GB | 2 TB | 2 days |
| **dgx2** | 14-16 | V100 | 1.5 TB | 7 days |
| **gpu** | 8 | General | 760 GB | 7 days |
| **ampere** | 2 | A-series | 252 GB | 2 days |

**Recommended:** Use `dgxh` partition (H100s) for fastest training.

### Available Software Modules

| Software | Versions Available |
|----------|-------------------|
| CUDA | 9.2 through 12.9 |
| cuDNN | 8.8, 8.9 |
| Python | 3.8 through 3.13 |
| Anaconda | 24.3 |
| NCCL | 2.12.10 (multi-GPU) |

**Note:** PyTorch not pre-installed. Requires environment setup.

### Environment Setup (When Ready)

```bash
cd ~/hpc-share
mkdir quantum-llm
cd quantum-llm

python -m venv venv
source venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Training Checklist

- [x] Finalize model size (1.2M parameters)
- [x] Gather training data (quantum computing corpus)
- [x] Clean Stack Exchange data (filtered >1024 tokens)
- [x] Train custom tokenizer
- [x] Upsample books 3x
- [x] Abandon ChatGPT data (94% garbage)
- [ ] **Generate Claude Q&A (~3,000 pairs)**
- [ ] Combine final dataset
- [ ] Retrain with 10 epochs
- [ ] Evaluate model
- [ ] Integrate with RAG system

---

## Component Details

### Embeddings: Voyage AI

**Why Voyage AI over alternatives:**

| Provider | Free Tokens | Free RPM | Paid Rate |
|----------|-------------|----------|-----------|
| Jina AI | 10M (one-time) | 10-30 RPM | $0.02/1M |
| Voyage AI | 200M | 3 RPM (free), 2000 RPM (with card) | $0.02/1M |
| Google AI | Gutted Dec 2025 | Unreliable | Unknown |

**Critical setup step:**
- Must add payment method to unlock 2000 RPM
- Without card: only 3 requests/minute (unusable)
- With card: 2000 RPM, still uses free tokens

**Token usage estimate:**
- Initial document embedding: ~1.5M tokens
- 10,000 queries/month × 50 tokens = 0.5M tokens/month
- First year total: ~7.5M tokens (3.75% of free allowance)
- Runway: 20+ years on free tier

---

### Database: Neon Free Tier

**Limits:**
| Resource | Limit |
|----------|-------|
| Storage | 0.5 GB per project |
| Compute | 100 CU-hours/month |
| Egress | 5 GB/month |
| Projects | Up to 20 |
| Branches | 1 (no extra branches on free) |

**Cold start behavior:**
- Suspends after 5 minutes of inactivity
- Cold start latency: ~500ms-1 second
- Auto-wakes on first connection (no manual intervention)

**Why not Supabase:**
- Supabase pauses after 7 days of inactivity
- Requires manual restart (recruiter sees dead app)
- Neon auto-wakes, only adds ~500ms delay

**Configuration tip:**
```
DATABASE_URL=postgres://...?connect_timeout=10
```

---

### Frontend + Backend: Railway Hobby (Monorepo)

**Cost:** $5/month (includes $5 usage credit)

**Architecture:**
```
/your-project
  /frontend    (Single HTML page)
  /backend     (FastAPI)
```

FastAPI serves the frontend as static files. Single service, single URL.

**Why monorepo on Railway:**
- Single platform (one dashboard, one deploy, one bill)
- No CORS configuration needed (same origin)
- $5 covers both frontend and backend
- One URL to share with recruiters

**Why Railway over alternatives:**

| Provider | Plan | Cost | Cold Start |
|----------|------|------|------------|
| Render free | Free | $0 | 30-50 seconds |
| Render paid | Starter | $7/month | None |
| Railway | Hobby | $5/month | None |
| Fly.io | Pay-as-go | ~$3-5/month | Configurable |

**Benefits:**
- Always on (no cold start)
- Usage-based within $5 credit
- Sufficient for light portfolio traffic

---

## Rejected Alternatives

### Google AI Studio (LLM + Embeddings)
- **Status:** Gutted in December 2025
- Gemini 2.5 Pro removed from free tier
- Flash daily requests cut from ~250 to ~20/day
- Unreliable for production use

### Local Embeddings (Sentence Transformers)
- Requires 1GB+ RAM
- Render free tier only has 512MB
- Solution: Use API-based embeddings instead

### Supabase (Database)
- Pauses after 7 days of inactivity
- Requires manual restart
- Bad for portfolio with sporadic traffic

### Jina AI (Embeddings)
- Only 10M free tokens (vs Voyage 200M)
- Same pricing after free tier exhausted
- No advantage over Voyage

### Google Colab (Training)
- Free tier: T4 GPU, 12-hour sessions
- Less powerful than HPC H100s
- Fallback option if HPC unavailable

### Groq (Fallback LLM)
- Originally planned as fallback
- Removed to focus on custom model as portfolio centerpiece
- Custom model handles all inference

### ChatGPT API for Synthetic Q&A
- **Status:** ABANDONED (December 22, 2025)
- Generated 85,643 Q&A pairs
- 83% contained repetitive boilerplate phrases
- 59% were templated (only numbers changed)
- After cleaning: only 4,808 usable (6%)
- **Replaced with:** Claude-generated Q&A via chat

---

## Capacity Planning

### Query Volume Estimates

| Scenario | Queries/Month | Within Limits? |
|----------|---------------|----------------|
| Light (realistic) | 500-2,000 | Yes |
| Moderate | 5,000 | Yes |
| Heavy (100 users × 100 queries) | 10,000 | Yes |

### Bottleneck Analysis

**Primary bottleneck:** Custom model inference speed on Railway

With ~1.2M param model on CPU:
- Inference time: ~100-500ms per query (estimated)
- Sufficient for portfolio traffic

If inference too slow, options:
1. Optimize model (quantization)
2. Add response caching

---

## User Experience

**Single URL:** `your-app.up.railway.app` (frontend and API on same origin)

| Event | Latency |
|-------|---------|
| First request after DB idle (5+ min) | +500ms-1s |
| Subsequent requests | Instant |
| Frontend + Backend | Always instant (Railway always on) |
| Model inference | ~100-500ms |
| Embedding | Normal API latency |

---

## Setup Checklist

### Voyage AI
- [ ] Create account at voyageai.com
- [ ] Add payment method (unlocks 2000 RPM)
- [ ] Generate API key
- [ ] Use `voyage-3.5-lite` for cost efficiency

### Neon
- [ ] Create account at neon.tech
- [ ] Create project
- [ ] Enable pgvector extension: `CREATE EXTENSION vector;`
- [ ] Set connection timeout: `?connect_timeout=10`
- [ ] Note connection string

### Railway
- [ ] Create account at railway.app
- [ ] Subscribe to Hobby plan ($5/month)
- [ ] Set up monorepo structure:
  - `/frontend` - Single HTML page
  - `/backend` - FastAPI app
- [ ] Configure FastAPI to serve frontend static files
- [ ] Deploy single service
- [ ] Set environment variables for all API keys
- [ ] Note: URL will be `your-app.up.railway.app`

---

## Cost Summary

| Component | Monthly Cost |
|-----------|--------------|
| Railway (Frontend + Backend) | $5 |
| Voyage AI (Embeddings) | $0 |
| Neon (Database) | $0 |
| HPC (Training) | $0 (one-time) |
| **Total** | **$5/month** |

---

## Cost Protection (Attack/Abuse Safeguards)

Protect against intentional traffic spikes and abuse that could surge costs.

### Attack Surface

| Component | Attack Vector | Unprotected Risk |
|-----------|---------------|------------------|
| Railway | HTTP request flood | Unlimited CPU/bandwidth charges |
| Voyage AI | Embedding spam | 200M tokens drained |
| Neon | Connection flood | Compute hours exhausted |

### Layer 1: Provider Spending Caps (CRITICAL)

| Provider | Hard Limit? | Action |
|----------|-------------|--------|
| **Railway** | Yes | Settings → Usage → Hard limit: **$10** |
| **Voyage AI** | No | Free tier ceiling (200M tokens) |
| **Neon** | Yes | Auto-stops at 100 compute hours |

**Set these immediately after account creation.**

### Layer 2: Application Rate Limiting

Add to FastAPI backend:

```python
# requirements.txt: add slowapi
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/query")
@limiter.limit("10/minute")  # 10 queries/min per IP
async def query_llm(request: Request, ...):
    ...

@app.post("/embed")
@limiter.limit("20/minute")  # 20 embeddings/min per IP
async def create_embedding(request: Request, ...):
    ...
```

### Layer 3: Daily Budget Guards

```python
from datetime import date
from fastapi import HTTPException

DAILY_QUERY_LIMIT = 500

def get_daily_count(name: str) -> int:
    filename = f"/tmp/{name}_{date.today()}.count"
    try:
        with open(filename) as f:
            return int(f.read())
    except FileNotFoundError:
        return 0

def increment_count(name: str):
    count = get_daily_count(name) + 1
    with open(f"/tmp/{name}_{date.today()}.count", "w") as f:
        f.write(str(count))

@app.post("/query")
async def query_llm(...):
    if get_daily_count("queries") >= DAILY_QUERY_LIMIT:
        raise HTTPException(503, "Daily limit reached. Try tomorrow.")
    # ... process query ...
    increment_count("queries")
```

### Layer 4: Cloudflare (Free Tier)

| Protection | Benefit |
|------------|---------|
| DDoS mitigation | Absorbs volumetric attacks |
| Bot detection | Blocks automated abuse |
| Rate limiting | 10,000 free requests/month |
| Caching | Reduces origin server hits |

**Setup:**
1. Add custom domain to Cloudflare (free)
2. Point DNS to Railway
3. Enable "Under Attack Mode" if needed

### Maximum Cost Exposure

With all protections in place:

| Component | Hard Cap |
|-----------|----------|
| Railway | $10 (set by you) |
| Voyage AI | $0 (free tier) |
| Neon | $0 (free tier) |
| **Absolute Maximum** | **$10/month** |

Even under sustained attack, costs cannot exceed your caps.

### What Happens When Limits Hit

| Trigger | User Experience | Your Cost |
|---------|-----------------|-----------|
| Railway cap reached | App goes offline | $10 max |
| Rate limit hit | "Too many requests" (429) | Normal |
| Daily cap hit | "Limit reached, try tomorrow" | Normal |

### Cost Protection Checklist

- [ ] Railway: Set hard spending limit ($10)
- [ ] Add `slowapi` to requirements.txt
- [ ] Implement per-IP rate limiting on expensive endpoints
- [ ] Implement daily query counter with graceful 503 response
- [ ] (Optional) Set up Cloudflare for DDoS protection
- [ ] (Optional) Add custom domain for Cloudflare integration

---

## Lessons from Philippine Legal Assistant

1. **Don't trust "free" shared limits** - Groq free tier failed due to global exhaustion
2. **Check RAM requirements** - Local models need more than free tier provides
3. **Verify current pricing** - Google gutted their free tier without warning
4. **Test cold starts** - Understand what happens after idle periods
5. **Add payment methods early** - Some providers gate features behind card-on-file
6. **Consolidate where possible** - If you're paying for a service, use all its features (Railway handles both frontend and backend)
7. **Set spending caps immediately** - Never deploy without hard limits on all paid services
8. **Inspect data at every step** - Don't process garbage through your pipeline
9. **Don't trust synthetic data blindly** - ChatGPT generated 94% garbage despite prompts

---

## Future Considerations

- If traffic grows significantly, budget $10-15/month
- Consider caching frequent queries to reduce inference load
- Monitor Voyage AI token usage (200M is generous but finite)
- Watch for provider pricing changes (set calendar reminder quarterly)
- Review Railway usage dashboard weekly during first month
- If hosting both Quantum Simulator and LLM project, monitor combined usage closely

---

*Document version: 3.0*
*Last updated: December 22, 2025*
