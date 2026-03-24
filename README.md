# Local LLM Forge

> **Private AI fine-tuning, on your hardware. Your data never leaves your machine.**

Local LLM Forge is a self-hosted platform for the complete LLM fine-tuning lifecycle — from raw data to a deployed model — with a browser UI and zero cloud dependencies. Fine-tune any HuggingFace model (LLaMA, Mistral, Qwen, Gemma…), evaluate with automated metrics, run RAG over your own documents, and serve everything locally through Ollama.

**Why local?** Cloud fine-tuning APIs expose your training data to third parties, cost hundreds of dollars per run, and lock you into a vendor. Local LLM Forge keeps your data, your models, and your inference completely under your control.

---

## Features

| | |
|---|---|
| **Dataset creation** | Upload files, extract PDFs, scrape websites, pull from HuggingFace, or generate synthetic Q&A pairs via Ollama |
| **Data quality** | EDA statistics, automated cleaning (nulls, duplicates, length filtering), token distribution charts |
| **Fine-tuning** | SFT, DPO, and ORPO via HuggingFace TRL — LoRA or QLoRA (4-bit) on any model |
| **Live monitoring** | Real-time loss curves, perplexity, and training logs streamed to the UI |
| **Export** | Merge adapter → GGUF → register in Ollama in one click |
| **Evaluation** | ROUGE-1/2/L and BLEU scoring, sortable comparison table across all training jobs |
| **Inference** | Chat UI and side-by-side model comparison in the browser |
| **RAG** | Upload documents (PDF, TXT, MD), vector-search with Qdrant, grounded answers via Ollama |
| **Private by design** | No telemetry, no cloud calls, no API keys required — everything runs on localhost |

---

## Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14 · TanStack Query · Tailwind CSS |
| Backend API | FastAPI · Beanie ODM |
| Database | MongoDB |
| Task queue | Celery · Redis |
| Training | HuggingFace Transformers · PEFT · TRL · bitsandbytes |
| Inference | Ollama |
| Vector DB | Qdrant (embedded, no server needed) |
| Embeddings | sentence-transformers (local, no API) |

---

## Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | 8 GB VRAM (NVIDIA) | 24 GB (RTX 3090 / 4090 / A100) |
| CUDA | 12.1+ | 12.4+ |
| RAM | 16 GB | 32 GB |
| Disk | 50 GB free | 200 GB+ SSD |
| Python | 3.10+ | 3.11+ |
| Node.js | 18+ | 20+ |

**Required services:** MongoDB · Redis · Ollama

---

## Quick Start

### 1. Install services

```bash
# MongoDB
sudo apt install -y mongodb && sudo systemctl start mongodb

# Redis
sudo apt install -y redis-server && sudo systemctl start redis-server

# Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.2:latest   # pull at least one model
```

### 2. Backend

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # review and edit if needed
```

### 3. Dashboard

```bash
cd dashboard
npm install
cp .env.example .env.local
```

### 4. Start everything

Open four terminals:

```bash
# Terminal 1 — API server
cd backend && source venv/bin/activate
uvicorn app.main:app --reload --port 8010

# Terminal 2 — Celery worker  (--pool=solo required for CUDA)
cd backend && source venv/bin/activate
celery -A workers.celery_app worker --pool=solo --loglevel=info

# Terminal 3 — Frontend
cd dashboard
npm run dev -- --port 3010

# Terminal 4 — Ollama (if not running as a system service)
ollama serve
```

Open **http://localhost:3010** — the navbar shows a green "API online" badge when everything is connected.

---

## Workflow

```
Create Project
     │
     ├── Datasets ──────────────────────────────────────────────────────┐
     │    Upload / PDF / Web / HuggingFace / Synthetic                  │
     │    → EDA → Clean → Format (Alpaca / ChatML) → Tokenize           │
     │                                                                   ▼
     ├── Training Jobs ─────────────────────────────────────────── Evaluate
     │    SFT / DPO / ORPO · LoRA / QLoRA (4-bit)               ROUGE · BLEU
     │    → Live monitor → Export to Ollama                       Comparison
     │
     ├── Inference
     │    Chat · Side-by-side model comparison
     │
     └── RAG
          Upload documents → Index (local embeddings) → Grounded Q&A
```

---

## Documentation

Full documentation is in [PLATFORM_GUIDE.md](PLATFORM_GUIDE.md), covering:

- Background and problem statement
- Detailed architecture diagram
- Step-by-step usage for every feature
- Training method reference (SFT vs DPO vs ORPO explained)
- Evaluation metrics explained (ROUGE, BLEU, Perplexity)
- Troubleshooting common issues
- Storage layout and disk space guidance

---

## Project Structure

```
.
├── backend/                  FastAPI + Celery + ML pipeline
│   ├── app/
│   │   ├── main.py           Entry point
│   │   ├── config.py         Settings (reads from .env)
│   │   ├── models/           MongoDB document models (Beanie)
│   │   └── routers/          REST API endpoints
│   ├── ml/                   Core ML modules
│   │   ├── train.py          SFTTrainer
│   │   ├── train_dpo.py      DPOTrainer
│   │   ├── train_orpo.py     ORPO via IPO loss (TRL 0.29+)
│   │   ├── merge.py          Adapter merge + GGUF export
│   │   ├── evaluate.py       ROUGE / BLEU scoring
│   │   ├── rag_embed.py      Qdrant embed + retrieval
│   │   ├── generate_qa.py    Synthetic dataset generation
│   │   ├── pdf_extract.py    PDF text extraction
│   │   └── web_scrape.py     Web page scraping
│   ├── workers/              Celery task definitions
│   ├── storage/              Runtime data (gitignored — structure only)
│   ├── requirements.txt
│   └── .env.example
│
├── dashboard/                Next.js 14 frontend
│   ├── app/
│   │   ├── page.tsx          Home — project list
│   │   └── projects/[id]/    Project workspace
│   │       ├── layout.tsx    Sidebar navigation
│   │       ├── datasets/     Dataset pipeline UI
│   │       ├── jobs/         Training job management
│   │       ├── evaluate/     Metrics comparison table
│   │       ├── inference/    Chat and model comparison
│   │       └── rag/          RAG collections and chat
│   ├── lib/                  API client, hooks, types
│   ├── package.json
│   └── .env.example
│
├── PLATFORM_GUIDE.md         Full documentation
├── LICENSE                   MIT
└── README.md                 This file
```

---

## Configuration

All backend settings are in `backend/.env` (copy from `backend/.env.example`):

| Variable | Default | Description |
|---|---|---|
| `MONGO_URL` | `mongodb://localhost:27017` | MongoDB connection |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis for Celery |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `HF_TOKEN` | _(empty)_ | HuggingFace token — only needed for gated models |
| `MODELS_HF_DIR` | `./storage/models/hf` | Where model weights are cached |

---

## Security & Privacy

- **No telemetry** — the platform makes no outbound network calls except to services you configure (Ollama, MongoDB, Redis — all local by default)
- **No API keys required** — everything runs on open-source local software
- **Data isolation** — training data, model weights, and generated outputs stay in `backend/storage/` on your machine
- **HuggingFace token** — only needed if you use gated/private models; stored in your local `.env` file, never transmitted elsewhere

---

## Training Notes

- **QLoRA** (4-bit) recommended for GPUs with less than 24 GB VRAM — a 7B model fits in ~5 GB
- Celery worker **must use `--pool=solo`** to avoid CUDA multiprocessing deadlocks
- Restart the Celery worker after editing any file in `ml/` or `workers/` (module caching)
- Only one job runs at a time (GPU constraint) — others queue automatically

---

## License

[MIT License](LICENSE) — free to use, modify, and distribute.
