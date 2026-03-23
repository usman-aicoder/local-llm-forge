# LLM Platform

**A self-hosted, visual platform for fine-tuning, evaluating, and deploying local language models — no cloud required.**

Fine-tune any HuggingFace model (LLaMA, Mistral, Qwen, Gemma…) on your own data with a browser UI. Export to Ollama in one click. Compare models with automated metrics. Build RAG pipelines over your documents. Everything runs on your machine.

---

## Features

| | |
|---|---|
| **Dataset creation** | Upload files, extract PDFs, scrape websites, pull from HuggingFace, or generate synthetic Q&A pairs with Ollama |
| **Data quality** | EDA statistics, automated cleaning (nulls, duplicates, length filtering), token distribution charts |
| **Fine-tuning** | SFT, DPO, and ORPO via HuggingFace TRL — with LoRA or QLoRA (4-bit) on any model |
| **Live monitoring** | Real-time loss curves, perplexity, and training logs streamed to the UI |
| **Export** | Merge adapter → GGUF → register in Ollama with a single click |
| **Evaluation** | ROUGE-1/2/L and BLEU scoring, sortable comparison table across all jobs |
| **Inference** | Chat UI and side-by-side model comparison in the browser |
| **RAG** | Upload documents (PDF, TXT, MD), vector-search with Qdrant, stream grounded answers via Ollama |

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
| Vector DB | Qdrant (embedded) |
| Embeddings | sentence-transformers |

---

## Requirements

- **GPU**: NVIDIA GPU with 8 GB+ VRAM (24 GB recommended for 7B models)
- **CUDA**: 12.1+
- **RAM**: 16 GB+ system RAM
- **Disk**: 50 GB+ free
- **Python**: 3.10+
- **Node.js**: 18+
- **Services**: MongoDB · Redis · Ollama

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
cp .env.example .env.local    # edit if backend runs on a different port
```

### 4. Start everything

Open four terminals:

```bash
# Terminal 1 — API server
cd backend && source venv/bin/activate
uvicorn app.main:app --reload --port 8010

# Terminal 2 — Celery worker  (--pool=solo is required for CUDA)
cd backend && source venv/bin/activate
celery -A workers.celery_app worker --pool=solo --loglevel=info

# Terminal 3 — Frontend
cd dashboard
npm run dev -- --port 3010

# Terminal 4 — Ollama (if not running as a service)
ollama serve
```

Open **http://localhost:3010** — the navbar shows a green "API online" badge when everything is connected.

---

## Workflow Overview

```
Create Project
     │
     ├── Datasets ──────────────────────────────────────────────────────┐
     │    Upload / PDF / Web / HuggingFace / Synthetic                  │
     │    → EDA → Clean → Format (Alpaca/ChatML) → Tokenize             │
     │                                                                   ▼
     ├── Training Jobs ─────────────────────────────────────────── Evaluate
     │    SFT / DPO / ORPO · LoRA / QLoRA                         ROUGE · BLEU
     │    → Live monitor → Export to Ollama                        Comparison table
     │
     ├── Inference
     │    Chat · Side-by-side model comparison
     │
     └── RAG
          Upload documents → Index → Grounded Q&A chat
```

---

## Documentation

See [PLATFORM_GUIDE.md](PLATFORM_GUIDE.md) for the complete guide including:

- Background and problem statement
- Detailed architecture diagram
- Step-by-step usage for every feature
- Training method reference (SFT vs DPO vs ORPO)
- Evaluation metrics explained
- Troubleshooting
- Storage layout

---

## Project Structure

```
.
├── backend/                  FastAPI + Celery + ML
│   ├── app/
│   │   ├── main.py           FastAPI application entry point
│   │   ├── config.py         Settings (reads from .env)
│   │   ├── models/           Beanie (MongoDB) document models
│   │   └── routers/          API route handlers
│   ├── ml/                   Training, evaluation, RAG, data pipelines
│   │   ├── train.py          SFTTrainer
│   │   ├── train_dpo.py      DPOTrainer
│   │   ├── train_orpo.py     ORPO via IPO loss
│   │   ├── merge.py          Adapter merge + GGUF export
│   │   ├── evaluate.py       ROUGE / BLEU scoring
│   │   ├── rag_embed.py      Qdrant embed + search
│   │   ├── generate_qa.py    Synthetic dataset generation
│   │   ├── pdf_extract.py    PDF text extraction
│   │   └── web_scrape.py     Web page scraping
│   ├── workers/              Celery task definitions
│   ├── storage/              Runtime data (gitignored — structure only)
│   ├── requirements.txt
│   └── .env.example
│
├── dashboard/                Next.js 14 frontend
│   ├── app/                  App Router pages
│   │   ├── page.tsx          Home — project list
│   │   └── projects/[id]/    Project workspace
│   │       ├── layout.tsx    Sidebar navigation
│   │       ├── datasets/     Dataset pipeline UI
│   │       ├── jobs/         Training job management
│   │       ├── evaluate/     Metrics comparison table
│   │       ├── inference/    Chat and model comparison
│   │       └── rag/          RAG collections and chat
│   ├── components/           Shared React components
│   ├── lib/                  API client, hooks, types
│   ├── package.json
│   └── .env.example
│
├── PLATFORM_GUIDE.md         Full documentation
└── README.md                 This file
```

---

## Configuration

All backend settings are read from `backend/.env` (see `backend/.env.example`).

Key settings:

| Variable | Default | Description |
|---|---|---|
| `MONGO_URL` | `mongodb://localhost:27017` | MongoDB connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL for Celery |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `HF_TOKEN` | _(empty)_ | HuggingFace token for gated models |
| `MODELS_HF_DIR` | `./storage/models/hf` | Where to cache HF model weights |

All storage paths default to `./backend/storage/` and are created automatically on first run.

---

## Training Notes

- **QLoRA** (4-bit quantization) is recommended for GPUs with less than 24 GB VRAM. A 7B model fits in ~5 GB with QLoRA enabled.
- The Celery worker **must use `--pool=solo`** to avoid CUDA multiprocessing deadlocks.
- After editing any file in `ml/` or `workers/`, **restart the Celery worker** — it caches imported modules.
- Only one training job runs at a time (GPU constraint). Other jobs queue automatically.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
