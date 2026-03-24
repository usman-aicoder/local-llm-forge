# Local LLM Forge — Complete Guide
> Private AI fine-tuning, on your hardware. Train, evaluate, and deploy local models — no cloud required.

---

## Table of Contents

1. [Background](#1-background)
2. [Problem Statement](#2-problem-statement)
3. [What This Platform Solves](#3-what-this-platform-solves)
4. [Architecture Overview](#4-architecture-overview)
5. [System Requirements](#5-system-requirements)
6. [Installation & Startup](#6-installation--startup)
7. [Core Concepts](#7-core-concepts)
8. [Step-by-Step Usage Guide](#8-step-by-step-usage-guide)
   - [Step 1 — Create a Project](#step-1--create-a-project)
   - [Step 2 — Create a Dataset](#step-2--create-a-dataset)
   - [Step 3 — Inspect & Explore (EDA)](#step-3--inspect--explore-eda)
   - [Step 4 — Clean the Dataset](#step-4--clean-the-dataset)
   - [Step 5 — Format the Dataset](#step-5--format-the-dataset)
   - [Step 6 — Tokenize](#step-6--tokenize)
   - [Step 7 — Create a Training Job](#step-7--create-a-training-job)
   - [Step 8 — Monitor Training](#step-8--monitor-training)
   - [Step 9 — Export to Ollama](#step-9--export-to-ollama)
   - [Step 10 — Evaluate](#step-10--evaluate)
   - [Step 11 — Inference & Comparison](#step-11--inference--comparison)
   - [Step 12 — RAG (Retrieval-Augmented Generation)](#step-12--rag-retrieval-augmented-generation)
9. [Training Method Reference](#9-training-method-reference)
10. [Evaluation Metrics Reference](#10-evaluation-metrics-reference)
11. [Troubleshooting](#11-troubleshooting)
12. [Storage Layout](#12-storage-layout)

---

## 1. Background

Large Language Models (LLMs) such as LLaMA, Mistral, and Qwen have democratized AI: powerful, capable models are freely downloadable and run on consumer hardware. However, a raw general-purpose model is rarely good enough out of the box for specialized domains.

**Domain adaptation** — teaching a model to speak like a doctor, answer like a legal assistant, or respond in a company's tone — has traditionally required:

- Submitting proprietary data to cloud providers (OpenAI fine-tuning API, Vertex AI, etc.)
- Hiring ML engineers fluent in HuggingFace, PyTorch, and distributed training
- Managing a complex chain of tools: Jupyter notebooks, custom scripts, experiment trackers, deployment pipelines

This platform was built to collapse that entire workflow into a single, visual, self-hosted application that runs on a single GPU workstation.

---

## 2. Problem Statement

### Data Privacy
Organizations with sensitive data — medical records, legal documents, customer interactions, proprietary manuals — cannot legally or safely send that data to third-party API providers. They need to fine-tune models **locally**, where data never leaves their infrastructure.

### Complexity Barrier
Fine-tuning a model the "right" way involves at least a dozen distinct technical steps: data collection, inspection, cleaning, formatting, tokenization, model selection, LoRA configuration, training loop, evaluation, merging, quantization, and deployment. Each step has pitfalls. No single tool has unified them.

### Cost of Cloud Fine-Tuning
Cloud fine-tuning APIs (OpenAI, Google) charge per token and lock you into their ecosystem. A single fine-tuning run on a moderate dataset can cost hundreds of dollars. Local training on a single 4090 is effectively free after hardware purchase.

### The RAG vs. Fine-Tuning Decision
Many teams do not know when to fine-tune versus when to use Retrieval-Augmented Generation (RAG). Both are needed in practice. They currently live in completely separate toolchains. This platform supports both in the same project.

---

## 3. What This Platform Solves

This platform provides a **complete, self-hosted, visual pipeline** for:

| Capability | What It Does |
|---|---|
| **Dataset creation** | Upload files, extract PDFs, scrape websites, pull HuggingFace datasets, or generate synthetic Q&A with Ollama |
| **Data quality** | EDA statistics (token distributions, null rates, duplicates), automated cleaning, flagging of too-short/too-long rows |
| **Dataset formatting** | Convert raw data to Alpaca or ChatML instruction format compatible with SFTTrainer |
| **Tokenization** | Tokenize with the target model's tokenizer, produce train/validation splits |
| **Training** | SFT (Supervised Fine-Tuning), DPO (Direct Preference Optimization), and ORPO — all with LoRA/QLoRA on any HuggingFace model |
| **Real-time monitoring** | Live loss curves, perplexity, epoch progress, and raw training logs streamed to the UI |
| **Export** | Merge adapter into base model → convert to GGUF → push to local Ollama with a single click |
| **Evaluation** | Automated ROUGE-1/2/L and BLEU scoring against validation data, comparison table across all jobs |
| **Inference** | Chat with any Ollama model directly in the UI; side-by-side comparison of multiple models on the same prompt |
| **RAG** | Upload documents (PDF, TXT, MD), embed with a local embedding model, vector-search with Qdrant, stream answers via Ollama |

Everything runs locally. No cloud accounts required.

---

## 4. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Browser  →  Next.js Dashboard (port 3010)                  │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP / SSE
┌────────────────────────▼────────────────────────────────────┐
│  FastAPI Backend (port 8010)                                │
│  ├── /projects  /datasets  /jobs  /evaluations             │
│  ├── /inference  /rag  /tasks                               │
│  └── Beanie ODM  →  MongoDB (port 27017)                   │
└────────┬─────────────────────────────────┬──────────────────┘
         │ Celery tasks                     │ Ollama HTTP
┌────────▼──────────┐             ┌────────▼──────────┐
│  Redis (port 6379)│             │  Ollama (port 11434)│
│  Task queue       │             │  Model inference   │
│  Progress pub/sub │             │  Model management  │
└────────┬──────────┘             └────────────────────┘
         │
┌────────▼────────────────────────────────────────────────────┐
│  Celery Worker  (--pool=solo, GPU process)                  │
│  ├── ml/train.py          SFTTrainer                        │
│  ├── ml/train_dpo.py      DPOTrainer                        │
│  ├── ml/train_orpo.py     DPOTrainer (loss_type=ipo)        │
│  ├── ml/merge.py          Adapter merge + GGUF export       │
│  ├── ml/evaluate.py       ROUGE / BLEU scoring              │
│  ├── ml/rag_embed.py      Qdrant vector store               │
│  └── ml/generate_qa.py    Synthetic data generation         │
└─────────────────────────────────────────────────────────────┘

Storage (./backend/storage/):
  datasets/raw/       cleaned/    formatted/    tokenized/
  models/hf/          checkpoints/
  merged_models/      gguf_exports/
  rag_documents/      qdrant/
```

**Technology Stack**

| Layer | Technology |
|---|---|
| Frontend | Next.js 14 (App Router), TanStack Query, Tailwind CSS |
| Backend API | FastAPI + Beanie ODM (async MongoDB driver) |
| Database | MongoDB |
| Task Queue | Celery + Redis |
| ML Training | HuggingFace Transformers, PEFT, TRL, bitsandbytes |
| Inference | Ollama |
| Vector DB | Qdrant (embedded, file-based) |
| Embeddings | sentence-transformers (BAAI/bge-small-en-v1.5) |

---

## 5. System Requirements

### Minimum
| Component | Requirement |
|---|---|
| GPU | NVIDIA GPU with 8 GB VRAM (RTX 3070 / 4060 Ti or better) |
| RAM | 16 GB system RAM |
| Disk | 50 GB free (models are large) |
| OS | Linux (Ubuntu 22.04+ recommended) |
| CUDA | 12.1+ |
| Python | 3.10+ |
| Node.js | 18+ |

### Recommended
- **24 GB VRAM** (RTX 3090 / 4090 / A100) — allows training 7B models comfortably with QLoRA
- **32 GB system RAM** — needed when merging large models
- **200 GB+ SSD** — multiple 7B model checkpoints fill up fast

### Required Services
- **MongoDB** — document store for all metadata
- **Redis** — Celery task queue and live progress streaming
- **Ollama** — local model inference and model management

---

## 6. Installation & Startup

### 1. Install system services

```bash
# MongoDB
sudo apt install -y mongodb
sudo systemctl start mongodb

# Redis
sudo apt install -y redis-server
sudo systemctl start redis-server

# Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &

# Pull at least one model to use for inference and synthetic generation
ollama pull llama3.2:latest
```

### 2. Set up the backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Create .env file to override defaults
cat > .env << 'EOF'
MONGO_URL=mongodb://localhost:27017
REDIS_URL=redis://localhost:6379/0
OLLAMA_URL=http://localhost:11434
HF_TOKEN=your_huggingface_token_here   # only needed for gated models
EOF
```

### 3. Set up the dashboard

```bash
cd dashboard
npm install
```

### 4. Start everything

Open **four terminals**:

```bash
# Terminal 1 — API server
cd backend && source venv/bin/activate
uvicorn app.main:app --reload --port 8010

# Terminal 2 — Celery worker (must use --pool=solo for CUDA)
cd backend && source venv/bin/activate
celery -A workers.celery_app worker --pool=solo --loglevel=info

# Terminal 3 — Frontend
cd dashboard
npm run dev -- --port 3010

# Terminal 4 — Ollama (if not already running as a service)
ollama serve
```

Open your browser at **http://localhost:3010**

The navbar shows a green "API online" indicator when FastAPI is reachable.

---

## 7. Core Concepts

### LoRA — Low-Rank Adaptation
Instead of updating all billions of parameters in a base model (which requires enormous memory), LoRA inserts small trainable matrices alongside the frozen base model weights. Training only these small matrices uses 10–100× less memory and produces an "adapter" — a small file you can swap onto any copy of the base model.

**Key parameters:**
- `r` (rank) — size of the LoRA matrices. Higher = more capacity, more memory. Typical: 8–64.
- `alpha` — scaling factor. Usually set to 2× rank.
- `dropout` — regularization. Usually 0.05.
- `target_modules` — which weight matrices to adapt. Default covers all attention + MLP layers.

### QLoRA — Quantized LoRA
QLoRA loads the base model in **4-bit NF4 quantization** (using bitsandbytes), dramatically reducing VRAM usage. A 7B model that normally needs 14 GB fits in ~5 GB. LoRA adapters train in full precision on top. This is the recommended mode for consumer GPUs.

### Training Methods

| Method | Use Case | Requires |
|---|---|---|
| **SFT** | Teach the model to follow instructions / answer questions | instruction + output pairs |
| **DPO** | Teach the model to prefer good responses over bad ones | prompt + chosen + rejected triples |
| **ORPO** | Combined SFT + preference alignment in one pass | prompt + chosen + rejected triples |

### Dataset Formats

| Format | Structure | Best For |
|---|---|---|
| **Alpaca** | `{"instruction": "...", "input": "...", "output": "..."}` | Single-turn Q&A |
| **ChatML** | `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]` | Multi-turn conversation |

---

## 8. Step-by-Step Usage Guide

---

### Step 1 — Create a Project

A **Project** is the top-level container. All datasets, training jobs, evaluations, and RAG collections live inside a project. Think of it as a workspace for a specific model customization goal.

1. Go to **http://localhost:3010**
2. Click **+ New Project**
3. Enter a name (e.g. "Customer Support Bot") and an optional description
4. Click **Create Project**

You will be taken to the project overview, which shows a summary dashboard with links to all sections.

---

### Step 2 — Create a Dataset

Navigate to **Datasets** in the sidebar. Click **+ Create Dataset**. Choose a source:

#### Option A — Upload a File
Upload a `.jsonl`, `.json`, or `.csv` file. Each row must have `instruction` and `output` fields (for SFT) or `prompt`, `chosen`, `rejected` (for DPO/ORPO).

#### Option B — PDF Extraction + Synthetic Generation
1. Select source type **PDF**
2. Upload a PDF file
3. Choose an Ollama model to generate Q&A pairs (e.g. `llama3.2:latest`)
4. Set the number of pairs per chunk

The platform extracts text from the PDF, splits it into overlapping 600-word chunks, and calls Ollama to generate instruction/output pairs from each chunk. This process runs in the background; the page updates automatically when complete.

**Good for:** technical manuals, product documentation, knowledge bases, research papers.

#### Option C — Web Scraping
1. Select source type **Web**
2. Enter a URL (must be publicly accessible)
3. Select a generation model

The platform scrapes the page text and generates Q&A pairs from it using the same chunk-based pipeline as PDF extraction.

#### Option D — HuggingFace Dataset
1. Select source type **HuggingFace**
2. Enter the dataset identifier (e.g. `tatsu-lab/alpaca`)
3. Optionally set the split and max rows

The dataset is downloaded from the HuggingFace Hub. If the dataset is gated (requires a login), set `HF_TOKEN` in your `.env` file.

#### Option E — Synthetic Generation
Generate instruction/output pairs from a text prompt or topic using Ollama directly, without any external document.

---

### Step 3 — Inspect & Explore (EDA)

After a dataset is created (status: **uploaded**), click **Open →** on the dataset card to enter the dataset pipeline.

The **EDA** (Exploratory Data Analysis) tab shows:

| Metric | What It Means |
|---|---|
| Total rows | How many examples are in the dataset |
| Null rate | Percentage of rows missing instruction or output |
| Duplicate rate | Percentage of exact-duplicate rows |
| Avg instruction tokens | Mean length of the question/prompt side |
| Avg output tokens | Mean length of the answer/response side |
| P95 total tokens | 95th percentile of total sequence length — useful for setting `max_seq_len` |
| Token histogram | Distribution of sequence lengths across the dataset |
| Flagged too short | Rows where output is suspiciously short (< 10 tokens) |
| Flagged too long | Rows that would be truncated at the target `max_seq_len` |

Use this information to decide whether cleaning is needed and what `max_seq_len` to use in training.

---

### Step 4 — Clean the Dataset

The **Clean** tab lets you remove low-quality rows automatically:

- **Remove nulls** — drops rows missing instruction or output
- **Remove duplicates** — keeps only unique rows
- **Remove too-short outputs** — configurable minimum token threshold
- **Remove too-long sequences** — drops rows exceeding a token limit

After reviewing the cleaning report (how many rows were removed and why), click **Apply Cleaning**. The dataset status updates to **cleaned**.

---

### Step 5 — Format the Dataset

The **Format** tab converts your cleaned data into the exact text format the model expects during training.

Choose a format:

- **Alpaca** — classic instruction-following template:
  ```
  ### Instruction:
  {instruction}

  ### Response:
  {output}
  ```

- **ChatML** — conversation format used by Qwen, Mistral, LLaMA-3:
  ```
  <|im_start|>user
  {instruction}<|im_end|>
  <|im_start|>assistant
  {output}<|im_end|>
  ```

For DPO/ORPO datasets, ensure your data already has `prompt`, `chosen`, and `rejected` columns — no formatting template is needed.

Click **Apply Format**. Status updates to **formatted**.

---

### Step 6 — Tokenize

The **Tokenize** tab prepares the formatted data for training:

1. Select the **base model** you intend to train (this determines which tokenizer is used)
2. Set `max_seq_len` — the maximum token length per example (use the P95 value from EDA as a guide; 1024–2048 is typical)
3. Set the **validation split** ratio (default: 10%)

Click **Tokenize**. The platform:
- Loads the tokenizer for the selected model
- Encodes all examples
- Splits into `train.jsonl` and `val.jsonl`
- Reports how many examples were retained vs. truncated

Status updates to **tokenized**. The dataset is now ready to train on.

---

### Step 7 — Create a Training Job

Navigate to **Jobs** in the sidebar. Click **+ New Job**.

#### Required Fields

| Field | Description |
|---|---|
| **Job name** | Descriptive name for this experiment |
| **Dataset** | Select a tokenized dataset from this project |
| **Base model** | HuggingFace model ID (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`) or local path |
| **Training method** | SFT, DPO, or ORPO |

#### LoRA Settings

| Setting | Typical Value | Notes |
|---|---|---|
| Use QLoRA | Enabled | Recommended; saves ~70% VRAM |
| LoRA rank (r) | 16 | Higher = more capacity, more memory |
| LoRA alpha | 32 | Usually 2× rank |
| LoRA dropout | 0.05 | Light regularization |

#### Training Hyperparameters

| Setting | Typical Value | Notes |
|---|---|---|
| Learning rate | 2e-4 (SFT), 8e-6 (DPO/ORPO) | DPO/ORPO need lower LR |
| Epochs | 3 | Start with 3; increase if underfitting |
| Batch size | 1–2 | Limited by VRAM |
| Gradient accumulation | 8 | Effective batch = batch_size × grad_accum |
| Max seq len | 1024 | Match the tokenization setting |
| BF16 | Enabled | Use on Ampere+ GPUs (3090, 4090, A100) |

Click **Start Training**. The job is queued and picked up by the Celery worker.

> **Note:** Only one training job can run at a time since training occupies the full GPU. Other jobs remain in the **queued** state until the current one finishes.

---

### Step 8 — Monitor Training

Click on a running job to open the **Job Monitor** page.

You will see:

- **Status badge** — queued / running / completed / failed
- **Live loss chart** — train loss and eval loss per epoch, updating in real time via SSE
- **Perplexity** — computed from eval loss at each epoch (lower is better)
- **Training log** — raw output from the trainer, scrollable, auto-follows tail
- **Progress bar** — current epoch / total epochs

If a job **fails**, the error is shown in the log. Common causes:
- Out of VRAM → reduce batch size, enable QLoRA, reduce max_seq_len or LoRA rank
- Dataset format mismatch → go back and verify the format step
- Missing columns for DPO/ORPO → dataset needs `prompt`, `chosen`, `rejected` columns

---

### Step 9 — Export to Ollama

Once a job completes, the job monitor shows an **Export to Ollama** button.

This triggers a three-step background process:

1. **Merge** — the LoRA adapter is merged back into the base model weights, producing a full-precision HuggingFace model
2. **Quantize to GGUF** — the merged model is converted to GGUF format using `llama.cpp`
3. **Load into Ollama** — the GGUF file is registered as a new Ollama model (e.g. `llmplatform/abc123`)

The export status is shown in the UI. Once complete, the model is available in **Inference** and **RAG** immediately.

> **Note:** The GGUF conversion requires `llama.cpp` to be installed. If it is missing, install it via `pip install llama-cpp-python` or build from source and ensure `llama-quantize` is on PATH.

---

### Step 10 — Evaluate

Navigate to **Evaluate** in the sidebar.

The evaluation table lists all completed jobs that have been evaluated. Columns:

| Column | Meaning |
|---|---|
| **ROUGE-1** | Unigram overlap between generated and reference answer |
| **ROUGE-2** | Bigram overlap |
| **ROUGE-L** | Longest common subsequence |
| **BLEU** | Precision of n-grams in the generated text vs. reference |
| **Perplexity** | From the final training checkpoint (lower = better fit) |
| **Human Score** | Optional manual annotation (editable in job evaluate detail) |

The **best value per column is highlighted in green**. This makes it easy to identify which training configuration produced the best results.

Click **Run Evaluation** on a completed job (from the job monitor) to trigger the evaluation task. For SFT jobs, it generates responses using the merged PEFT model. For DPO/ORPO jobs, it calls Ollama with the exported model.

Click **Details →** on any row to see per-sample results: the prompt, ground-truth response, and generated response for each evaluated example.

**How to interpret results:**
- ROUGE-L > 0.30 and BLEU > 0.10 generally indicates the model is staying on-topic
- If ROUGE-L drops after more epochs, the model may be overfitting
- Low BLEU with reasonable ROUGE-L suggests the model is paraphrasing rather than copying — often desirable

---

### Step 11 — Inference & Comparison

Navigate to **Inference** in the sidebar.

#### Chat Mode
1. Select a model from the dropdown (base Ollama models and all fine-tuned exports appear here)
2. Adjust generation parameters: Temperature, Max Tokens, Repeat Penalty
3. Type a prompt and press Enter or click Send

#### Side-by-Side Comparison
Switch to the **Side by Side** tab. Select two different models. Type a prompt once and both models respond simultaneously. This is the fastest way to see whether fine-tuning improved the model's behavior on your target domain.

**Generation Parameters:**

| Parameter | Effect |
|---|---|
| Temperature | Higher = more creative / random. Lower = more deterministic. Range: 0.1–2.0 |
| Max Tokens | Hard cap on response length |
| Repeat Penalty | Penalizes the model for repeating itself. 1.1–1.3 is typical |

---

### Step 12 — RAG (Retrieval-Augmented Generation)

Navigate to **RAG** in the sidebar.

RAG lets you ask questions over a specific set of documents without fine-tuning. The model's answer is grounded in text retrieved from your collection at query time.

#### Create a Collection
1. Click **+ New** in the Collections sidebar
2. Enter a name (e.g. "Product Manuals")
3. The collection is created with the default embedding model (`BAAI/bge-small-en-v1.5`)

#### Upload Documents
1. Select the collection
2. Click **+ Upload**
3. Select a PDF, TXT, or MD file

The document is saved and a background task:
- Extracts text from the file
- Splits it into chunks
- Encodes each chunk with the embedding model
- Stores the vectors in the Qdrant collection

Status progresses: **uploaded** → **processing** → **indexed**

#### Query
Once at least one document is **indexed**:
1. Select a model from the dropdown (any Ollama model)
2. Type a question in the chat box
3. Press Enter

The platform:
1. Encodes the question with the same embedding model
2. Retrieves the top-5 most relevant chunks from Qdrant
3. Streams the answer from Ollama, providing the retrieved chunks as context

The **source citations** are shown below each answer — click any source to expand the retrieved text.

#### When to Use RAG vs. Fine-Tuning

| Use Case | Recommended Approach |
|---|---|
| Answer questions from a specific document set | RAG |
| Change the model's tone, style, or persona | Fine-tuning |
| Domain knowledge that changes frequently | RAG |
| Teaching the model a new task format | Fine-tuning |
| Large knowledge base (thousands of docs) | RAG |
| Small, high-quality instruction dataset | Fine-tuning |
| Both style + grounded facts needed | Fine-tune first, then RAG |

---

## 9. Training Method Reference

### SFT — Supervised Fine-Tuning
**Best for:** Teaching the model to follow a new instruction format, answer questions in a specific style, or develop domain expertise from a labeled dataset.

**Dataset format:**
```jsonl
{"text": "<formatted prompt including instruction and expected response>"}
```
(The platform's Format step produces this automatically.)

**Typical hyperparameters:**
- Learning rate: 1e-4 to 3e-4
- Epochs: 2–5
- Batch × grad_accum effective batch: 16–32

**When SFT is enough:** You have a clean instruction/output dataset and want the model to generalize to similar prompts.

---

### DPO — Direct Preference Optimization
**Best for:** Improving response quality by teaching the model to prefer well-written, accurate, or safe responses over poor ones.

**Dataset format:**
```jsonl
{"prompt": "user question", "chosen": "good response", "rejected": "bad response"}
```

**Typical hyperparameters:**
- Learning rate: 5e-7 to 1e-5 (much lower than SFT)
- Epochs: 1–3
- Beta: 0.1 (controls how far the model can deviate from the reference)

**When DPO is appropriate:** You have existing SFT outputs you can rank, or you can generate chosen/rejected pairs from human feedback or an LLM judge.

---

### ORPO — Odds Ratio Preference Optimization (implemented as IPO)
**Best for:** Combined SFT + alignment in one training pass. More efficient than running SFT then DPO separately. Works well with smaller datasets.

**Dataset format:** Same as DPO — `prompt`, `chosen`, `rejected`.

**Key advantage over DPO:** No reference model is needed. Training is faster and uses less memory.

**Note:** In TRL 0.29+, ORPO is implemented via `DPOTrainer` with `loss_type="ipo"` — the Identity Preference Optimization loss, which is the closest reference-free equivalent available.

---

## 10. Evaluation Metrics Reference

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Measures **overlap** between the generated text and the reference (ground truth):

- **ROUGE-1** — overlap of individual words (unigrams). Measures vocabulary coverage.
- **ROUGE-2** — overlap of word pairs (bigrams). Measures phrase accuracy.
- **ROUGE-L** — longest common subsequence. Measures fluency and structural similarity.

All scores are between 0 and 1. Higher is better. 0.30+ ROUGE-L is generally considered decent for open-ended generation.

### BLEU (Bilingual Evaluation Understudy)

Measures **precision** of n-grams in the generated text relative to the reference. Originally designed for machine translation; adapted here for response evaluation.

- Score range: 0.0 to 1.0 (higher is better)
- Values below 0.1 are common for open-ended tasks
- Penalizes responses that are shorter than the reference (brevity penalty)

### Perplexity

Measures how well the model predicts the validation set. Computed as `exp(eval_loss)`.

- **Lower is better**
- A perplexity of 1.0 means perfect prediction (not realistic)
- Typical values after fine-tuning: 2.0–20.0 depending on task difficulty
- Perplexity rising across epochs signals overfitting

### Practical Interpretation

| Scenario | What To Do |
|---|---|
| ROUGE-L is low despite clear training signal | More data, more epochs, or verify formatting |
| ROUGE-L is high but answers are wrong | Model is memorizing phrasing, not understanding — check data quality |
| Eval loss rising while train loss falls | Overfitting — reduce epochs or add more training data |
| All metrics equal across jobs | The fine-tuning had no effect — check that the adapter was applied correctly |

---

## 11. Troubleshooting

### API shows "offline" in the navbar
Verify the FastAPI server is running: `curl http://localhost:8010/health`

### Training job stays in "queued" indefinitely
The Celery worker is not running. Start it:
```bash
cd backend && source venv/bin/activate
celery -A workers.celery_app worker --pool=solo --loglevel=info
```

### CUDA out of memory during training
- Enable QLoRA (4-bit quantization)
- Reduce batch size to 1
- Reduce `max_seq_len` (try 512 or 768)
- Reduce LoRA rank (try r=8)
- Close other GPU applications

### Export to Ollama fails with "llama-quantize not found"
Install llama.cpp:
```bash
pip install llama-cpp-python
# or build from source:
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make
# then add to PATH
```

### RAG document stuck at "processing"
Check the Celery worker logs. Common causes:
- Embedding model not downloaded yet (first run downloads it automatically, but needs internet)
- Qdrant storage directory permission issues — ensure `./storage/qdrant/` is writable

### Dataset generation returns 0 rows
- The Ollama model may have returned malformed JSON. Try a larger/smarter model.
- The source document may be too short (< 100 words after extraction).
- Check the Celery worker logs for the exact error.

### Evaluation returns all zeros
- For DPO/ORPO jobs: the model must be exported to Ollama first before evaluation runs.
- For SFT jobs: ensure `val.jsonl` exists in `storage/datasets/tokenized/{dataset_id}/`.

### Celery picks up old code after file changes
Celery with `--pool=solo` caches imported modules. **Restart the worker** after any change to `ml/` or `workers/` files:
```bash
# Kill and restart
kill $(pgrep -f "celery.*worker")
celery -A workers.celery_app worker --pool=solo --loglevel=info
```

---

## 12. Storage Layout

All data is stored under `backend/storage/`:

```
storage/
├── datasets/
│   ├── raw/              Original uploaded files
│   ├── cleaned/          After the cleaning step
│   ├── formatted/        Alpaca/ChatML formatted JSONL
│   └── tokenized/
│       └── {dataset_id}/
│           ├── train.jsonl
│           └── val.jsonl
├── models/
│   └── hf/               Downloaded HuggingFace model weights
├── checkpoints/
│   └── {job_id}/         Per-epoch checkpoint saves
├── merged_models/
│   └── {job_id}/         Adapter merged into full model
├── gguf_exports/
│   └── {job_id}/         GGUF quantized file for Ollama
├── rag_documents/
│   └── {collection_id}/  Uploaded source documents
└── qdrant/               Qdrant vector store (embedded)
```

**Disk space guidance:**
- A 7B model in HF format: ~14 GB
- A 7B model in GGUF Q4_K_M: ~4 GB
- A LoRA adapter: ~100–400 MB
- A tokenized dataset of 10K rows: ~50 MB

To reclaim disk space, delete jobs from the UI (which removes checkpoints and merged models) or manually clean old entries from the storage directories.

---

*Built with FastAPI · MongoDB · Celery · HuggingFace · Ollama · Qdrant · Next.js*
