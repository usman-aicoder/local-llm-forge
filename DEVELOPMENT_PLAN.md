# Local LLM Forge — Development Plan
> Phased roadmap for expanding the platform. Every phase ends with a full test pass before the next begins.

**Ground rule:** No phase starts until the previous phase passes all its tests. Existing features are regression-tested at the end of every phase.

---

## Table of Contents

- [Phase 0 — Stabilization & Test Baseline](#phase-0--stabilization--test-baseline)
- [Phase 1 — Developer Experience (Docker + Setup)](#phase-1--developer-experience-docker--setup)
- [Phase 2 — Training Performance (Unsloth + Resume + Config)](#phase-2--training-performance-unsloth--resume--config)
- [Phase 3 — Deployment & API (OpenAI API + vLLM + Model Card)](#phase-3--deployment--api-openai-api--vllm--model-card)
- [Phase 4 — User Experience (Model Browser + Notifications + Augmentation + FFT)](#phase-4--user-experience-model-browser--notifications--augmentation--fft)
- [Phase 5 — Analytics & Sharing (Experiment Tracking + HF Hub Push)](#phase-5--analytics--sharing-experiment-tracking--hf-hub-push)
- [Phase 6 — Scale (Multi-GPU + Node Pipeline)](#phase-6--scale-multi-gpu--node-pipeline)
- [Regression Test Checklist (runs after every phase)](#regression-test-checklist-runs-after-every-phase)

---

## Phase 0 — Stabilization & Test Baseline

**Goal:** Before adding anything new, document and verify that every existing feature works correctly. Build the test infrastructure that will catch regressions in later phases.

**Duration:** 3–4 days

---

### 0.1 Audit Existing Features

Walk through the entire platform end-to-end and document the current behavior of every feature. Record any existing bugs. Fix critical bugs before proceeding.

**Checklist:**
- [ ] Create project → appears in list
- [ ] Delete project → removed from list
- [ ] Create dataset (upload JSONL) → status = uploaded
- [ ] EDA runs and returns stats
- [ ] Clean runs and updates status to cleaned
- [ ] Format (Alpaca) runs → status = formatted
- [ ] Format (ChatML) runs → status = formatted
- [ ] Tokenize runs → train.jsonl + val.jsonl created
- [ ] SFT training job starts, runs, completes, adapter saved
- [ ] DPO training job starts, runs, completes, adapter saved
- [ ] ORPO training job starts, runs, completes, adapter saved
- [ ] Training logs stream live to UI
- [ ] Loss chart updates per epoch
- [ ] Export to Ollama → merge + GGUF + ollama create completes
- [ ] SFT evaluation runs, returns ROUGE/BLEU scores
- [ ] DPO/ORPO evaluation runs via Ollama, returns scores
- [ ] Inference chat sends prompt, receives streamed response
- [ ] Side-by-side inference with 2 models works
- [ ] RAG: create collection
- [ ] RAG: upload PDF → status = indexed
- [ ] RAG: query returns grounded answer with source citations
- [ ] RAG: delete document removes it from collection

---

### 0.2 Backend Unit Tests

**New directory:** `backend/tests/`

**Files to create:**

```
backend/tests/
├── conftest.py              Pytest fixtures (test DB, mock Celery)
├── test_projects.py         CRUD for projects
├── test_datasets.py         Dataset creation, status transitions
├── test_pipeline.py         EDA → clean → format → tokenize chain
├── test_jobs.py             Job creation, status updates
├── test_evaluation.py       _extract_id(), ROUGE/BLEU calculation
├── test_rag.py              Collection CRUD, document delete
└── test_health.py           /health endpoint
```

**Key tests to write:**

```python
# test_evaluation.py
def test_extract_id_with_dbref():
    from bson import DBRef, ObjectId
    from workers.evaluation_tasks import _extract_id
    oid = ObjectId()
    ref = DBRef("datasets", oid)
    assert _extract_id(ref) == str(oid)

def test_extract_id_with_link():
    # Beanie Link mock
    ...

# test_pipeline.py
def test_dataset_status_transitions():
    # uploaded → cleaned → formatted → tokenized
    # Each step must not be callable if previous step not done
    ...

# test_health.py
def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
```

**Run:** `cd backend && pytest tests/ -v`

---

### 0.3 Frontend Smoke Tests

**New directory:** `dashboard/tests/`

Using Playwright for end-to-end browser tests.

```bash
cd dashboard
npm install --save-dev @playwright/test
npx playwright install chromium
```

**Files to create:**
```
dashboard/tests/
├── homepage.spec.ts         Load page, see Projects heading
├── create-project.spec.ts   Create project, verify in list
├── navigation.spec.ts       Sidebar links work on project page
```

**Example:**
```typescript
// homepage.spec.ts
test('home page loads and shows Projects heading', async ({ page }) => {
  await page.goto('http://localhost:3010')
  await expect(page.getByRole('heading', { name: 'Projects' })).toBeVisible()
})

test('API online badge shows green', async ({ page }) => {
  await page.goto('http://localhost:3010')
  await expect(page.getByText('API online')).toBeVisible()
})
```

**Run:** `npx playwright test`

---

### 0.4 Create Regression Test Script

A single shell script that runs all tests and reports pass/fail. This becomes the gate between every phase.

**File:** `scripts/test-all.sh`

```bash
#!/bin/bash
set -e
echo "=== Backend unit tests ==="
cd backend && source venv/bin/activate && pytest tests/ -v --tb=short
echo ""
echo "=== Frontend smoke tests ==="
cd ../dashboard && npx playwright test --reporter=list
echo ""
echo "=== All tests passed ==="
```

---

### Phase 0 Exit Criteria

- [ ] All backend unit tests pass (`pytest tests/ -v`)
- [ ] All Playwright smoke tests pass
- [ ] No existing bugs introduced by test setup
- [ ] `scripts/test-all.sh` runs clean end-to-end

---
---

## Phase 1 — Developer Experience (Docker + Setup)

**Goal:** A new user can clone the repo and have the platform running with a single command. This is the most impactful change for adoption.

**Duration:** 2–3 days

---

### 1.1 Docker Compose Setup

**New files to create:**

```
docker-compose.yml           Main compose file
docker-compose.dev.yml       Development overrides (hot reload)
backend/Dockerfile
dashboard/Dockerfile
.dockerignore
backend/.dockerignore
dashboard/.dockerignore
```

---

#### `docker-compose.yml`

```yaml
version: "3.9"

services:

  mongodb:
    image: mongo:7
    restart: unless-stopped
    volumes:
      - mongo_data:/data/db
    ports:
      - "27017:27017"
    healthcheck:
      test: ["CMD", "mongosh", "--eval", "db.adminCommand('ping')"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    restart: unless-stopped
    ports:
      - "8010:8010"
    volumes:
      - ./backend/storage:/app/storage    # model weights + datasets persist
    environment:
      - MONGO_URL=mongodb://mongodb:27017
      - REDIS_URL=redis://redis:6379/0
      - OLLAMA_URL=http://host.docker.internal:11434   # Ollama runs on host
    env_file:
      - ./backend/.env
    depends_on:
      mongodb:
        condition: service_healthy
      redis:
        condition: service_healthy
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  worker:
    build:
      context: ./backend
      dockerfile: Dockerfile
    restart: unless-stopped
    command: celery -A workers.celery_app worker --pool=solo --loglevel=info
    volumes:
      - ./backend/storage:/app/storage
    environment:
      - MONGO_URL=mongodb://mongodb:27017
      - REDIS_URL=redis://redis:6379/0
      - OLLAMA_URL=http://host.docker.internal:11434
    env_file:
      - ./backend/.env
    depends_on:
      - backend
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  dashboard:
    build:
      context: ./dashboard
      dockerfile: Dockerfile
    restart: unless-stopped
    ports:
      - "3010:3010"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8010
    depends_on:
      - backend

volumes:
  mongo_data:
  redis_data:
```

---

#### `backend/Dockerfile`

```dockerfile
FROM python:3.11-slim

# CUDA base not included — user must use nvidia/cuda base if needed
# For GPU training, replace FROM with:
# FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Create storage directories
RUN mkdir -p storage/datasets/raw storage/datasets/cleaned \
    storage/datasets/formatted storage/datasets/tokenized \
    storage/models/hf storage/checkpoints \
    storage/merged_models storage/gguf_exports \
    storage/rag_documents storage/qdrant

EXPOSE 8010

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]
```

---

#### `dashboard/Dockerfile`

```dockerfile
FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

EXPOSE 3010

CMD ["npm", "start", "--", "-p", "3010"]
```

---

### 1.2 Environment Validation on Startup

**File to modify:** `backend/app/main.py`

Add a startup check that validates required services are reachable and logs clear errors if not:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await _validate_services()
    yield

async def _validate_services():
    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{settings.ollama_url}/api/tags")
            if r.status_code == 200:
                models = r.json().get("models", [])
                logger.info(f"Ollama: OK — {len(models)} model(s) available")
            else:
                logger.warning("Ollama: reachable but returned non-200")
    except Exception:
        logger.warning(
            f"Ollama not reachable at {settings.ollama_url}. "
            "Inference and export will fail until Ollama is running."
        )

    # Check storage directories exist and are writable
    for path_key in ["datasets_raw_dir", "checkpoints_dir", "merged_models_dir"]:
        p = settings.abs(getattr(settings, path_key))
        p.mkdir(parents=True, exist_ok=True)
    logger.info("Storage directories: OK")
```

---

### 1.3 Update README Quick Start

Update the Quick Start in `README.md` to offer two paths:

**Path A — Docker (recommended):**
```bash
git clone https://github.com/usman-aicoder/local-llm-forge
cd local-llm-forge
cp backend/.env.example backend/.env
docker compose up
# Open http://localhost:3010
```

**Path B — Manual (existing instructions):**
Existing 4-terminal setup remains documented for users who want to develop or have non-Docker environments.

---

### Phase 1 Tests

#### New tests to add: `backend/tests/test_docker_config.py`

```python
def test_env_example_has_all_required_keys():
    """Ensure .env.example covers every key used in config.py"""
    from app.config import Settings
    import re

    example = open("backend/.env.example").read()
    for field_name in Settings.model_fields:
        assert field_name.upper() in example.upper(), \
            f"Missing key in .env.example: {field_name}"

def test_storage_dirs_created_on_startup(tmp_path, monkeypatch):
    """Startup should create storage dirs if they don't exist"""
    monkeypatch.setenv("STORAGE_BASE", str(tmp_path / "storage"))
    # Re-import config to pick up monkeypatched env
    ...
```

#### Docker smoke test (manual, once):
```bash
docker compose up -d
sleep 15
curl http://localhost:8010/health    # must return {"status":"ok",...}
curl http://localhost:3010           # must return 200
docker compose down
```

#### Regression:
```bash
bash scripts/test-all.sh
```

### Phase 1 Exit Criteria

- [ ] `docker compose up` starts all services with no manual steps
- [ ] `/health` endpoint returns OK within 30s of compose up
- [ ] Dashboard loads at `localhost:3010`
- [ ] Startup logs show clear warnings (not crashes) when Ollama is unreachable
- [ ] All Phase 0 tests still pass
- [ ] `.env.example` validated against all config keys

---
---

## Phase 2 — Training Performance (Unsloth + Resume + Config)

**Goal:** Close the training speed gap with Unsloth Studio and add experiment reproducibility.

**Duration:** 4–5 days

---

### 2.1 Unsloth Library Integration

**Philosophy:** Unsloth is an optional drop-in accelerator. If it is not installed, training falls back to standard TRL with no error. Users can opt in per-job.

---

#### `backend/ml/train.py` — Add Unsloth path

Add a new parameter `use_unsloth: bool = False`. When `True`, swap the model loading:

```python
def _load_model_and_tokenizer(
    model_path: str,
    use_qlora: bool,
    use_unsloth: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
    max_seq_len: int,
):
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=max_seq_len,
                dtype=None,           # auto-detect
                load_in_4bit=use_qlora,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                use_gradient_checkpointing="unsloth",
                bias="none",
            )
            return model, tokenizer, "unsloth"
        except ImportError:
            # Unsloth not installed — fall through to standard path
            pass

    # Standard HuggingFace path (existing code)
    ...
    return model, tokenizer, "standard"
```

Apply the same pattern to `train_dpo.py` and `train_orpo.py`.

Log which backend was used: `on_log(f"Training backend: {backend}")` so users can see in the UI whether Unsloth was active.

---

#### `backend/app/models/job.py` — Add field

```python
use_unsloth: bool = False
```

---

#### `backend/app/routers/jobs.py` — Pass through

Pass `use_unsloth` from job document to the training task.

---

#### `dashboard/app/projects/[id]/jobs/` — UI toggle

In the "New Job" form, add a toggle in the LoRA settings section:

```
Use Unsloth optimization
  ☑ 2× faster training, 70% less VRAM (requires: pip install unsloth)
```

Show a soft warning if backend reports Unsloth is unavailable (via a new `/api/system/capabilities` endpoint that checks `importlib.util.find_spec("unsloth")`).

---

### 2.2 Resume Training from Checkpoint

---

#### `backend/ml/train.py` — Add resume support

```python
def run_training(
    ...
    resume_from_checkpoint: str | None = None,   # NEW: path to checkpoint dir
) -> dict:
    ...
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
```

The Hugging Face `Trainer` natively supports `resume_from_checkpoint`. No further changes to the training loop are needed.

---

#### `backend/app/models/job.py` — Add field

```python
resume_from_job_id: str | None = None    # job whose latest checkpoint to resume from
```

---

#### `backend/workers/training_tasks.py` — Resolve checkpoint path

```python
if job.get("resume_from_job_id"):
    source_job = db["training_jobs"].find_one({"_id": ObjectId(job["resume_from_job_id"])})
    if source_job and source_job.get("adapter_path"):
        resume_path = source_job["adapter_path"]
    else:
        # Find latest checkpoint subdirectory
        checkpoint_dir = settings.abs(settings.checkpoints_dir) / job["resume_from_job_id"]
        checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"),
                             key=lambda p: int(p.name.split("-")[1]))
        resume_path = str(checkpoints[-1]) if checkpoints else None
```

---

#### Dashboard UI changes

In the "New Job" form, add an optional "Resume from" dropdown that lists completed or failed jobs from the same project with the same base model. Selecting one pre-fills the checkpoint path.

---

### 2.3 Training Config Export / Import (YAML)

---

#### `backend/app/routers/jobs.py` — New endpoints

```python
@router.get("/{job_id}/config")
async def export_job_config(job_id: str):
    """Export all training parameters as a YAML file."""
    job = await TrainingJob.get(job_id)
    config = {
        "base_model": job.model_path,
        "training_method": job.training_method,
        "lora": {
            "use_qlora": job.use_qlora,
            "r": job.lora_r,
            "alpha": job.lora_alpha,
            "dropout": job.lora_dropout,
        },
        "training": {
            "learning_rate": job.learning_rate,
            "epochs": job.epochs,
            "batch_size": job.batch_size,
            "grad_accum": job.grad_accum,
            "max_seq_len": job.max_seq_len,
            "bf16": job.bf16,
        },
        "use_unsloth": job.use_unsloth,
    }
    yaml_content = yaml.dump(config, default_flow_style=False)
    return Response(
        content=yaml_content,
        media_type="application/x-yaml",
        headers={"Content-Disposition": f"attachment; filename=job-{job_id[:8]}-config.yaml"}
    )

@router.post("/from-config")
async def create_job_from_config(
    project_id: str,
    dataset_id: str,
    config_file: UploadFile,
):
    """Create a new job pre-filled from a YAML config file."""
    content = await config_file.read()
    config = yaml.safe_load(content)
    # Validate and create job with config values as defaults
    ...
```

---

#### Dashboard UI changes

- On every completed job card: **"Download Config"** button (calls `GET /jobs/{id}/config`)
- On "New Job" form header: **"Import Config"** button — file picker that uploads a YAML and pre-fills all form fields

---

### Phase 2 Tests

#### `backend/tests/test_unsloth_fallback.py`

```python
def test_training_falls_back_when_unsloth_not_installed(monkeypatch):
    """If unsloth is not installed, training should proceed with standard TRL"""
    import builtins
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "unsloth":
            raise ImportError("unsloth not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    # Run a minimal training call and verify it completes
    ...

def test_unsloth_flag_stored_in_job():
    """use_unsloth field persists to MongoDB"""
    ...
```

#### `backend/tests/test_resume.py`

```python
def test_resume_path_resolved_from_job_id():
    """Given a source job_id, the correct checkpoint path is resolved"""
    ...

def test_resume_ignored_when_no_checkpoint_exists():
    """If source job has no checkpoint, training starts fresh without error"""
    ...
```

#### `backend/tests/test_config_export.py`

```python
def test_config_export_produces_valid_yaml(client, sample_job):
    resp = client.get(f"/jobs/{sample_job.id}/config")
    assert resp.status_code == 200
    config = yaml.safe_load(resp.content)
    assert "base_model" in config
    assert "lora" in config
    assert "training" in config

def test_config_import_prefills_job_fields(client, sample_yaml, sample_project, sample_dataset):
    resp = client.post("/jobs/from-config", data={...}, files={"config_file": sample_yaml})
    assert resp.status_code == 200
    job = resp.json()
    assert job["lora_r"] == sample_yaml["lora"]["r"]
```

#### End-to-end training regression

Run a minimal SFT job (1 epoch, tiny dataset) both with and without Unsloth flag. Both must complete and produce an adapter file. Compare adapter sizes — they should be identical (Unsloth produces the same adapter format).

#### Regression:
```bash
bash scripts/test-all.sh
```

### Phase 2 Exit Criteria

- [ ] Unsloth toggle appears in New Job form
- [ ] Training with `use_unsloth=True` succeeds when Unsloth is installed
- [ ] Training with `use_unsloth=True` falls back gracefully when Unsloth is NOT installed
- [ ] Resume from checkpoint: job picks up from last saved checkpoint, not epoch 1
- [ ] Config YAML export downloads a valid YAML with all hyperparameters
- [ ] Config YAML import pre-fills all form fields correctly
- [ ] All Phase 0 and Phase 1 tests still pass

---
---

## Phase 3 — Deployment & API (OpenAI API + vLLM + Model Card)

**Goal:** Make trained models usable beyond the platform's own UI.

**Duration:** 3–4 days

---

### 3.1 OpenAI-Compatible API

**New file:** `backend/app/routers/openai_compat.py`

This is a proxy layer — it translates OpenAI API format to Ollama format and streams the response back in OpenAI format. No new ML code needed.

```python
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import httpx, json, time, uuid

router = APIRouter(prefix="/v1", tags=["openai-compat"])

@router.get("/models")
async def list_models():
    """Return all Ollama models in OpenAI format."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{settings.ollama_url}/api/tags")
    models = resp.json().get("models", [])
    return {
        "object": "list",
        "data": [
            {
                "id": m["name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local-llm-forge",
            }
            for m in models
        ]
    }

@router.post("/chat/completions")
async def chat_completions(request: Request):
    """Proxy chat completions to Ollama in OpenAI format."""
    body = await request.json()
    model = body.get("model")
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 512)

    # Convert OpenAI messages format to Ollama prompt
    prompt = _messages_to_prompt(messages)

    ollama_payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }

    if stream:
        return StreamingResponse(
            _stream_ollama_as_openai(ollama_payload),
            media_type="text/event-stream"
        )
    else:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{settings.ollama_url}/api/generate",
                json=ollama_payload
            )
        ollama_resp = resp.json()
        return _wrap_as_openai_response(model, ollama_resp["response"])

async def _stream_ollama_as_openai(payload: dict):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST",
                f"{settings.ollama_url}/api/generate", json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": payload["model"],
                    "choices": [{
                        "index": 0,
                        "delta": {"content": data.get("response", "")},
                        "finish_reason": "stop" if data.get("done") else None,
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"
```

**Register in `main.py`:**
```python
from app.routers import openai_compat
app.include_router(openai_compat.router)
```

**Result:** Any tool that supports OpenAI-compatible APIs can point to `http://localhost:8010/v1` and use fine-tuned models directly. This includes LangChain, LlamaIndex, Continue.dev, Cursor, Open WebUI, and countless others.

---

### 3.2 vLLM Export Option

**New file:** `backend/ml/export_vllm.py`

After the merge step, add a second export path that skips GGUF and prepares a vLLM-compatible directory:

```python
def prepare_vllm_export(merged_model_dir: str, output_dir: str) -> dict:
    """
    vLLM needs only the merged HuggingFace model directory.
    No conversion needed — just copy/symlink and generate the launch command.
    """
    from pathlib import Path
    import shutil

    src = Path(merged_model_dir)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    # Symlink (saves disk space) or copy
    for f in src.iterdir():
        target = dst / f.name
        if not target.exists():
            target.symlink_to(f.resolve())

    launch_cmd = (
        f"python -m vllm.entrypoints.openai.api_server \\\n"
        f"  --model {dst} \\\n"
        f"  --port 8000 \\\n"
        f"  --dtype bfloat16"
    )
    return {
        "vllm_model_path": str(dst),
        "launch_command": launch_cmd,
    }
```

**UI change:** On the completed job page, the export section shows two buttons:
- **Export to Ollama** (existing)
- **Export for vLLM** (new) — triggers the export, then shows the generated launch command in a copyable code block

---

### 3.3 Auto Model Card Generation

**New file:** `backend/ml/model_card.py`

```python
def generate_model_card(job: dict, evaluation: dict | None, dataset: dict) -> str:
    """Generate a HuggingFace-compatible MODEL_CARD.md for a trained job."""

    method_descriptions = {
        "sft":  "Supervised Fine-Tuning (SFT) using TRL SFTTrainer",
        "dpo":  "Direct Preference Optimization (DPO) using TRL DPOTrainer",
        "orpo": "Identity Preference Optimization (IPO) via TRL DPOTrainer (loss_type=ipo), equivalent to ORPO",
    }

    eval_section = ""
    if evaluation:
        eval_section = f"""
## Evaluation Results

| Metric | Score |
|--------|-------|
| ROUGE-1 | {evaluation.get('rouge1', 'N/A'):.4f} |
| ROUGE-2 | {evaluation.get('rouge2', 'N/A'):.4f} |
| ROUGE-L | {evaluation.get('rougeL', 'N/A'):.4f} |
| BLEU | {evaluation.get('bleu', 'N/A'):.4f} |
| Perplexity | {evaluation.get('perplexity', 'N/A'):.4f} |

Evaluated on {evaluation.get('num_samples', 'N/A')} samples from the validation split.
"""

    return f"""---
base_model: {job['model_path']}
library_name: peft
tags:
  - lora
  - {"qlora" if job.get('use_qlora') else "lora"}
  - {job.get('training_method', 'sft')}
  - local-llm-forge
---

# {job.get('name', 'Fine-tuned Model')}

Fine-tuned from [{job['model_path']}](https://huggingface.co/{job['model_path']}) using
[Local LLM Forge](https://github.com/usman-aicoder/local-llm-forge).

## Training Details

| Parameter | Value |
|-----------|-------|
| Method | {method_descriptions.get(job.get('training_method', 'sft'), 'SFT')} |
| Base model | `{job['model_path']}` |
| Dataset size | {dataset.get('num_rows', 'N/A')} rows |
| LoRA rank (r) | {job.get('lora_r', 16)} |
| LoRA alpha | {job.get('lora_alpha', 32)} |
| QLoRA (4-bit) | {job.get('use_qlora', True)} |
| Learning rate | {job.get('learning_rate', 'N/A')} |
| Epochs | {job.get('epochs', 3)} |
| Max seq len | {job.get('max_seq_len', 1024)} |
| Unsloth | {job.get('use_unsloth', False)} |
{eval_section}
## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{job['model_path']}")
model = PeftModel.from_pretrained(base, "<path-to-this-adapter>")
tokenizer = AutoTokenizer.from_pretrained("{job['model_path']}")
```

## License

This adapter inherits the license of the base model: `{job['model_path']}`.
Trained with [Local LLM Forge](https://github.com/usman-aicoder/local-llm-forge) (MIT).
"""
```

**New endpoint:** `GET /jobs/{job_id}/model-card` — generates and returns the markdown as a downloadable file.

**UI:** "Download Model Card" button on the completed job page.

---

### Phase 3 Tests

#### `backend/tests/test_openai_compat.py`

```python
def test_list_models_returns_openai_format(client, mock_ollama):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)
    for model in data["data"]:
        assert "id" in model
        assert "object" in model

def test_chat_completions_non_streaming(client, mock_ollama):
    resp = client.post("/v1/chat/completions", json={
        "model": "llama3.2:latest",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert data["choices"][0]["message"]["role"] == "assistant"

def test_chat_completions_streaming_returns_sse(client, mock_ollama):
    # Verify SSE format: lines start with "data: "
    ...
```

#### `backend/tests/test_model_card.py`

```python
def test_model_card_contains_required_sections(sample_job, sample_dataset):
    from ml.model_card import generate_model_card
    card = generate_model_card(sample_job, evaluation=None, dataset=sample_dataset)
    assert "base_model:" in card        # YAML frontmatter
    assert "## Training Details" in card
    assert sample_job["model_path"] in card

def test_model_card_includes_eval_when_provided(sample_job, sample_evaluation, sample_dataset):
    from ml.model_card import generate_model_card
    card = generate_model_card(sample_job, sample_evaluation, sample_dataset)
    assert "## Evaluation Results" in card
    assert "ROUGE-L" in card

def test_model_card_endpoint_returns_markdown(client, sample_completed_job):
    resp = client.get(f"/jobs/{sample_completed_job.id}/model-card")
    assert resp.status_code == 200
    assert "text/markdown" in resp.headers["content-type"]
```

#### Manual integration test:
- Point a local LangChain script at `http://localhost:8010/v1` and verify a chat completion goes through
- Download a model card and verify it renders correctly on GitHub

#### Regression:
```bash
bash scripts/test-all.sh
```

### Phase 3 Exit Criteria

- [ ] `GET /v1/models` returns all Ollama models in OpenAI format
- [ ] `POST /v1/chat/completions` works in both streaming and non-streaming mode
- [ ] LangChain / LlamaIndex can use `http://localhost:8010/v1` as a base URL
- [ ] vLLM export generates a valid model directory and correct launch command
- [ ] Model card downloads as a valid markdown file with all fields populated
- [ ] Model card endpoint returns 404 gracefully if job has no evaluation
- [ ] All previous phase tests still pass

---
---

## Phase 4 — User Experience (Model Browser + Notifications + Augmentation + FFT)

**Goal:** Make the platform accessible to non-ML users and address the small-dataset problem.

**Duration:** 5–6 days

---

### 4.1 HuggingFace Model Browser

**New endpoint:** `backend/app/routers/models.py` — extend existing file

```python
@router.get("/browse")
async def browse_hf_models(
    size: str | None = None,      # "1b", "3b", "7b", "13b", "30b+"
    task: str = "text-generation",
    limit: int = 20,
):
    """Search HuggingFace Hub for instruction-tuned models."""
    params = {
        "filter": "text-generation",
        "sort": "downloads",
        "direction": -1,
        "limit": limit,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://huggingface.co/api/models",
            params=params,
            headers={"Authorization": f"Bearer {settings.hf_token}"} if settings.hf_token else {}
        )
    models = resp.json()

    # Check which are already downloaded locally
    local_models = set()
    hf_dir = settings.abs(settings.models_hf_dir)
    if hf_dir.exists():
        local_models = {d.name for d in hf_dir.iterdir() if d.is_dir()}

    return [
        {
            "id": m["id"],
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "tags": m.get("tags", []),
            "is_downloaded": any(m["id"].replace("/", "--") in lm for lm in local_models),
            "vram_estimate_gb": _estimate_vram(m),
        }
        for m in models
    ]

def _estimate_vram(model_info: dict) -> dict:
    """Estimate VRAM needed based on model parameter count."""
    tags = model_info.get("tags", [])
    # Rough heuristic from tags like "1.5B", "7B", etc.
    for tag in tags:
        for size, vram_qlora, vram_lora in [
            ("1b", 2, 4), ("3b", 3, 7), ("7b", 5, 14),
            ("13b", 9, 26), ("30b", 18, 60), ("70b", 40, 140),
        ]:
            if size in tag.lower():
                return {"qlora_gb": vram_qlora, "full_lora_gb": vram_lora}
    return {"qlora_gb": "unknown", "full_lora_gb": "unknown"}
```

**UI — Model picker modal:**

When the user clicks "Choose Base Model" in the New Job form, open a modal that:
- Shows a searchable list of popular models with download counts, VRAM estimate, and "Downloaded" badge
- Allows typing a custom HuggingFace model ID directly
- Has size filter tabs: All / 1–3B / 7B / 13B / 30B+
- Clicking a model selects it and closes the modal

---

### 4.2 Training Notifications

**Frontend change:** `dashboard/lib/notifications.ts` (new file)

```typescript
export async function requestNotificationPermission(): Promise<boolean> {
  if (!("Notification" in window)) return false
  if (Notification.permission === "granted") return true
  const perm = await Notification.requestPermission()
  return perm === "granted"
}

export function sendNotification(title: string, body: string, jobId?: string) {
  if (Notification.permission !== "granted") return
  const n = new Notification(title, {
    body,
    icon: "/favicon.ico",
    tag: jobId,      // prevents duplicate notifications for the same job
  })
  n.onclick = () => {
    if (jobId) window.focus()
    n.close()
  }
}
```

**Usage in job monitor page:**

```typescript
// When job status transitions from "running" to "completed" or "failed"
useEffect(() => {
  if (prevStatus === "running" && status === "completed") {
    sendNotification(
      "Training Complete",
      `Job "${job.name}" finished successfully.`,
      job.id
    )
  }
  if (prevStatus === "running" && status === "failed") {
    sendNotification(
      "Training Failed",
      `Job "${job.name}" encountered an error. Check the logs.`,
      job.id
    )
  }
}, [status])
```

**Backend — optional webhook:**

Add `webhook_url: str | None = None` to the TrainingJob model. On job completion in `training_tasks.py`, if set, POST a JSON payload to that URL (for Slack/Discord/ntfy integration).

```python
if job.get("webhook_url"):
    try:
        requests.post(job["webhook_url"], json={
            "text": f"Training job '{job['name']}' completed. "
                    f"Status: {status}. ROUGE-L: {rouge_l:.4f}"
        }, timeout=10)
    except Exception:
        pass   # Webhook failure must not affect job status
```

---

### 4.3 Dataset Augmentation

**New file:** `backend/ml/augment.py`

```python
"""
Dataset augmentation — uses Ollama to expand small datasets by generating
paraphrased versions of existing instruction/output pairs.
"""

PARAPHRASE_PROMPT = """
You are a training-data augmentation assistant.

Given the instruction-output pair below, generate {n} alternative versions.
Each version should ask the same question differently and give an equivalent answer.

Original instruction: {instruction}
Original output: {output}

Output ONLY a JSON array of objects with "instruction" and "output" keys.
"""

def augment_dataset(
    input_path: str,
    output_path: str,
    ollama_url: str,
    model_name: str,
    paraphrases_per_row: int = 2,
    max_rows_to_augment: int = 200,
    on_log: Callable[[str], None] | None = None,
) -> dict:
    rows = [json.loads(l) for l in open(input_path)]
    augmented = list(rows)   # start with originals

    for i, row in enumerate(rows[:max_rows_to_augment]):
        if on_log:
            on_log(f"Augmenting row {i+1}/{min(len(rows), max_rows_to_augment)}")

        prompt = PARAPHRASE_PROMPT.format(
            n=paraphrases_per_row,
            instruction=row.get("instruction", ""),
            output=row.get("output", ""),
        )
        try:
            resp = httpx.post(
                f"{ollama_url}/api/generate",
                json={"model": model_name, "prompt": prompt, "stream": False},
                timeout=60,
            )
            pairs = _parse_pairs(resp.json().get("response", ""))
            augmented.extend(pairs)
        except Exception as e:
            if on_log:
                on_log(f"  Row {i+1} augmentation failed: {e}")

    # Write augmented dataset
    with open(output_path, "w") as f:
        for row in augmented:
            f.write(json.dumps(row) + "\n")

    return {
        "original_rows": len(rows),
        "augmented_rows": len(augmented),
        "new_rows_added": len(augmented) - len(rows),
    }
```

**Backend:** New Celery task `augment_dataset_task` and endpoint `POST /datasets/{id}/augment`.

**UI:** In the dataset pipeline, after the Clean step, add an "Augment" option — shows a form with model picker and number of paraphrases per row. The augmented dataset becomes a new version (original is preserved).

---

### 4.4 Full Fine-Tuning (FFT) Mode

**New file:** `backend/ml/train_fft.py`

```python
"""
Full Fine-Tuning — trains ALL model parameters (no LoRA adapter).
Practical only for models ≤ 3B on consumer hardware, or larger models on A100+.
Output: a full HuggingFace model directory (no merge step needed).
"""

def run_fft_training(
    job_id: str,
    model_path: str,
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    learning_rate: float = 1e-5,    # Lower than LoRA — all params update
    epochs: int = 3,
    batch_size: int = 1,
    grad_accum: int = 16,
    max_seq_len: int = 1024,
    bf16: bool = True,
    on_epoch_end=None,
    on_log=None,
) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    if on_log:
        on_log("[FFT] Loading model in full precision (no quantization)")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    # No LoRA — all parameters trainable
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Training is identical to SFT but without PEFT wrapping
    ...

    # Output is a full model directory, not an adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {"model_path": output_dir, "is_full_model": True}
```

**Key difference in the export flow:** FFT jobs skip the merge step (there's no adapter to merge). The export goes directly from the output directory to GGUF conversion.

**UI warning:** When FFT is selected, show the estimated VRAM requirement based on the selected model size, and warn if it likely exceeds available VRAM.

---

### Phase 4 Tests

#### `backend/tests/test_augmentation.py`

```python
def test_augment_returns_more_rows_than_input(tmp_path, mock_ollama_augment):
    input_file = tmp_path / "input.jsonl"
    input_file.write_text(
        '{"instruction": "What is 2+2?", "output": "4"}\n' * 10
    )
    output_file = tmp_path / "augmented.jsonl"

    result = augment_dataset(
        str(input_file), str(output_file),
        ollama_url="http://localhost:11434",
        model_name="llama3.2:latest",
        paraphrases_per_row=2,
    )
    assert result["augmented_rows"] > result["original_rows"]

def test_augment_preserves_original_rows(tmp_path, mock_ollama_augment):
    """Original rows must appear verbatim in augmented output."""
    ...

def test_augment_handles_ollama_failure_gracefully(tmp_path, mock_ollama_failure):
    """Augmentation should continue (skipping failed rows) not crash."""
    ...
```

#### `backend/tests/test_fft.py`

```python
def test_fft_job_skips_merge_step():
    """is_full_model=True jobs should not call run_merge()"""
    ...

def test_fft_output_is_loadable_hf_model(tmp_path, tiny_model):
    """FFT output directory must be loadable by AutoModelForCausalLM"""
    ...
```

#### `backend/tests/test_notifications.py`

```python
def test_webhook_called_on_job_completion(requests_mock, sample_job_with_webhook):
    """Webhook URL is called with job status payload when job completes"""
    ...

def test_webhook_failure_does_not_affect_job_status(requests_mock):
    """If webhook returns 500, job status is still saved correctly"""
    ...
```

#### Regression:
```bash
bash scripts/test-all.sh
```

### Phase 4 Exit Criteria

- [ ] HuggingFace model browser opens as a modal from the New Job form
- [ ] Model browser shows VRAM estimates and "Downloaded" badges
- [ ] Browser notification fires when a training job completes
- [ ] Webhook fires when `webhook_url` is set on a job
- [ ] Dataset augmentation produces a new JSONL with more rows than input
- [ ] Augmentation failure on individual rows does not abort the whole job
- [ ] FFT mode trains without LoRA and produces a full model directory
- [ ] FFT jobs skip the merge step in the export flow
- [ ] All previous phase tests still pass

---
---

## Phase 5 — Analytics & Sharing (Experiment Tracking + HF Hub Push)

**Goal:** Enable systematic experimentation and community sharing.

**Duration:** 4–5 days

---

### 5.1 Experiment Tracking Dashboard

**New backend endpoint:** `GET /projects/{id}/experiments/summary`

Returns all completed jobs for a project with their evaluation metrics, hyperparameters, and training duration — formatted for charting:

```python
@router.get("/{project_id}/experiments/summary")
async def experiments_summary(project_id: str):
    jobs = await TrainingJob.find(
        TrainingJob.project_id == PydanticObjectId(project_id),
        TrainingJob.status == "completed"
    ).to_list()

    evaluations = {
        str(e.job_id): e
        for e in await Evaluation.find(
            In(Evaluation.job_id, [j.id for j in jobs])
        ).to_list()
    }

    return [
        {
            "job_id": str(j.id),
            "name": j.name,
            "method": j.training_method,
            "learning_rate": j.learning_rate,
            "lora_r": j.lora_r,
            "epochs": j.epochs,
            "use_qlora": j.use_qlora,
            "use_unsloth": getattr(j, "use_unsloth", False),
            "rouge_l": evaluations[str(j.id)].rouge_l if str(j.id) in evaluations else None,
            "bleu": evaluations[str(j.id)].bleu if str(j.id) in evaluations else None,
            "perplexity": evaluations[str(j.id)].perplexity if str(j.id) in evaluations else None,
            "final_eval_loss": j.final_eval_loss,
            "duration_minutes": j.duration_minutes,
            "created_at": j.created_at.isoformat(),
        }
        for j in jobs
    ]
```

**New frontend page:** `dashboard/app/projects/[id]/experiments/page.tsx`

Charts to show:
1. **ROUGE-L vs. Learning Rate** — scatter plot, one dot per job, colored by training method. Shows which learning rate range works best.
2. **ROUGE-L vs. LoRA Rank** — shows impact of rank on quality.
3. **Training Duration vs. ROUGE-L** — shows efficiency (is more training time worth it?).
4. **Method Comparison Bar Chart** — average ROUGE-L per training method (SFT / DPO / ORPO).
5. **Best Configuration Summary Card** — highlights the job with the best ROUGE-L and lists its hyperparameters as a suggested starting point.

Library: Use `recharts` (already common in Next.js projects) — `npm install recharts`.

**"Clone with Modifications" button:** On any job in the table, a button opens the New Job form pre-filled with that job's configuration. The user changes one parameter and runs the experiment. Makes systematic ablations frictionless.

---

### 5.2 HuggingFace Hub Push

**New endpoint:** `POST /jobs/{job_id}/push-to-hub`

```python
@router.post("/{job_id}/push-to-hub")
async def push_to_hub(job_id: str, repo_name: str, private: bool = False):
    """Push the LoRA adapter to HuggingFace Hub."""
    job = await TrainingJob.get(PydanticObjectId(job_id))

    if not job.adapter_path:
        raise HTTPException(400, "Job has no adapter — training must complete first")

    if not settings.hf_token:
        raise HTTPException(400, "HF_TOKEN not set in .env — required for Hub push")

    # Run in background (can take a few minutes for upload)
    task = push_to_hub_task.delay(job_id, repo_name, private)
    return {"task_id": task.id, "status": "queued"}
```

**New Celery task:** `backend/workers/export_tasks.py` — add `push_to_hub_task`

```python
@celery_app.task(bind=True, name="workers.export_tasks.push_to_hub_task")
def push_to_hub_task(self, job_id: str, repo_name: str, private: bool) -> dict:
    from huggingface_hub import HfApi
    from ml.model_card import generate_model_card

    db = _get_db()
    job = db["training_jobs"].find_one({"_id": ObjectId(job_id)})
    evaluation = db["evaluations"].find_one({"job_id": ObjectId(job_id)})
    dataset = db["datasets"].find_one({"_id": job["dataset_id"]}) if job.get("dataset_id") else {}

    api = HfApi(token=settings.hf_token)

    # Create repo if it doesn't exist
    repo_url = api.create_repo(repo_name, private=private, exist_ok=True)

    # Upload adapter files
    api.upload_folder(
        folder_path=job["adapter_path"],
        repo_id=repo_name,
        ignore_patterns=["checkpoint-*"],   # Skip intermediate checkpoints
    )

    # Upload model card
    card = generate_model_card(job, evaluation, dataset or {})
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_name,
    )

    hub_url = f"https://huggingface.co/{repo_name}"
    db["training_jobs"].update_one(
        {"_id": ObjectId(job_id)},
        {"$set": {"hub_url": hub_url}}
    )
    return {"hub_url": hub_url}
```

**UI:** On completed job page, a "Push to HuggingFace Hub" button opens a small modal: repo name input (pre-filled with project name + job name), private/public toggle. On success, shows the HuggingFace URL with a "View on Hub" link.

---

### Phase 5 Tests

#### `backend/tests/test_experiments.py`

```python
def test_experiments_summary_includes_all_completed_jobs(client, project_with_jobs):
    resp = client.get(f"/projects/{project_with_jobs.id}/experiments/summary")
    assert resp.status_code == 200
    data = resp.json()
    completed_count = sum(1 for j in project_with_jobs.jobs if j.status == "completed")
    assert len(data) == completed_count

def test_experiments_summary_excludes_running_jobs(client, project_with_jobs):
    data = client.get(f"/projects/{project_with_jobs.id}/experiments/summary").json()
    statuses = [j["job_id"] for j in data]
    # No running/queued jobs in the summary
    ...
```

#### `backend/tests/test_hub_push.py`

```python
def test_push_to_hub_requires_hf_token(client, sample_completed_job, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "")
    resp = client.post(f"/jobs/{sample_completed_job.id}/push-to-hub",
                       json={"repo_name": "test/repo"})
    assert resp.status_code == 400
    assert "HF_TOKEN" in resp.json()["detail"]

def test_push_to_hub_requires_completed_adapter(client, sample_running_job):
    resp = client.post(f"/jobs/{sample_running_job.id}/push-to-hub",
                       json={"repo_name": "test/repo"})
    assert resp.status_code == 400

def test_hub_url_saved_to_job_on_success(mock_hf_api, sample_completed_job):
    # Mock HfApi upload, verify hub_url is saved to MongoDB
    ...
```

#### Regression:
```bash
bash scripts/test-all.sh
```

### Phase 5 Exit Criteria

- [ ] Experiments page shows charts for all completed + evaluated jobs
- [ ] "Clone with Modifications" opens New Job form with all fields pre-filled
- [ ] HF Hub push queues a Celery task and returns a task ID
- [ ] On push success, `hub_url` is saved to the job in MongoDB
- [ ] Push returns 400 (not 500) when HF_TOKEN is missing
- [ ] Push returns 400 (not 500) when job has no adapter
- [ ] All previous phase tests still pass

---
---

## Phase 6 — Scale (Multi-GPU + Node Pipeline)

**Goal:** Support larger models and more complex data workflows. Highest engineering effort — only after all previous phases are stable.

**Duration:** 3–4 weeks

---

### 6.1 Multi-GPU Training

**Architecture change:** Replace Celery with a subprocess-based launcher for training jobs. Use HuggingFace `accelerate` for multi-GPU distribution.

**New file:** `backend/ml/train_accelerate.py`

```python
def build_accelerate_config(num_gpus: int, use_deepspeed: bool) -> str:
    """Generate accelerate config YAML for the given GPU count."""
    if num_gpus == 1:
        return yaml.dump({
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "NO",
            "num_processes": 1,
        })
    else:
        return yaml.dump({
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "DEEPSPEED" if use_deepspeed else "MULTI_GPU",
            "num_processes": num_gpus,
            "deepspeed_config": {
                "zero_stage": 2,
                "offload_optimizer_device": "none",
            } if use_deepspeed else {},
        })
```

**Training worker change:** Instead of calling `run_training()` directly in a Celery task, write the config to a temp file and launch:

```bash
accelerate launch --config_file /tmp/accelerate_{job_id}.yaml \
  backend/ml/train_script.py --job-id {job_id}
```

Progress is still streamed via Redis (the launched subprocess writes to Redis directly).

**UI:** Add "Number of GPUs" field to the New Job form (1, 2, 4, 8 — automatically hidden/disabled for single-GPU systems).

---

### 6.2 Node-Based Data Pipeline

**Frontend:** New page `dashboard/app/projects/[id]/pipeline/page.tsx`

A visual drag-and-drop pipeline builder using `reactflow`:

```bash
npm install @xyflow/react
```

**Node types to implement:**

| Node | Inputs | Outputs | Maps to |
|---|---|---|---|
| Source: Upload | — | rows | `POST /datasets` |
| Source: HuggingFace | dataset_id | rows | existing HF pull |
| Source: PDF | file | rows | `ml/pdf_extract.py` + `ml/generate_qa.py` |
| Source: Web Scrape | url | rows | `ml/web_scrape.py` |
| Process: Clean | rows | rows | `ml/clean.py` |
| Process: Filter | rows + condition | rows | new: filter by column value / length |
| Process: Augment | rows + config | rows | `ml/augment.py` (Phase 4) |
| Process: Format | rows + format | formatted rows | `ml/format_dataset.py` |
| Merge: Union | rows A + rows B | rows | concat two datasets |
| Merge: Deduplicate | rows | rows | cross-source deduplication |
| Output: Tokenize | formatted rows + model | dataset | `ml/tokenize_dataset.py` |

**Backend:** New endpoint `POST /datasets/pipeline/execute` that accepts a JSON graph (nodes + edges) and executes each node in topological order as a Celery task chain.

**Storage:** Pipeline definitions saved as JSON in MongoDB so users can reuse and share them.

---

### Phase 6 Tests

#### Multi-GPU:
```python
def test_accelerate_config_single_gpu():
    config = build_accelerate_config(num_gpus=1, use_deepspeed=False)
    parsed = yaml.safe_load(config)
    assert parsed["distributed_type"] == "NO"

def test_accelerate_config_multi_gpu_deepspeed():
    config = build_accelerate_config(num_gpus=4, use_deepspeed=True)
    parsed = yaml.safe_load(config)
    assert parsed["distributed_type"] == "DEEPSPEED"
    assert parsed["num_processes"] == 4

def test_single_gpu_training_still_works_after_accelerate_refactor():
    """Regression: 1-GPU path must behave identically to pre-Phase-6"""
    ...
```

#### Node pipeline:
```python
def test_pipeline_topological_sort():
    """Nodes are executed in dependency order, not insertion order"""
    ...

def test_pipeline_with_merge_node():
    """Union node correctly concatenates two source datasets"""
    ...

def test_pipeline_execution_is_idempotent():
    """Re-running the same pipeline produces the same output"""
    ...
```

#### Regression:
```bash
bash scripts/test-all.sh
```

### Phase 6 Exit Criteria

- [ ] Single-GPU training works identically to pre-Phase-6 (regression)
- [ ] Multi-GPU training launches with correct accelerate config
- [ ] Progress streams correctly from multi-GPU subprocess to Redis to UI
- [ ] Node pipeline: basic chain (Upload → Clean → Format → Tokenize) executes correctly
- [ ] Node pipeline: Union node merges two datasets
- [ ] Node pipeline: re-execution of a saved pipeline produces same output
- [ ] All previous phase tests still pass

---
---

## Regression Test Checklist (runs after every phase)

This is the non-negotiable check that runs at the end of every phase before starting the next one. All items must pass.

```bash
bash scripts/test-all.sh
```

### Existing features to manually verify (rotate through these):

**Dataset pipeline:**
- [ ] Upload JSONL → EDA → Clean → Format (Alpaca) → Format (ChatML) → Tokenize
- [ ] PDF upload → synthetic generation → dataset created
- [ ] HuggingFace dataset pull

**Training:**
- [ ] SFT job: start → runs → completes → adapter file exists
- [ ] DPO job: start → runs → completes → adapter file exists
- [ ] ORPO job: start → runs → completes → adapter file exists
- [ ] Failed job: shows error in log, does not block queue
- [ ] Queued job: second job starts after first completes

**Export:**
- [ ] Merge + GGUF + Ollama export completes
- [ ] Exported model appears in Inference model dropdown

**Evaluation:**
- [ ] SFT evaluation runs and returns ROUGE/BLEU scores
- [ ] DPO/ORPO evaluation runs via Ollama and returns scores
- [ ] Evaluation comparison table shows correct best-value highlighting

**Inference:**
- [ ] Chat sends prompt and receives streamed response
- [ ] Side-by-side comparison with two models works

**RAG:**
- [ ] Create collection
- [ ] Upload PDF → indexed
- [ ] Query returns grounded answer with source citation
- [ ] Delete document removes it

**API:**
- [ ] `GET /health` returns `{"status":"ok"}`
- [ ] All endpoints return 422 (not 500) for invalid input

---

## Summary Table

| Phase | What's Built | Duration | Key Test |
|---|---|---|---|
| 0 | Test baseline, audit, regression script | 3–4 days | pytest + playwright all green |
| 1 | Docker Compose, env validation | 2–3 days | `docker compose up` + full regression |
| 2 | Unsloth integration, resume, YAML config | 4–5 days | Fallback test + resume test + config roundtrip |
| 3 | OpenAI API, vLLM export, model card | 3–4 days | LangChain connects + card downloads |
| 4 | Model browser, notifications, augmentation, FFT | 5–6 days | Augment adds rows + FFT skips merge |
| 5 | Experiments dashboard, HF Hub push | 4–5 days | Charts render + push queues task |
| 6 | Multi-GPU, node pipeline | 3–4 weeks | Single-GPU regression + pipeline chain |

**Total estimated duration:** 10–12 weeks (phases 0–5), +4 weeks for phase 6.

**Recommendation:** Phases 0–3 are the highest priority. They remove the biggest adoption barriers (setup complexity, speed gap, deployment options) and introduce no architectural risk. Phase 4 and 5 can overlap once Phase 3 is stable. Phase 6 is optional for v1.
