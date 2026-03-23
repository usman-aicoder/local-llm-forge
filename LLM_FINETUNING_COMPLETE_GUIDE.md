# Complete LLM Fine-Tuning Implementation Guide
> LoRA · QLoRA · Instruction Tuning · Production Pipeline

---

## Table of Contents

1. [Overview](#overview)
2. [Hardware Requirements](#hardware)
3. [Environment Setup](#setup)
4. [Stage 1 — Define Use Case](#stage1)
5. [Stage 2 — Data Collection](#stage2)
6. [Stage 3 — Data Inspection & EDA](#stage3)
7. [Stage 4 — Data Cleaning](#stage4)
8. [Stage 5 — Dataset Formatting](#stage5)
9. [Stage 6 — Tokenization & Splits](#stage6)
10. [Stage 7 — Model Selection](#stage7)
11. [Stage 8 — LoRA / QLoRA Configuration](#stage8)
12. [Stage 9 — Training](#stage9)
13. [Stage 10 — Saving & Merging](#stage10)
14. [Stage 11 — Evaluation](#stage11)
15. [Stage 12 — Iteration Loop](#stage12)
16. [Stage 13 — Deployment](#stage13)
17. [How to Give This as a Skill to AI](#skill)
18. [Quick Reference Card](#reference)

---

## 1. Overview <a name="overview"></a>

This guide walks through a production-grade LLM fine-tuning pipeline — every stage from raw data to a deployed, monitored model. It reflects the same workflow used by real ML engineering teams.

**What you will build:**

Raw Data → Cleaned Dataset → Instruction Format → Fine-Tuned Model → Deployed API

**Methods covered:**

- **LoRA** (Low-Rank Adaptation) — attaches small trainable adapter layers to a frozen base model, updating only ~1–2% of parameters
- **QLoRA** — LoRA combined with 4-bit quantization, dramatically reducing VRAM requirements
- **Instruction Tuning** — training the model to reliably follow task instructions
- **SFT** (Supervised Fine-Tuning) — the standard supervised training loop that underpins all of the above

---

## 2. Hardware Requirements <a name="hardware"></a>

Your available GPU memory determines which model sizes and training strategies are realistic.

| Setup | VRAM | Model Size | Recommended Method |
|---|---|---|---|
| Laptop / Budget GPU | 8 GB | Up to 7B | QLoRA 4-bit |
| RTX 3090 / 4080 | 16–20 GB | 7B–13B | QLoRA 4-bit |
| RTX 4090 / A100 40GB | 24–40 GB | 13B–34B | QLoRA or LoRA |
| A100 80GB | 80 GB | Up to 70B | LoRA or Full Fine-Tune |
| Multi-GPU cluster | 2× or more | 70B+ | DeepSpeed ZeRO |

The RTX 4090 (24 GB) is the recommended single-GPU starting point — it comfortably handles Mistral-7B or LLaMA-3-8B with QLoRA and leaves room for longer sequences.

---

## 3. Environment Setup <a name="setup"></a>

### Virtual environment

Always work inside an isolated virtual environment to avoid dependency conflicts. Create one with Python's built-in `venv`, then activate it before installing anything.

### Required packages

Install these libraries in your environment. Pin versions for reproducibility in production:

- **torch** (with CUDA) — core deep learning framework
- **transformers** — model loading, tokenizers, training utilities
- **peft** — LoRA and QLoRA adapter support
- **trl** — high-level SFT training loop (SFTTrainer)
- **bitsandbytes** — 4-bit and 8-bit quantization kernels
- **datasets** — dataset loading, processing, and splitting
- **accelerate** — multi-GPU and mixed-precision training support
- **wandb** — experiment tracking (optional but strongly recommended)
- **rouge-score** — automated evaluation metrics
- **pandas, matplotlib, seaborn** — data inspection and visualization
- **sentencepiece, protobuf** — tokenizer dependencies for many models

### Recommended project structure

Organize your project into clearly separated folders from the start:

- `data/raw/` — original source data, never modified directly
- `data/cleaned/` — after deduplication and noise removal
- `data/formatted/` — instruction-formatted, ready for tokenization
- `scripts/` — one script per pipeline stage
- `configs/` — YAML files for hyperparameters and use-case settings
- `outputs/checkpoints/` — model checkpoints saved during training
- `outputs/merged_model/` — final merged model ready for deployment
- `notebooks/` — exploratory analysis and visualization

---

## 4. Stage 1 — Define the Use Case <a name="stage1"></a>

Before collecting a single data point, define what success looks like. Vague goals produce vague models.

Answer these questions before writing any code:

- **Task type** — question answering, summarization, classification, chat, code generation?
- **Input format** — plain text, documents, structured data?
- **Output format** — short answer, long-form prose, structured JSON?
- **Evaluation criteria** — factual accuracy, ROUGE score, human preference rating?
- **Deployment target** — local API, cloud endpoint, embedded device?
- **Base model and hardware** — which model fits your GPU, and which variant suits your task?

### Use case reference table

| Use Case | Recommended Base Model | Dataset Type | Key Consideration |
|---|---|---|---|
| Customer support bot | Mistral-7B | Q&A pairs | Keep answers concise |
| Legal document analysis | LLaMA-3-8B | Contract + analysis | Long context support needed |
| Code assistant | CodeLlama-7B | Code + docstring pairs | Preserve indentation exactly |
| Medical Q&A | Mistral-7B | Question + evidence + answer | Accuracy is non-negotiable |
| Internal knowledge bot | LLaMA-3-8B | Document + Q&A | Consider RAG as an alternative |

---

## 5. Stage 2 — Data Collection <a name="stage2"></a>

Data quality determines model quality more than any hyperparameter. Collect from the best sources available.

### Source types

**Internal data (highest value):** company documentation, support ticket resolutions, email threads, chat logs, internal wikis. This is domain-specific and unique — prioritize it.

**Public datasets:** Hugging Face Hub hosts hundreds of instruction datasets. Well-known ones include Alpaca (52k general instructions), Dolly-15k (diverse open-source tasks), and OpenAssistant (multi-turn conversations).

**Synthetic data:** Use a capable model (GPT-4, Claude) to generate instruction-output pairs from your raw source documents. Prompt the model to produce diverse question-answer pairs for each document, then review the output before using it for training.

### Minimum dataset sizes

More data is generally better, but there are practical minimums below which fine-tuning produces unreliable results:

| Model Size | Minimum for Usable Results | Target for Production |
|---|---|---|
| 3B parameters | 5,000 samples | 20,000+ |
| 7B parameters | 20,000 samples | 50,000–100,000 |
| 13B parameters | 50,000 samples | 100,000–200,000 |
| 70B parameters | 100,000 samples | 500,000+ |

---

## 6. Stage 3 — Data Inspection & EDA <a name="stage3"></a>

Never clean or train on data you haven't looked at. Exploratory Data Analysis (EDA) reveals problems that are invisible until they degrade your model.

### What to measure

- **Dataset size** — total number of rows and columns
- **Null and missing values** — empty instructions or outputs break training
- **Duplicate rows** — exact and near-duplicates inflate apparent dataset size and cause overfitting
- **Token length distribution** — plot histograms for instruction length, input length, and output length; flag samples exceeding your model's max length (typically 2048), as they will be silently truncated
- **Very short outputs** — outputs under 10 tokens are almost always noise
- **Language and encoding issues** — unexpected non-ASCII characters, HTML entities, or broken encodings

### What to look for in visualizations

A healthy dataset shows a roughly bell-shaped or right-skewed token distribution, with most outputs falling well within the max length limit. A spike at exactly the max length is a red flag — it means many samples are being truncated and losing information.

---

## 7. Stage 4 — Data Cleaning <a name="stage4"></a>

Clean data is the single highest-leverage investment in fine-tuning quality. Plan to remove 10–30% of a typical raw dataset.

### Cleaning operations (in order)

**1. Strip HTML and markup** — raw web-scraped text often contains HTML tags, entities, and markdown artifacts that confuse the model.

**2. Normalize whitespace** — collapse multiple spaces, tabs, and newlines into single spaces.

**3. Remove URLs** — unless URLs are meaningful to the task, strip them to reduce noise.

**4. Remove duplicates** — deduplicate on the combination of instruction + output, not just one field. An instruction that appears 50 times with different outputs is fine; an exact instruction-output pair that appears 50 times is waste.

**5. Filter invalid samples** — remove instructions shorter than 5 characters, outputs shorter than 20 characters, samples where the output is identical to the instruction (copy-paste errors), and samples where required fields are null or empty.

**6. Log removal statistics** — always record how many rows were removed at each step so you can spot unexpected data loss.

A clean dataset should shrink by roughly 10–30% from the raw version. If you're removing more than 50%, your data source quality needs attention before you proceed.

---

## 8. Stage 5 — Dataset Formatting <a name="stage5"></a>

LLMs learn to follow instructions through a consistent prompt structure. The model must see the same format during training that it will receive at inference time.

### Format A — Alpaca Instruction Format

Best for single-turn tasks such as Q&A, summarization, and classification. Each sample has three fields: an instruction describing the task, an optional input providing context, and an output containing the expected response. The three fields are separated by labeled section headers. If there is no input context for a given sample, the input section is omitted entirely.

### Format B — Chat / Messages Format

Best for conversational models. Each sample is a list of messages with roles: system, user, and assistant. This format aligns with how chat-tuned models are prompted at inference time and handles multi-turn conversations naturally.

### Model-specific chat templates

Each model family expects a slightly different prompt wrapper. Mistral wraps user turns in `[INST]` and `[/INST]` tags. LLaMA-3 uses special header tokens. Alpaca-style models use plain section headers. Always use the template that matches your chosen base model — mismatched templates are one of the most common causes of poor fine-tuning results.

### Formatting checklist

- [ ] Every sample is wrapped in the correct prompt template for your base model
- [ ] The EOS (end-of-sequence) token is appended to the end of each output
- [ ] The system prompt (if any) is included consistently in every sample
- [ ] You have verified the formatted output looks correct by manually inspecting at least 10 samples

---

## 9. Stage 6 — Tokenization & Train/Val Split <a name="stage6"></a>

### Tokenization settings

- Set `pad_token` to `eos_token` if the model doesn't have a dedicated pad token (common for decoder-only models like Mistral and LLaMA)
- Set `padding_side` to right for causal language models
- Enable truncation with a max length of 2048 (or your chosen context window)
- Use dynamic padding when possible — it is more memory-efficient than padding all sequences to the maximum length

### Train / validation split

Split your dataset before training — never evaluate on training data. The standard split is 90% training and 10% validation, using a fixed random seed for reproducibility. Save the split datasets to disk so training can be resumed from a checkpoint without re-processing.

### Verifying tokenization output

After tokenizing a formatted sample, manually inspect several examples. The token IDs should begin with the beginning-of-sequence token, progress through the formatted instruction and output, and end with the EOS token. Confirm this before launching training — silent tokenization bugs are among the hardest failures to diagnose after the fact.

---

## 10. Stage 7 — Model Selection <a name="stage7"></a>

Choose your base model based on three factors: available VRAM, task type, and language requirements.

### Recommended models (2025)

| Model | Best For | Notes |
|---|---|---|
| Mistral-7B-v0.1 | General instruction following | Strong default choice |
| Mistral-7B-Instruct-v0.2 | Chat and Q&A | Pre-tuned for instructions |
| LLaMA-3-8B-Instruct | Reasoning and chat | Meta's latest open model |
| CodeLlama-7B | Code generation | Trained on code-heavy data |
| Phi-3-mini | Resource-constrained deployment | Very small, surprisingly capable |
| Qwen2-7B | Multilingual tasks | Strong non-English performance |

### Base vs. Instruct variant

Use the Instruct variant of a model if it exists and your task involves instruction-following or chat. The Instruct variant has already been aligned to follow directions — you are fine-tuning it further for your specific domain, which is easier and requires less data.

Use the base variant only when you need full control over output formatting from scratch, or when you are training on raw text rather than instruction pairs.

---

## 11. Stage 8 — LoRA / QLoRA Configuration <a name="stage8"></a>

### Understanding LoRA

LoRA does not modify the original model weights. Instead, it injects small trainable matrices into specific transformer layers. During training, only these adapter matrices are updated — the base model remains frozen. This means you train only ~1–2% of the total parameters, which drastically reduces memory usage and training time.

### Understanding QLoRA

QLoRA adds a quantization step on top of LoRA. The base model is loaded in 4-bit precision (NormalFloat4 quantization), reducing its VRAM footprint by roughly 4×. The LoRA adapters themselves remain in full precision. This combination allows a 7B model to be fine-tuned on a single 16 GB GPU.

### Key LoRA hyperparameters

| Parameter | What It Controls | Typical Value |
|---|---|---|
| r (rank) | Adapter capacity — higher rank means more parameters | 8–32 |
| lora_alpha | Scaling factor applied to adapter outputs | 2× rank |
| target_modules | Which transformer layers receive adapters | All attention + MLP projections |
| lora_dropout | Regularization to reduce overfitting | 0.05 |

### Which layers to target

For Mistral and LLaMA architectures, target all seven projection layers: the four attention projections (q, k, v, o) and all three MLP projections (gate, up, down). Targeting only q and v projections — a common shortcut — leaves most of the model's adaptable capacity unused and produces weaker results.

### QLoRA quantization settings

Use NormalFloat4 (nf4) quantization with double quantization enabled and bfloat16 as the compute dtype. Double quantization adds a second layer of quantization on the quantization constants themselves, saving approximately 0.4 bits per parameter at negligible quality cost.

---

## 12. Stage 9 — Training <a name="stage9"></a>

### Recommended tool: SFTTrainer

`SFTTrainer` from the `trl` library handles prompt formatting, sequence packing, and the training loop correctly for instruction fine-tuning. It is the current industry standard for this workflow and is far less error-prone than a custom training loop.

### Key training hyperparameters

| Hyperparameter | 7B Model Default | Purpose |
|---|---|---|
| learning_rate | 2e-4 | Step size — too high causes instability, too low slows convergence |
| per_device_train_batch_size | 2 | Samples per GPU per step |
| gradient_accumulation_steps | 8 | Multiplied with batch size to get effective batch size (2×8=16) |
| num_train_epochs | 3 | Full passes through the dataset |
| lr_scheduler_type | cosine | Decays the learning rate smoothly |
| warmup_ratio | 0.05 | Fraction of steps for gradual LR warmup |
| weight_decay | 0.001 | L2 regularization |
| bf16 | True | Mixed precision on Ampere+ GPUs |
| gradient_checkpointing | True | Trades compute for memory savings |

### Effective batch size

Effective batch size equals `per_device_train_batch_size × gradient_accumulation_steps × number of GPUs`. For most 7B fine-tuning tasks, an effective batch size of 16–32 works well. If VRAM is limited, keep the per-device batch size at 1–2 and increase gradient accumulation steps to compensate.

### Training monitoring checklist

- Training loss should decrease steadily — a flat or oscillating loss usually indicates the learning rate is wrong or the data format is broken
- Eval loss should track training loss — if it rises while training loss falls, the model is overfitting
- GPU utilization should stay above 80% — lower utilization means data loading is the bottleneck
- Gradient norms should remain stable (typically below 1.0) — large spikes indicate instability
- Watch for OOM errors — if they occur, reduce batch size or verify gradient checkpointing is enabled

---

## 13. Stage 10 — Saving & Merging <a name="stage10"></a>

### What gets saved

After training, you have two artifacts:

**LoRA adapter only** (50–200 MB): Contains only the adapter weights. Requires the original base model to run. Save this for quick iteration — it is small enough to version-control and share.

**Merged model** (full model size, typically 14–28 GB for a 7B model): Combines base model weights and adapter weights into a single standalone model file. Required for most deployment tools including vLLM and Ollama.

### The merging process

Load the base model in bfloat16 precision and call `merge_and_unload()` on the PEFT model to produce a standard HuggingFace model with adapter weights permanently baked in. Save it using `safe_serialization=True` to produce `.safetensors` files rather than the older pickle-based format.

Always save the tokenizer alongside the model — the tokenizer configuration, special tokens, and chat template are all required for correct inference.

---

## 14. Stage 11 — Evaluation <a name="stage11"></a>

Evaluation is where most projects underinvest. Use both automatic metrics and human judgment — neither alone is sufficient.

### Automatic metrics

**Perplexity** measures how confidently the model predicts the next token across the validation set. Lower is better. Calculate it as `exp(eval_loss)` after training. A well-tuned model typically scores 20–50; a perplexity above 100 suggests something is wrong with the data or training setup.

**ROUGE scores** measure overlap between generated text and reference answers. Most useful for summarization and long-form Q&A tasks. ROUGE-L (longest common subsequence) is the most informative variant.

**BLEU** is primarily used for translation tasks. It measures n-gram precision against reference translations.

### Human evaluation

Automatic metrics capture surface-level quality but miss reasoning quality, factual accuracy, and tone. Human evaluation must be part of your pipeline.

Create 50–100 test prompts representing real usage. For each, generate a response and score it on these four dimensions on a 1–5 scale:

| Dimension | What It Measures |
|---|---|
| Accuracy | Is the answer factually correct? |
| Relevance | Does it address the question asked? |
| Fluency | Is it well-written and natural sounding? |
| Completeness | Is the answer sufficiently complete? |

Average these scores across your test set and track them across training runs to measure whether each iteration actually improves the model.

### LLM-as-judge

For comparing many model variants quickly, prompt a capable model (GPT-4, Claude) to score outputs against reference answers and return structured scores with reasoning. This is a useful complement to human evaluation, not a replacement for it.

---

## 15. Stage 12 — Iteration Loop <a name="stage12"></a>

Fine-tuning is rarely successful on the first run. Plan for 3–5 iteration cycles.

**Train → Evaluate → Diagnose → Fix → Retrain**

### Diagnosing common failure modes

| Symptom | Most Likely Cause | Fix |
|---|---|---|
| Loss not decreasing after epoch 1 | LR too low, or data format is broken | Increase LR by 2–5×; manually inspect 20 formatted samples |
| Eval loss diverges upward | Overfitting | Reduce number of epochs; add more diverse data |
| Model repeats phrases or loops | Repetitive text in dataset | Deduplicate more aggressively; add repetition penalty at inference |
| Output format doesn't match expectation | Prompt template mismatch | Verify training and inference templates match exactly |
| Model hallucinates facts | Too little domain-grounding data | Add more factual examples; add retrieved context to inputs |
| OOM crash during training | Batch size too large | Reduce per-device batch size to 1; increase gradient accumulation |
| GPU utilization below 60% | Data loading bottleneck | Pre-tokenize and save datasets to disk; increase dataloader workers |

### Data improvement checklist

After each evaluation round, improve the dataset before retraining:

- [ ] Remove examples where the model consistently fails (they may be noisy or contradictory)
- [ ] Add more examples of the failure types identified in evaluation
- [ ] Add more diversity in instruction phrasing by paraphrasing existing instructions
- [ ] Verify output length distribution matches what you want at inference time
- [ ] Add explicit examples of correct format if the model is producing incorrect structures

---

## 16. Stage 13 — Deployment <a name="stage13"></a>

### Option A — vLLM (recommended for production)

vLLM is the fastest open-source inference engine for LLMs. It implements PagedAttention for high throughput and exposes an OpenAI-compatible REST API out of the box. Use it when you need to serve multiple concurrent users or require high tokens-per-second throughput. Key configuration decisions: use bfloat16 as the dtype, set max model length to match your training context window, and adjust tensor parallel size if running across multiple GPUs.

### Option B — Ollama (recommended for local / personal use)

Ollama is the simplest way to run a fine-tuned model locally. It requires converting the merged model to GGUF format using llama.cpp's conversion script, then quantizing it — Q4_K_M offers the best quality-to-size tradeoff for most use cases. A simple Modelfile defines the model parameters and system prompt. Once created, the model is available via a single `ollama run` command.

### Option C — FastAPI (recommended for custom integrations)

When you need custom request handling, authentication, preprocessing logic, or integration with other services, wrap the HuggingFace pipeline in a FastAPI application. This provides full control over the serving logic and is straightforward to containerize with Docker for cloud deployment.

### Production monitoring

Track these metrics from the first day of deployment:

- **Latency** — track p50, p95, and p99 response times; alert if p95 exceeds your SLA
- **Throughput** — tokens per second; watch for degradation under load
- **Error rate** — failed or incomplete generations as a percentage of total requests
- **Input length distribution** — if users send much longer inputs than your training data, context overflow is likely
- **Output quality sampling** — manually review 10–20 live outputs per week to catch quality drift early

Set a retraining trigger: if your quality score on a held-out test set drops more than 5% from your baseline, begin a new training run with recent production examples added to the dataset.

---

## 17. How to Give This as a Skill to AI <a name="skill"></a>

A **skill** is a structured markdown file (`SKILL.md`) that permanently teaches an AI assistant how to perform a complex task. Instead of re-explaining the fine-tuning pipeline every time you start a new project or conversation, you install the skill once and the AI follows it automatically whenever the topic comes up.

### How a skill file works

A skill file has two sections separated by a YAML frontmatter block.

The **frontmatter** (between the `---` lines) contains the skill's name and a description that tells the AI when to use it. This is the trigger — the AI reads the description and decides whether the current task calls for this skill. The description must be specific and include the exact keywords users are likely to say, such as "fine-tune", "LoRA", "QLoRA", "training data", "instruction dataset", and "custom model".

The **body** is everything below the frontmatter — the actual step-by-step instructions the AI follows, in the same format as this guide.

### Skill folder structure

The skill lives in a folder that can optionally include supporting reference files:

- `llm-finetuning/SKILL.md` — required: the trigger description and all instructions
- `llm-finetuning/references/lora_theory.md` — optional: deeper explanation of LoRA math
- `llm-finetuning/references/evaluation_guide.md` — optional: extended evaluation methodology

The AI loads `SKILL.md` automatically when the skill triggers and only reads reference files when it needs additional detail for a specific sub-task.

### Installing the skill in Claude.ai

1. Go to Claude.ai → Settings → Skills
2. Click "Add Skill" or "Upload Skill"
3. Upload the `SKILL.md` file (or the packaged `.skill` file if you have one)
4. The skill is now permanently active for your account

### Triggering the skill

Once installed, natural language is sufficient. The AI will automatically apply the pipeline when you say things like:

- "I want to fine-tune Mistral-7B on my support tickets"
- "Help me set up QLoRA training on my dataset"
- "How do I format my data for instruction tuning?"
- "My fine-tuned model is hallucinating — what do I check?"
- "Help me evaluate my fine-tuned model before deployment"

### Adding a system prompt for stricter enforcement

If you want the AI to always follow the skill strictly and never skip stages, pair the skill with a system prompt in Claude.ai under Settings → Customize:

> You are an ML engineering assistant specializing in LLM fine-tuning. Whenever a user asks about model training, fine-tuning, LoRA, QLoRA, or dataset preparation, follow the llm-finetuning skill exactly — all 13 stages in order. Before recommending a training strategy, always ask for the user's available GPU VRAM. Never skip the data inspection or evaluation stages.

### Keeping the skill current

A skill is a living document. As you accumulate project-specific knowledge, add it to the skill so future sessions benefit automatically. Open `SKILL.md` in any text editor and append a section with your team's specific defaults: your standard base model, your dataset storage paths, your ROUGE threshold for deployment approval, and any domain-specific quirks discovered in practice. The skill becomes progressively more valuable over time without requiring you to re-explain context in every conversation.

---

## 18. Quick Reference Card <a name="reference"></a>

### Hardware Selection

| VRAM | Training Method | Max Model Size |
|---|---|---|
| 8 GB | QLoRA 4-bit | 7B |
| 16 GB | QLoRA 4-bit | 13B |
| 24 GB | QLoRA or LoRA | 13–34B |
| 80 GB | LoRA or Full FT | 70B |

### Default Hyperparameters — 7B Model

| Setting | Value |
|---|---|
| Learning rate | 2e-4 |
| Effective batch size | 16 (2 per device × 8 grad accum) |
| Epochs | 3 |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Max sequence length | 2048 |
| LR scheduler | cosine |
| Warmup ratio | 5% |

### Pipeline at a Glance

Stage 1 — Define use case and success criteria  
Stage 2 — Collect data from internal, public, or synthetic sources  
Stage 3 — Inspect and visualize the raw dataset (EDA)  
Stage 4 — Clean: remove noise, duplicates, and invalid samples  
Stage 5 — Format into instruction or chat template  
Stage 6 — Tokenize and split into train / validation sets  
Stage 7 — Select base model based on task and hardware  
Stage 8 — Configure LoRA adapters and QLoRA quantization  
Stage 9 — Train with SFTTrainer and monitor loss curves  
Stage 10 — Save the adapter, then merge into a full model  
Stage 11 — Evaluate with perplexity, ROUGE, and human scoring  
Stage 12 — Diagnose failures, improve data, retrain  
Stage 13 — Deploy with vLLM, Ollama, or FastAPI; monitor in production  

### Required Packages

torch · transformers · peft · trl · bitsandbytes · datasets · accelerate · wandb · rouge-score · pandas · matplotlib · sentencepiece

---

*Guide version 2.0 — Updated for 2025 tooling (TRL SFTTrainer, LLaMA-3, Mistral)*
