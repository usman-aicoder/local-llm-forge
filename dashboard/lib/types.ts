// Mirrors backend Pydantic schemas

export interface Project {
  id: string
  name: string
  description: string | null
  created_at: string
  updated_at: string
}

export interface DatasetStats {
  total_rows: number
  null_count: number
  duplicate_count: number
  removed_count: number
  avg_instruction_tokens: number
  avg_output_tokens: number
  p95_total_tokens: number
  flagged_too_long: number
  flagged_too_short: number
  token_histogram: { buckets: number[]; counts: number[] }
  cleaning_report: Record<string, number>
}

export interface Dataset {
  id: string
  project_id: string
  name: string
  source_type: "upload" | "huggingface" | "synthetic" | "pdf" | "web"
  source_url: string | null
  file_path: string
  row_count: number | null
  status: "generating" | "uploaded" | "inspected" | "cleaned" | "formatted" | "tokenized" | "failed"
  format_type: "alpaca" | "chat" | null
  base_model_name: string | null
  generation_model: string | null
  stats: DatasetStats | null
  created_at: string
  updated_at: string
}

export interface TrainingJob {
  id: string
  project_id: string
  dataset_id: string
  name: string
  base_model: string
  model_path: string
  training_method: "sft" | "dpo" | "orpo"
  use_qlora: boolean
  lora_r: number
  lora_alpha: number
  lora_dropout: number
  target_modules: string[]
  learning_rate: number
  epochs: number
  batch_size: number
  grad_accum: number
  max_seq_len: number
  bf16: boolean
  status: "queued" | "running" | "completed" | "failed" | "cancelled"
  celery_task_id: string | null
  error_message: string | null
  adapter_path: string | null
  merged_path: string | null
  gguf_path: string | null
  ollama_model_name: string | null
  started_at: string | null
  completed_at: string | null
  created_at: string
}

export interface Checkpoint {
  id: string
  job_id: string
  epoch: number
  step: number
  train_loss: number
  eval_loss: number
  perplexity: number
  file_path: string
  created_at: string
}

export interface Evaluation {
  id: string
  job_id: string
  rouge_l: number | null
  rouge_1: number | null
  rouge_2: number | null
  bleu: number | null
  perplexity: number | null
  human_avg_score: number | null
  sample_results: Array<{
    prompt: string
    response: string
    ground_truth?: string
    accuracy: number
    relevance: number
    fluency: number
    completeness: number
  }>
  created_at: string
}

export interface OllamaModel {
  name: string
  size: number
  modified_at: string
  digest: string
}

export interface HFModel {
  name: string
  path: string
  size_gb: number
}

export interface RAGCollection {
  id: string
  project_id: string
  name: string
  embedding_model: string
  qdrant_collection: string
  document_count: number
  created_at: string
}

export interface RAGDocument {
  id: string
  collection_id: string
  filename: string
  file_path: string
  chunk_count: number | null
  status: "uploaded" | "processing" | "indexed" | "failed"
  created_at: string
}

export interface EvalRow {
  job_id: string
  job_name: string
  base_model: string
  epochs: number
  rouge_l: number | null
  rouge_1: number | null
  rouge_2: number | null
  bleu: number | null
  perplexity: number | null
  human_avg_score: number | null
  created_at: string | null
}

export interface HealthStatus {
  status: string
  mongo: string
  redis: string
  ollama: string
}
