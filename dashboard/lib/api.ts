import type { Project, Dataset, TrainingJob, Checkpoint, Evaluation, EvalRow, OllamaModel, HFModel, RAGCollection, RAGDocument, HealthStatus } from "./types"

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8010"

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? "Request failed")
  }
  return res.json() as Promise<T>
}

// ── Health ──────────────────────────────────────────────────
export const api = {
  health: () => apiFetch<HealthStatus>("/health"),

  // ── Projects ──────────────────────────────────────────────
  projects: {
    list: () => apiFetch<{ projects: Project[] }>("/projects"),
    get: (id: string) => apiFetch<Project>(`/projects/${id}`),
    create: (body: { name: string; description?: string }) =>
      apiFetch<Project>("/projects", { method: "POST", body: JSON.stringify(body) }),
    update: (id: string, body: { name?: string; description?: string }) =>
      apiFetch<Project>(`/projects/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
    delete: (id: string) =>
      apiFetch<void>(`/projects/${id}`, { method: "DELETE" }),
  },

  // ── Models ────────────────────────────────────────────────
  models: {
    ollama: () => apiFetch<{ models: OllamaModel[] }>("/models/ollama"),
    hf: () => apiFetch<{ models: HFModel[] }>("/models/hf"),
  },

  // ── Datasets (Phase 2) ────────────────────────────────────
  datasets: {
    list: (projectId: string) =>
      apiFetch<{ datasets: Dataset[] }>(`/projects/${projectId}/datasets`),
    get: (id: string) => apiFetch<Dataset>(`/datasets/${id}`),
    inspect: (id: string) => apiFetch<{ task_id: string }>(`/datasets/${id}/inspect`, { method: "POST" }),
    clean: (id: string, config: Record<string, boolean>) =>
      apiFetch<{ task_id: string }>(`/datasets/${id}/clean`, { method: "POST", body: JSON.stringify(config) }),
    formatPreview: (id: string) =>
      apiFetch<{ samples: string[] }>(`/datasets/${id}/format/preview`),
    format: (id: string, body: { format_type: string; base_model: string }) =>
      apiFetch<{ task_id: string }>(`/datasets/${id}/format`, { method: "POST", body: JSON.stringify(body) }),
    tokenize: (id: string, body: { max_seq_len: number; val_split: number }) =>
      apiFetch<{ task_id: string }>(`/datasets/${id}/tokenize`, { method: "POST", body: JSON.stringify(body) }),
    delete: (id: string) =>
      apiFetch<void>(`/datasets/${id}`, { method: "DELETE" }),
    fromPdf: (projectId: string, form: FormData) =>
      apiFetch<{ dataset_id: string; task_id: string }>(`/projects/${projectId}/datasets/from-pdf`, {
        method: "POST",
        body: form,
        headers: {},   // let browser set Content-Type with boundary
      }),
    fromUrl: (projectId: string, body: { url: string; name: string; ollama_model: string; pairs_per_chunk: number }) =>
      apiFetch<{ dataset_id: string; task_id: string }>(`/projects/${projectId}/datasets/from-url`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
  },

  // ── Jobs (Phase 3) ────────────────────────────────────────
  jobs: {
    list: (projectId: string) =>
      apiFetch<{ jobs: TrainingJob[] }>(`/projects/${projectId}/jobs`),
    get: (id: string) => apiFetch<TrainingJob>(`/jobs/${id}`),
    create: (projectId: string, body: Partial<TrainingJob>) =>
      apiFetch<TrainingJob>(`/projects/${projectId}/jobs`, { method: "POST", body: JSON.stringify(body) }),
    delete: (id: string) =>
      apiFetch<void>(`/jobs/${id}`, { method: "DELETE" }),
    cancel: (id: string) =>
      apiFetch<void>(`/jobs/${id}/cancel`, { method: "DELETE" }),
    checkpoints: (id: string) =>
      apiFetch<{ checkpoints: Checkpoint[] }>(`/jobs/${id}/checkpoints`),
    merge: (id: string) =>
      apiFetch<{ task_id: string }>(`/jobs/${id}/merge`, { method: "POST" }),
    export: (id: string) =>
      apiFetch<{ task_id: string }>(`/jobs/${id}/export`, { method: "POST" }),
  },

  // ── Evaluations (Phase 5) ─────────────────────────────────
  evaluations: {
    get: (jobId: string) => apiFetch<Evaluation>(`/jobs/${jobId}/evaluation`),
    runAuto: (jobId: string) =>
      apiFetch<{ task_id: string }>(`/jobs/${jobId}/evaluation/auto`, { method: "POST" }),
    submitHuman: (jobId: string, results: Evaluation["sample_results"]) =>
      apiFetch<Evaluation>(`/jobs/${jobId}/evaluation/human`, { method: "POST", body: JSON.stringify({ results }) }),
    listForProject: (projectId: string) =>
      apiFetch<{ evaluations: EvalRow[] }>(`/projects/${projectId}/evaluations`),
  },

  // ── RAG (Phase 7) ─────────────────────────────────────────
  rag: {
    listCollections: (projectId: string) =>
      apiFetch<{ collections: RAGCollection[] }>(`/projects/${projectId}/rag/collections`),
    createCollection: (projectId: string, body: { name: string; embedding_model?: string }) =>
      apiFetch<RAGCollection>(`/projects/${projectId}/rag/collections`, { method: "POST", body: JSON.stringify(body) }),
    deleteCollection: (colId: string) =>
      apiFetch<{ ok: boolean }>(`/rag/collections/${colId}`, { method: "DELETE" }),
    listDocuments: (colId: string) =>
      apiFetch<{ documents: RAGDocument[] }>(`/rag/collections/${colId}/documents`),
    uploadDocument: (colId: string, form: FormData) =>
      apiFetch<RAGDocument>(`/rag/collections/${colId}/documents`, { method: "POST", body: form, headers: {} }),
    deleteDocument: (docId: string) =>
      apiFetch<{ ok: boolean }>(`/rag/documents/${docId}`, { method: "DELETE" }),
  },
}
