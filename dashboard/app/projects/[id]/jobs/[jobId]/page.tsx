"use client"

import { useState, useEffect, useCallback } from "react"
import { useParams } from "next/navigation"
import { useQueryClient } from "@tanstack/react-query"
import { useJob, useCheckpoints } from "@/lib/hooks/useJobs"
import { useJobStream, type ProgressPoint } from "@/lib/hooks/useJobStream"
import { LossChart } from "@/components/charts/LossChart"
import { LogConsole } from "@/components/jobs/LogConsole"
import { GPUWidget } from "@/components/jobs/GPUWidget"
import { api } from "@/lib/api"

const BASE = process.env.NEXT_PUBLIC_API_URL

const STATUS_STYLE: Record<string, string> = {
  queued:    "bg-zinc-800 text-zinc-400",
  running:   "bg-blue-900 text-blue-300 animate-pulse",
  completed: "bg-green-900 text-green-300",
  failed:    "bg-red-900 text-red-300",
  cancelled: "bg-zinc-800 text-zinc-500",
}

const METHOD_STYLE: Record<string, string> = {
  sft:  "bg-zinc-800 text-zinc-300",
  dpo:  "bg-purple-900 text-purple-300",
  orpo: "bg-blue-900 text-blue-300",
  fft:  "bg-amber-900 text-amber-300",
}

// ── Push to Hub modal ─────────────────────────────────────────────────────────

function PushToHubModal({ jobId, onClose, onDone }: { jobId: string; onClose: () => void; onDone: (url: string) => void }) {
  const [repoId, setRepoId] = useState("")
  const [isPrivate, setIsPrivate] = useState(true)
  const [message, setMessage] = useState("Upload fine-tuned model")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handlePush() {
    if (!repoId.trim()) return
    setBusy(true)
    setError(null)
    try {
      const result = await api.jobs.pushToHub(jobId, {
        repo_id: repoId.trim(),
        private: isPrivate,
        commit_message: message,
      })
      onDone(result.url)
    } catch (e: unknown) {
      setError((e as Error).message)
      setBusy(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4">
      <div className="w-full max-w-md bg-zinc-950 border border-zinc-700 rounded-xl p-5 shadow-2xl">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-bold text-sm">Push to HuggingFace Hub</h3>
          <button onClick={onClose} className="text-zinc-500 hover:text-white">×</button>
        </div>

        <div className="space-y-3">
          <div>
            <label className="text-zinc-500 text-xs block mb-1">Repository ID</label>
            <input
              value={repoId}
              onChange={e => setRepoId(e.target.value)}
              placeholder="username/my-fine-tuned-model"
              className="input w-full"
            />
          </div>
          <div>
            <label className="text-zinc-500 text-xs block mb-1">Commit message</label>
            <input
              value={message}
              onChange={e => setMessage(e.target.value)}
              className="input w-full"
            />
          </div>
          <label className="flex items-center gap-2 cursor-pointer text-sm">
            <input type="checkbox" checked={isPrivate} onChange={e => setIsPrivate(e.target.checked)} className="accent-white" />
            <span>Private repository</span>
          </label>

          {error && <p className="text-red-400 text-xs">{error}</p>}

          <div className="flex gap-2 pt-1">
            <button onClick={onClose} className="px-4 py-2 border border-zinc-700 text-sm rounded hover:bg-zinc-800 flex-1">
              Cancel
            </button>
            <button
              onClick={handlePush}
              disabled={busy || !repoId.trim()}
              className="px-4 py-2 bg-white text-black text-sm font-medium rounded hover:bg-zinc-200 disabled:opacity-50 flex-1"
            >
              {busy ? "Uploading…" : "Push"}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function JobMonitorPage() {
  const { id, jobId } = useParams<{ id: string; jobId: string }>()
  const qc = useQueryClient()
  const { data: job, refetch: refetchJob } = useJob(jobId)
  const { data: ckptData, refetch: refetchCkpts } = useCheckpoints(jobId)

  const [livePoints, setLivePoints] = useState<ProgressPoint[]>([])
  const [logs, setLogs] = useState<string[]>([])
  const [streamDone, setStreamDone] = useState(false)

  // Action states
  const [vllmCmd, setVllmCmd] = useState<string | null>(null)
  const [cardContent, setCardContent] = useState<string | null>(null)
  const [showCard, setShowCard] = useState(false)
  const [showPushModal, setShowPushModal] = useState(false)
  const [hubUrl, setHubUrl] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)

  // Seed chart from persisted checkpoints
  useEffect(() => {
    const ckpts = ckptData?.checkpoints ?? []
    if (ckpts.length && livePoints.length === 0) {
      setLivePoints(ckpts.map(c => ({
        epoch: c.epoch,
        train_loss: c.train_loss,
        eval_loss: c.eval_loss,
        perplexity: c.perplexity,
      })))
    }
  }, [ckptData])

  // Pre-load persisted vllm cmd and card from job
  useEffect(() => {
    if (job?.vllm_launch_cmd) setVllmCmd(job.vllm_launch_cmd)
    if (job?.hf_repo_id) setHubUrl(`https://huggingface.co/${job.hf_repo_id}`)
  }, [job?.vllm_launch_cmd, job?.hf_repo_id])

  const handleProgress = useCallback((p: ProgressPoint) => {
    setLivePoints(prev => {
      if (prev.find(x => x.epoch === p.epoch)) return prev
      return [...prev, p]
    })
  }, [])

  const handleLog = useCallback((line: string) => {
    setLogs(prev => [...prev.slice(-800), line])
  }, [])

  const handleDone = useCallback((_status: string) => {
    setStreamDone(true)
    refetchJob()
    refetchCkpts()
    qc.invalidateQueries({ queryKey: ["jobs", id] })
  }, [refetchJob, refetchCkpts, qc, id])

  const isActive = job?.status === "running" || job?.status === "queued"

  useJobStream(jobId, {
    enabled: isActive,
    onProgress: handleProgress,
    onLog: handleLog,
    onDone: handleDone,
  })

  async function cancel() {
    await fetch(`${BASE}/jobs/${jobId}/cancel`, { method: "DELETE" })
    refetchJob()
  }
  async function merge() {
    await fetch(`${BASE}/jobs/${jobId}/merge`, { method: "POST" })
    refetchJob()
  }
  async function exportToOllama() {
    await fetch(`${BASE}/jobs/${jobId}/export`, { method: "POST" })
    refetchJob()
  }
  async function exportVllm() {
    setActionError(null)
    try {
      const res = await api.jobs.exportVllm(jobId)
      setVllmCmd(res.launch_command)
    } catch (e: unknown) {
      setActionError((e as Error).message)
    }
  }
  async function generateCard() {
    setActionError(null)
    try {
      const res = await api.jobs.generateModelCard(jobId)
      setCardContent(res.content)
      setShowCard(true)
    } catch (e: unknown) {
      setActionError((e as Error).message)
    }
  }
  async function loadCard() {
    if (cardContent) { setShowCard(v => !v); return }
    try {
      const res = await api.jobs.getModelCard(jobId)
      setCardContent(res.content)
      setShowCard(true)
    } catch {
      setShowCard(false)
    }
  }

  if (!job) return <div className="text-zinc-500 text-sm p-4">Loading...</div>

  const chartData = livePoints.length
    ? livePoints
    : (ckptData?.checkpoints ?? []).map(c => ({
        epoch: c.epoch,
        train_loss: c.train_loss,
        eval_loss: c.eval_loss,
        perplexity: c.perplexity,
      }))

  const latestCkpt = chartData[chartData.length - 1]

  return (
    <div className="max-w-6xl mx-auto">

      {/* ── Header ── */}
      <div className="flex items-start justify-between mb-5">
        <div>
          <a href={`/projects/${id}/jobs`} className="text-zinc-600 text-xs hover:text-zinc-400">← Jobs</a>
          <h2 className="text-lg font-bold mt-1">{job.name}</h2>
          <div className="flex items-center gap-2 mt-1 flex-wrap">
            <span className={`px-2 py-0.5 rounded text-xs font-mono ${STATUS_STYLE[job.status] ?? ""}`}>
              {job.status}
            </span>
            <span className={`px-2 py-0.5 rounded text-xs uppercase font-mono ${METHOD_STYLE[job.training_method] ?? ""}`}>
              {job.training_method}
            </span>
            <span className="text-zinc-500 text-xs">{job.base_model}</span>
            {!job.is_full_model && (
              <span className="text-zinc-600 text-xs">r={job.lora_r} · alpha={job.lora_alpha}</span>
            )}
            <span className="text-zinc-600 text-xs">{job.epochs} epochs · lr={job.learning_rate}</span>
            {job.use_qlora && !job.is_full_model && <span className="text-zinc-600 text-xs">QLoRA</span>}
            {job.use_unsloth && <span className="text-zinc-600 text-xs">Unsloth</span>}
            {job.is_full_model && <span className="text-amber-700 text-xs">Full Model</span>}
          </div>
        </div>

        <div className="flex gap-2 flex-wrap justify-end">
          {isActive && (
            <button onClick={cancel}
              className="px-3 py-1.5 border border-red-800 text-red-400 text-xs rounded hover:bg-red-950">
              Cancel
            </button>
          )}
          {job.status === "completed" && !job.merged_path && !job.is_full_model && (
            <button onClick={merge}
              className="px-3 py-1.5 border border-zinc-700 text-zinc-300 text-xs rounded hover:bg-zinc-800">
              Merge Model
            </button>
          )}
          {job.merged_path && !job.ollama_model_name && (
            <button onClick={exportToOllama}
              className="px-3 py-1.5 border border-zinc-700 text-zinc-300 text-xs rounded hover:bg-zinc-800">
              Export to Ollama
            </button>
          )}
          {job.ollama_model_name && (
            <span className="px-3 py-1.5 border border-green-800 text-green-400 text-xs rounded font-mono">
              {job.ollama_model_name}
            </span>
          )}
        </div>
      </div>

      {/* ── Stat strip ── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-5">
        {[
          ["Effective Batch", `${job.batch_size} × ${job.grad_accum} = ${job.batch_size * job.grad_accum}`],
          ["Max Seq Len",     job.max_seq_len],
          ["Latest Train Loss", latestCkpt ? latestCkpt.train_loss.toFixed(4) : "—"],
          ["Latest Perplexity", latestCkpt ? latestCkpt.perplexity.toFixed(2) : "—"],
        ].map(([label, val]) => (
          <div key={String(label)} className="p-3 border border-zinc-800 rounded bg-zinc-900">
            <p className="text-zinc-600 text-xs">{label}</p>
            <p className="text-white font-mono text-sm mt-0.5">{String(val)}</p>
          </div>
        ))}
      </div>

      {/* ── Main layout: chart + GPU widget ── */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_200px] gap-4 mb-4">
        <div className="p-4 border border-zinc-800 rounded bg-zinc-900">
          <div className="flex items-center justify-between mb-3">
            <p className="text-zinc-400 text-xs font-mono">loss curves</p>
            {isActive && chartData.length > 0 && (
              <p className="text-zinc-600 text-xs">epoch {chartData.length} / {job.epochs}</p>
            )}
          </div>
          <LossChart data={chartData} />
        </div>
        <GPUWidget jobId={jobId} active={isActive} />
      </div>

      {/* ── Log console ── */}
      <LogConsole lines={logs} isLive={isActive} maxHeight="h-72" />

      {/* ── Error ── */}
      {job.error_message && (
        <div className="mt-4 p-3 border border-red-800 rounded bg-red-950 text-red-300 text-xs">
          <p className="font-bold mb-1">Error</p>
          <pre className="whitespace-pre-wrap overflow-auto max-h-48">{job.error_message}</pre>
        </div>
      )}

      {/* ── Completed: next steps ── */}
      {job.status === "completed" && (
        <div className="mt-5 p-4 border border-green-800 rounded bg-green-950">
          <p className="text-green-300 font-medium text-sm mb-3">Training complete</p>
          <div className="flex gap-3 flex-wrap">
            {!job.merged_path && !job.is_full_model && (
              <button onClick={merge}
                className="px-4 py-2 bg-white text-black text-sm font-medium rounded hover:bg-zinc-200">
                Merge Model
              </button>
            )}
            <a href={`/projects/${id}/evaluate`}
              className="px-4 py-2 border border-zinc-600 text-sm rounded hover:bg-zinc-800">
              Evaluate →
            </a>
            <a href={`/projects/${id}/inference`}
              className="px-4 py-2 border border-zinc-600 text-sm rounded hover:bg-zinc-800">
              Test Inference →
            </a>
            <a href={`/projects/${id}/jobs/new?clone=${jobId}`}
              className="px-4 py-2 border border-zinc-600 text-sm rounded hover:bg-zinc-800">
              Clone with Modifications →
            </a>
          </div>
        </div>
      )}

      {/* ── Actions panel (completed jobs) ── */}
      {job.status === "completed" && (
        <div className="mt-5 p-4 border border-zinc-800 rounded bg-zinc-900">
          <p className="text-zinc-400 text-xs font-medium uppercase tracking-wider mb-4">Actions</p>

          {actionError && (
            <div className="mb-3 p-2 border border-red-800 rounded bg-red-950 text-red-300 text-xs">{actionError}</div>
          )}

          <div className="flex flex-wrap gap-2">
            {/* Download Config */}
            <a
              href={api.jobs.configUrl(jobId)}
              download
              className="px-3 py-2 border border-zinc-700 text-zinc-300 text-xs rounded hover:bg-zinc-800"
            >
              Download Config (YAML)
            </a>

            {/* Export vLLM */}
            {(job.merged_path || job.is_full_model) && (
              <button
                onClick={exportVllm}
                className="px-3 py-2 border border-zinc-700 text-zinc-300 text-xs rounded hover:bg-zinc-800"
              >
                {vllmCmd ? "Re-generate vLLM Command" : "Export vLLM Launch Command"}
              </button>
            )}

            {/* Generate Model Card */}
            {job.adapter_path && (
              <button
                onClick={generateCard}
                className="px-3 py-2 border border-zinc-700 text-zinc-300 text-xs rounded hover:bg-zinc-800"
              >
                Generate Model Card
              </button>
            )}
            {job.model_card_path && !cardContent && (
              <button
                onClick={loadCard}
                className="px-3 py-2 border border-zinc-700 text-zinc-300 text-xs rounded hover:bg-zinc-800"
              >
                {showCard ? "Hide Model Card" : "View Model Card"}
              </button>
            )}

            {/* Push to Hub */}
            <button
              onClick={() => { setActionError(null); setShowPushModal(true) }}
              className="px-3 py-2 border border-zinc-700 text-zinc-300 text-xs rounded hover:bg-zinc-800"
            >
              Push to HuggingFace Hub
            </button>
          </div>

          {/* vLLM launch command */}
          {vllmCmd && (
            <div className="mt-4">
              <p className="text-zinc-500 text-xs mb-2">vLLM launch command:</p>
              <pre className="p-3 bg-zinc-950 border border-zinc-800 rounded text-xs text-green-400 overflow-x-auto whitespace-pre">{vllmCmd}</pre>
            </div>
          )}

          {/* Hub URL */}
          {hubUrl && (
            <div className="mt-3 p-3 border border-green-900 rounded bg-green-950 text-green-300 text-xs">
              Pushed to Hub:{" "}
              <a href={hubUrl} target="_blank" rel="noopener noreferrer" className="underline">{hubUrl}</a>
            </div>
          )}

          {/* Model card */}
          {cardContent && showCard && (
            <div className="mt-4">
              <div className="flex items-center justify-between mb-2">
                <p className="text-zinc-500 text-xs">MODEL_CARD.md</p>
                <button onClick={() => setShowCard(false)} className="text-zinc-600 text-xs hover:text-zinc-400">hide</button>
              </div>
              <pre className="p-3 bg-zinc-950 border border-zinc-800 rounded text-xs text-zinc-300 overflow-auto max-h-80 whitespace-pre-wrap">{cardContent}</pre>
            </div>
          )}

          {/* Webhook */}
          {job.webhook_url && (
            <div className="mt-3 flex items-center gap-2 text-xs text-zinc-600">
              <span>Webhook:</span>
              <span className="font-mono text-zinc-500 truncate max-w-xs">{job.webhook_url}</span>
            </div>
          )}
        </div>
      )}

      {job.status === "failed" && (
        <div className="mt-5 flex gap-3">
          <a href={`/projects/${id}/jobs/new`}
            className="px-4 py-2 border border-zinc-700 text-sm rounded hover:bg-zinc-800">
            New Job →
          </a>
        </div>
      )}

      {showPushModal && (
        <PushToHubModal
          jobId={jobId}
          onClose={() => setShowPushModal(false)}
          onDone={(url) => {
            setHubUrl(url)
            setShowPushModal(false)
            refetchJob()
          }}
        />
      )}
    </div>
  )
}
