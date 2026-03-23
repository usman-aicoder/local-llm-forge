"use client"

import { useState, useEffect, useCallback } from "react"
import { useParams } from "next/navigation"
import { useQueryClient } from "@tanstack/react-query"
import { useJob, useCheckpoints } from "@/lib/hooks/useJobs"
import { useJobStream, type ProgressPoint } from "@/lib/hooks/useJobStream"
import { LossChart } from "@/components/charts/LossChart"
import { LogConsole } from "@/components/jobs/LogConsole"
import { GPUWidget } from "@/components/jobs/GPUWidget"

const BASE = process.env.NEXT_PUBLIC_API_URL

const STATUS_STYLE: Record<string, string> = {
  queued:    "bg-zinc-800 text-zinc-400",
  running:   "bg-blue-900 text-blue-300 animate-pulse",
  completed: "bg-green-900 text-green-300",
  failed:    "bg-red-900 text-red-300",
  cancelled: "bg-zinc-800 text-zinc-500",
}

export default function JobMonitorPage() {
  const { id, jobId } = useParams<{ id: string; jobId: string }>()
  const qc = useQueryClient()
  const { data: job, refetch: refetchJob } = useJob(jobId)
  const { data: ckptData, refetch: refetchCkpts } = useCheckpoints(jobId)

  const [livePoints, setLivePoints] = useState<ProgressPoint[]>([])
  const [logs, setLogs] = useState<string[]>([])
  const [streamDone, setStreamDone] = useState(false)

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

  const handleProgress = useCallback((p: ProgressPoint) => {
    setLivePoints(prev => {
      if (prev.find(x => x.epoch === p.epoch)) return prev
      return [...prev, p]
    })
  }, [])

  const handleLog = useCallback((line: string) => {
    setLogs(prev => [...prev.slice(-800), line])
  }, [])

  const handleDone = useCallback((status: string) => {
    setStreamDone(true)
    refetchJob()
    refetchCkpts()
    qc.invalidateQueries({ queryKey: ["jobs", id] })
  }, [refetchJob, refetchCkpts, qc, id])

  const isActive = job?.status === "running" || job?.status === "queued"

  // SSE — only open while job is active
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
          <div className="flex items-center gap-3 mt-1 flex-wrap">
            <span className={`px-2 py-0.5 rounded text-xs font-mono ${STATUS_STYLE[job.status] ?? ""}`}>
              {job.status}
            </span>
            <span className="text-zinc-500 text-xs">{job.base_model}</span>
            <span className="text-zinc-600 text-xs">r={job.lora_r} · alpha={job.lora_alpha}</span>
            <span className="text-zinc-600 text-xs">{job.epochs} epochs · lr={job.learning_rate}</span>
            {job.use_qlora && <span className="text-zinc-600 text-xs">QLoRA</span>}
          </div>
        </div>

        <div className="flex gap-2 flex-wrap justify-end">
          {isActive && (
            <button onClick={cancel}
              className="px-3 py-1.5 border border-red-800 text-red-400 text-xs rounded hover:bg-red-950">
              Cancel
            </button>
          )}
          {job.status === "completed" && !job.merged_path && (
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

        {/* Loss chart */}
        <div className="p-4 border border-zinc-800 rounded bg-zinc-900">
          <div className="flex items-center justify-between mb-3">
            <p className="text-zinc-400 text-xs font-mono">loss curves</p>
            {isActive && chartData.length > 0 && (
              <p className="text-zinc-600 text-xs">
                epoch {chartData.length} / {job.epochs}
              </p>
            )}
          </div>
          <LossChart data={chartData} />
        </div>

        {/* GPU widget */}
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

      {/* ── Completed banner + next steps ── */}
      {job.status === "completed" && (
        <div className="mt-5 p-4 border border-green-800 rounded bg-green-950">
          <p className="text-green-300 font-medium text-sm mb-3">Training complete</p>
          <div className="flex gap-3 flex-wrap">
            {!job.merged_path && (
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
          </div>
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
    </div>
  )
}
