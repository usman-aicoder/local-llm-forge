"use client"

import { useState } from "react"
import { useParams } from "next/navigation"
import { useDataset } from "@/lib/hooks/useDatasets"
import { useTaskStatus } from "@/lib/hooks/useTasks"
import { useQueryClient } from "@tanstack/react-query"
import { PipelineStatus } from "@/components/dataset/PipelineStatus"

const BASE = process.env.NEXT_PUBLIC_API_URL

export default function TokenizePage() {
  const { id, dsId } = useParams<{ id: string; dsId: string }>()
  const { data: ds, isLoading } = useDataset(dsId)
  const qc = useQueryClient()
  const [taskId, setTaskId] = useState<string | null>(null)
  const { data: task } = useTaskStatus(taskId)

  const [maxSeqLen, setMaxSeqLen] = useState(2048)
  const [valSplit, setValSplit] = useState(0.1)

  async function runTokenize() {
    const res = await fetch(`${BASE}/datasets/${dsId}/tokenize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ max_seq_len: maxSeqLen, val_split: valSplit }),
    })
    const json = await res.json()
    setTaskId(json.task_id)
  }

  if (task?.status === "completed") {
    qc.invalidateQueries({ queryKey: ["datasets", "detail", dsId] })
  }

  if (isLoading) return <div className="text-zinc-500 text-sm">Loading...</div>
  if (!ds) return <div className="text-red-400 text-sm">Dataset not found.</div>

  const result = task?.result as Record<string, unknown> | null
  const isTokenized = ds.status === "tokenized"

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-6">
        <a href={`/projects/${id}/datasets/${dsId}/format`} className="text-zinc-600 text-xs hover:text-zinc-400">← Format</a>
        <h2 className="text-lg font-bold mt-1">{ds.name} — Tokenize</h2>
        <div className="mt-2"><PipelineStatus dataset={ds} projectId={id} /></div>
      </div>

      {/* Config */}
      <div className="grid gap-4 mb-6">
        <div>
          <label className="text-zinc-400 text-xs block mb-1">Max Sequence Length</label>
          <input
            type="number"
            value={maxSeqLen}
            onChange={e => setMaxSeqLen(Number(e.target.value))}
            className="bg-zinc-900 border border-zinc-700 rounded px-3 py-2 text-sm w-32 outline-none focus:border-zinc-500"
          />
          <p className="text-zinc-600 text-xs mt-1">Samples longer than this will be truncated</p>
        </div>

        <div>
          <label className="text-zinc-400 text-xs block mb-1">
            Validation Split — {Math.round(valSplit * 100)}%
          </label>
          <input
            type="range"
            min={0.05} max={0.3} step={0.05}
            value={valSplit}
            onChange={e => setValSplit(Number(e.target.value))}
            className="w-64"
          />
          <p className="text-zinc-600 text-xs mt-1">
            Train: {Math.round((1 - valSplit) * 100)}% / Val: {Math.round(valSplit * 100)}%
          </p>
        </div>
      </div>

      <button
        onClick={runTokenize}
        disabled={!!taskId && task?.status !== "failed"}
        className="px-4 py-2 bg-white text-black text-sm font-medium rounded hover:bg-zinc-200 disabled:opacity-50 mb-6"
      >
        {isTokenized ? "Re-Tokenize" : "Tokenize & Split"}
      </button>

      {taskId && (!task || task.status === "pending" || task.status === "running") && (
        <div className="flex items-center gap-2 text-sm text-zinc-400 mb-6">
          <span className="animate-spin">⟳</span> Tokenizing... (downloading tokenizer if needed)
        </div>
      )}
      {task?.status === "failed" && (
        <div className="p-3 mb-6 border border-red-800 rounded bg-red-950 text-red-300 text-xs">
          <pre className="whitespace-pre-wrap">{task.error}</pre>
        </div>
      )}

      {/* Results */}
      {(result || isTokenized) && (
        <div className="space-y-4">
          {result && (
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: "Train Samples",   value: String(result.train_count ?? "–") },
                { label: "Val Samples",     value: String(result.val_count ?? "–") },
                { label: "Truncated",       value: String(result.truncated_count ?? "–") },
                { label: "Tokenizer Used",  value: String(result.tokenizer_used ?? "–") },
              ].map(({ label, value }) => (
                <div key={label} className="p-3 border border-zinc-800 rounded bg-zinc-900">
                  <p className="text-zinc-600 text-xs">{label}</p>
                  <p className="text-white font-mono text-sm mt-1 truncate">{value}</p>
                </div>
              ))}
            </div>
          )}

          <div className="p-3 border border-green-900 rounded bg-green-950 text-green-300 text-sm">
            Dataset is ready for training.
          </div>

          <a
            href={`/projects/${id}/jobs/new`}
            className="inline-block px-4 py-2 bg-white text-black text-sm font-medium rounded hover:bg-zinc-200"
          >
            Go to New Job →
          </a>
        </div>
      )}
    </div>
  )
}
