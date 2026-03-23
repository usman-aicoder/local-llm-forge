"use client"

import { useState } from "react"
import { useParams } from "next/navigation"
import { useDataset } from "@/lib/hooks/useDatasets"
import { useOllamaModels } from "@/lib/hooks/useModels"
import { useTaskStatus } from "@/lib/hooks/useTasks"
import { useQueryClient } from "@tanstack/react-query"
import { PipelineStatus } from "@/components/dataset/PipelineStatus"

const BASE = process.env.NEXT_PUBLIC_API_URL

export default function FormatPage() {
  const { id, dsId } = useParams<{ id: string; dsId: string }>()
  const { data: ds, isLoading } = useDataset(dsId)
  const { data: modelsData } = useOllamaModels()
  const qc = useQueryClient()
  const [taskId, setTaskId] = useState<string | null>(null)
  const { data: task } = useTaskStatus(taskId)

  const [formatType, setFormatType] = useState<"alpaca" | "chat" | "dpo">("alpaca")
  const [baseModel, setBaseModel] = useState("mistral:7b")
  const [preview, setPreview] = useState<string[]>([])
  const [previewing, setPreviewing] = useState(false)

  async function loadPreview() {
    setPreviewing(true)
    try {
      const res = await fetch(
        `${BASE}/datasets/${dsId}/format/preview?format_type=${formatType}&base_model=${encodeURIComponent(baseModel)}`
      )
      const json = await res.json()
      setPreview(json.samples ?? [])
    } finally {
      setPreviewing(false)
    }
  }

  async function runFormat() {
    const res = await fetch(`${BASE}/datasets/${dsId}/format`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ format_type: formatType, base_model: baseModel }),
    })
    const json = await res.json()
    setTaskId(json.task_id)
  }

  if (task?.status === "completed") {
    qc.invalidateQueries({ queryKey: ["datasets", "detail", dsId] })
  }

  if (isLoading) return <div className="text-zinc-500 text-sm">Loading...</div>
  if (!ds) return <div className="text-red-400 text-sm">Dataset not found.</div>

  const isFormatted = ds.status === "formatted" || ds.status === "tokenized"

  return (
    <div className="max-w-3xl mx-auto">
      <div className="mb-6">
        <a href={`/projects/${id}/datasets/${dsId}/clean`} className="text-zinc-600 text-xs hover:text-zinc-400">← Clean</a>
        <h2 className="text-lg font-bold mt-1">{ds.name} — Format</h2>
        <div className="mt-2"><PipelineStatus dataset={ds} projectId={id} /></div>
      </div>

      {/* Config */}
      <div className="grid gap-4 mb-6">
        {/* Format type */}
        <div>
          <p className="text-zinc-400 text-xs mb-2">Format Type</p>
          <div className="flex gap-2 flex-wrap">
            {([
              { id: "alpaca", label: "Alpaca (single-turn)" },
              { id: "chat",   label: "Chat / Messages" },
              { id: "dpo",    label: "DPO / Preference" },
            ] as const).map(({ id: ft, label }) => (
              <button
                key={ft}
                onClick={() => setFormatType(ft)}
                className={`px-4 py-2 rounded text-sm border transition-colors ${
                  formatType === ft
                    ? "bg-white text-black border-white"
                    : "border-zinc-700 text-zinc-400 hover:border-zinc-500"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          {formatType === "dpo" && (
            <div className="mt-2 px-3 py-2 bg-blue-950 border border-blue-800 rounded text-blue-300 text-xs">
              Requires columns: <code className="font-mono">prompt</code> (or instruction/question), <code className="font-mono">chosen</code> (or good_answer/accepted), <code className="font-mono">rejected</code> (or bad_answer/negative). DPO datasets skip the tokenize step — use directly when creating a DPO/ORPO training job.
            </div>
          )}
        </div>

        {/* Base model — hidden for DPO since it's model-agnostic */}
        {formatType !== "dpo" && (
          <div>
            <p className="text-zinc-400 text-xs mb-2">Base Model (sets prompt template)</p>
            <select
              value={baseModel}
              onChange={e => setBaseModel(e.target.value)}
              className="bg-zinc-900 border border-zinc-700 rounded px-3 py-2 text-sm text-white outline-none focus:border-zinc-500 w-full max-w-xs"
            >
              {modelsData?.models.map(m => (
                <option key={m.name} value={m.name}>{m.name}</option>
              )) ?? <option value="mistral:7b">mistral:7b</option>}
            </select>
          </div>
        )}
      </div>

      {/* Preview */}
      <div className="flex gap-2 mb-4">
        <button
          onClick={loadPreview}
          disabled={previewing}
          className="px-4 py-2 border border-zinc-700 text-sm rounded hover:bg-zinc-800 text-zinc-300 disabled:opacity-50"
        >
          {previewing ? "Loading..." : "Preview 10 Samples"}
        </button>
        <button
          onClick={runFormat}
          disabled={!!taskId}
          className="px-4 py-2 bg-white text-black text-sm font-medium rounded hover:bg-zinc-200 disabled:opacity-50"
        >
          {isFormatted ? "Re-Format" : "Format Dataset"}
        </button>
      </div>

      {preview.length > 0 && (
        <div className="mb-6 space-y-2">
          <p className="text-zinc-500 text-xs">Sample Preview</p>
          {preview.map((s, i) => (
            <pre
              key={i}
              className="p-3 border border-zinc-800 rounded bg-zinc-900 text-xs text-zinc-300 font-mono whitespace-pre-wrap overflow-auto max-h-40"
            >
              {s}
            </pre>
          ))}
        </div>
      )}

      {/* Task state */}
      {taskId && (!task || task.status === "pending" || task.status === "running") && (
        <div className="flex items-center gap-2 text-sm text-zinc-400 mb-4">
          <span className="animate-spin">⟳</span> Formatting...
        </div>
      )}
      {task?.status === "failed" && (
        <div className="p-3 mb-4 border border-red-800 rounded bg-red-950 text-red-300 text-xs">
          <pre className="whitespace-pre-wrap">{task.error}</pre>
        </div>
      )}
      {(task?.status === "completed" || isFormatted) && (
        formatType === "dpo" ? (
          <a
            href={`/projects/${id}/jobs/new`}
            className="inline-block px-4 py-2 bg-white text-black text-sm font-medium rounded hover:bg-zinc-200"
          >
            Next: Create DPO/ORPO Job →
          </a>
        ) : (
          <a
            href={`/projects/${id}/datasets/${dsId}/tokenize`}
            className="inline-block px-4 py-2 bg-white text-black text-sm font-medium rounded hover:bg-zinc-200"
          >
            Next: Tokenize →
          </a>
        )
      )}
    </div>
  )
}
