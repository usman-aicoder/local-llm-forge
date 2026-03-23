"use client"

import { useState } from "react"
import { useParams } from "next/navigation"
import { useDataset } from "@/lib/hooks/useDatasets"
import { useTaskStatus } from "@/lib/hooks/useTasks"
import { useQueryClient } from "@tanstack/react-query"
import { TokenHistogram } from "@/components/charts/TokenHistogram"
import { PipelineStatus } from "@/components/dataset/PipelineStatus"

const BASE = process.env.NEXT_PUBLIC_API_URL

export default function EDAPage() {
  const { id, dsId } = useParams<{ id: string; dsId: string }>()
  const { data: ds, isLoading } = useDataset(dsId)
  const qc = useQueryClient()
  const [taskId, setTaskId] = useState<string | null>(null)
  const { data: task } = useTaskStatus(taskId)

  async function runEDA() {
    const res = await fetch(`${BASE}/datasets/${dsId}/inspect`, { method: "POST" })
    const json = await res.json()
    setTaskId(json.task_id)
  }

  // Refresh dataset when task completes
  if (task?.status === "completed") {
    qc.invalidateQueries({ queryKey: ["datasets", "detail", dsId] })
  }

  if (isLoading) return <div className="text-zinc-500 text-sm">Loading...</div>
  if (!ds) return <div className="text-red-400 text-sm">Dataset not found.</div>

  const stats = ds.stats

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-6">
        <a href={`/projects/${id}/datasets`} className="text-zinc-600 text-xs hover:text-zinc-400">← Datasets</a>
        <h2 className="text-lg font-bold mt-1">{ds.name} — Inspect</h2>
        <div className="mt-2">
          <PipelineStatus dataset={ds} projectId={id} />
        </div>
      </div>

      {/* Run button */}
      {!stats && !taskId && (
        <button
          onClick={runEDA}
          className="px-4 py-2 bg-white text-black text-sm font-medium rounded hover:bg-zinc-200 mb-6"
        >
          Run EDA Inspection
        </button>
      )}

      {/* Task running */}
      {taskId && (!task || task.status === "pending" || task.status === "running") && (
        <div className="flex items-center gap-2 text-sm text-zinc-400 mb-6">
          <span className="animate-spin">⟳</span> Analysing dataset...
        </div>
      )}

      {task?.status === "failed" && (
        <div className="p-3 mb-6 border border-red-800 rounded bg-red-950 text-red-300 text-xs">
          <p className="font-bold mb-1">Inspection failed</p>
          <pre className="whitespace-pre-wrap">{task.error}</pre>
        </div>
      )}

      {/* Stats */}
      {stats && (
        <div className="space-y-6">
          {/* Stat cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { label: "Total Rows",     value: stats.total_rows.toLocaleString() },
              { label: "Null Values",    value: stats.null_count.toLocaleString() },
              { label: "Duplicates",     value: stats.duplicate_count.toLocaleString() },
              { label: "Flagged Long",   value: stats.flagged_too_long.toLocaleString() },
              { label: "Flagged Short",  value: stats.flagged_too_short.toLocaleString() },
              { label: "Avg Instr Tokens", value: stats.avg_instruction_tokens.toFixed(1) },
              { label: "Avg Out Tokens", value: stats.avg_output_tokens.toFixed(1) },
              { label: "p95 Total Tokens", value: stats.p95_total_tokens.toFixed(0) },
            ].map(({ label, value }) => (
              <div key={label} className="p-3 border border-zinc-800 rounded bg-zinc-900">
                <p className="text-zinc-600 text-xs">{label}</p>
                <p className="text-white font-mono text-lg mt-1">{value}</p>
              </div>
            ))}
          </div>

          {/* Histogram */}
          <div className="p-4 border border-zinc-800 rounded bg-zinc-900">
            <p className="text-zinc-400 text-xs mb-3">Token Length Distribution</p>
            <TokenHistogram
              buckets={stats.token_histogram.buckets ?? []}
              counts={stats.token_histogram.counts ?? []}
            />
          </div>

          {/* Next step */}
          <a
            href={`/projects/${id}/datasets/${dsId}/clean`}
            className="inline-block px-4 py-2 bg-white text-black text-sm font-medium rounded hover:bg-zinc-200"
          >
            Next: Clean →
          </a>
        </div>
      )}

      {/* Re-run button */}
      {stats && !taskId && (
        <button
          onClick={runEDA}
          className="mt-4 px-4 py-1.5 border border-zinc-700 text-xs rounded hover:bg-zinc-800 text-zinc-400"
        >
          Re-run inspection
        </button>
      )}
    </div>
  )
}
