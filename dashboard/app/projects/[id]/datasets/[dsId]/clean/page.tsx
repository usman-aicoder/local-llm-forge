"use client"

import { useState } from "react"
import { useParams } from "next/navigation"
import { useDataset } from "@/lib/hooks/useDatasets"
import { useTaskStatus } from "@/lib/hooks/useTasks"
import { useQueryClient } from "@tanstack/react-query"
import { PipelineStatus } from "@/components/dataset/PipelineStatus"

const BASE = process.env.NEXT_PUBLIC_API_URL

const STEPS = [
  { key: "strip_html",           label: "Strip HTML",          desc: "Remove HTML tags and entities" },
  { key: "normalize_whitespace", label: "Normalize Whitespace", desc: "Collapse extra spaces and newlines" },
  { key: "remove_urls",          label: "Remove URLs",          desc: "Strip http/www links from text" },
  { key: "deduplicate",          label: "Deduplicate",          desc: "Remove exact instruction+output pairs" },
  { key: "filter_short",         label: "Filter Short",         desc: "Remove samples with very short content" },
]

export default function CleanPage() {
  const { id, dsId } = useParams<{ id: string; dsId: string }>()
  const { data: ds, isLoading } = useDataset(dsId)
  const qc = useQueryClient()
  const [taskId, setTaskId] = useState<string | null>(null)
  const { data: task } = useTaskStatus(taskId)
  const [config, setConfig] = useState<Record<string, boolean>>(
    Object.fromEntries(STEPS.map(s => [s.key, true]))
  )

  async function runClean() {
    const res = await fetch(`${BASE}/datasets/${dsId}/clean`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    })
    const json = await res.json()
    setTaskId(json.task_id)
  }

  if (task?.status === "completed") {
    qc.invalidateQueries({ queryKey: ["datasets", "detail", dsId] })
  }

  if (isLoading) return <div className="text-zinc-500 text-sm">Loading...</div>
  if (!ds) return <div className="text-red-400 text-sm">Dataset not found.</div>

  const report = ds.stats?.cleaning_report ?? {}
  const alreadyCleaned = ds.status === "cleaned" || ds.status === "formatted" || ds.status === "tokenized"

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-6">
        <a href={`/projects/${id}/datasets/${dsId}/eda`} className="text-zinc-600 text-xs hover:text-zinc-400">← Inspect</a>
        <h2 className="text-lg font-bold mt-1">{ds.name} — Clean</h2>
        <div className="mt-2">
          <PipelineStatus dataset={ds} projectId={id} />
        </div>
      </div>

      {/* Config toggles */}
      <div className="space-y-2 mb-6">
        {STEPS.map(step => (
          <label key={step.key} className="flex items-center justify-between p-3 border border-zinc-800 rounded bg-zinc-900 cursor-pointer hover:border-zinc-700">
            <div>
              <p className="text-sm font-medium">{step.label}</p>
              <p className="text-zinc-500 text-xs">{step.desc}</p>
            </div>
            <div
              onClick={() => setConfig(c => ({ ...c, [step.key]: !c[step.key] }))}
              className={`w-10 h-5 rounded-full transition-colors cursor-pointer ${config[step.key] ? "bg-white" : "bg-zinc-700"}`}
            >
              <div className={`w-4 h-4 mt-0.5 rounded-full bg-black transition-transform ${config[step.key] ? "translate-x-5 ml-0.5" : "translate-x-0.5"}`} />
            </div>
          </label>
        ))}
      </div>

      {/* Run button */}
      {!taskId && (
        <button
          onClick={runClean}
          className="px-4 py-2 bg-white text-black text-sm font-medium rounded hover:bg-zinc-200 mb-6"
        >
          {alreadyCleaned ? "Re-run Cleaning" : "Run Cleaning"}
        </button>
      )}

      {/* Task state */}
      {taskId && (!task || task.status === "pending" || task.status === "running") && (
        <div className="flex items-center gap-2 text-sm text-zinc-400 mb-6">
          <span className="animate-spin">⟳</span> Cleaning dataset...
        </div>
      )}

      {task?.status === "failed" && (
        <div className="p-3 mb-6 border border-red-800 rounded bg-red-950 text-red-300 text-xs">
          <p className="font-bold mb-1">Cleaning failed</p>
          <pre className="whitespace-pre-wrap">{task.error}</pre>
        </div>
      )}

      {/* Cleaning report */}
      {alreadyCleaned && Object.keys(report).length > 0 && (
        <div className="mb-6">
          <p className="text-zinc-400 text-xs mb-2">Cleaning Report</p>
          <div className="border border-zinc-800 rounded overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-zinc-800 bg-zinc-900">
                  <th className="text-left px-3 py-2 text-zinc-500 text-xs">Step</th>
                  <th className="text-right px-3 py-2 text-zinc-500 text-xs">Rows Removed</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(report).map(([step, count]) => (
                  <tr key={step} className="border-b border-zinc-900">
                    <td className="px-3 py-2 text-zinc-300 font-mono text-xs">{step}</td>
                    <td className="px-3 py-2 text-right font-mono text-xs text-zinc-400">{String(count)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-zinc-600 text-xs mt-2">{ds.row_count?.toLocaleString()} rows remaining</p>
        </div>
      )}

      {alreadyCleaned && (
        <a
          href={`/projects/${id}/datasets/${dsId}/format`}
          className="inline-block px-4 py-2 bg-white text-black text-sm font-medium rounded hover:bg-zinc-200"
        >
          Next: Format →
        </a>
      )}
    </div>
  )
}
