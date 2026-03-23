"use client"

import { useState, useMemo } from "react"
import { useParams } from "next/navigation"
import { useProjectEvaluations, type EvalRow } from "@/lib/hooks/useEvaluations"

type SortKey = keyof Pick<EvalRow, "job_name" | "base_model" | "epochs" | "rouge_l" | "bleu" | "perplexity" | "human_avg_score" | "created_at">

const COLUMNS: { key: SortKey; label: string; numeric?: boolean }[] = [
  { key: "job_name",        label: "Job" },
  { key: "base_model",      label: "Base Model" },
  { key: "epochs",          label: "Epochs",      numeric: true },
  { key: "rouge_l",         label: "ROUGE-L",     numeric: true },
  { key: "bleu",            label: "BLEU",        numeric: true },
  { key: "perplexity",      label: "Perplexity",  numeric: true },
  { key: "human_avg_score", label: "Human Score", numeric: true },
  { key: "created_at",      label: "Date" },
]

function bestVal(rows: EvalRow[], key: SortKey): number | string | null {
  const vals = rows.map(r => r[key]).filter(v => v !== null && v !== undefined)
  if (!vals.length) return null
  if (key === "perplexity") return Math.min(...(vals as number[]))
  if (typeof vals[0] === "number") return Math.max(...(vals as number[]))
  return null
}

function fmt(val: number | string | null | undefined, key: SortKey): string {
  if (val === null || val === undefined) return "—"
  if (typeof val === "number") {
    if (key === "epochs") return String(val)
    if (key === "perplexity") return val.toFixed(2)
    return val.toFixed(4)
  }
  if (key === "created_at") return new Date(val).toLocaleDateString()
  return String(val)
}

export default function EvaluatePage() {
  const { id } = useParams<{ id: string }>()
  const { data, isLoading } = useProjectEvaluations(id)

  const [sortKey, setSortKey]   = useState<SortKey>("created_at")
  const [sortAsc, setSortAsc]   = useState(false)

  const rows = data?.evaluations ?? []

  const best = useMemo(() => {
    const b: Partial<Record<SortKey, number | string | null>> = {}
    for (const col of COLUMNS) {
      if (col.numeric) b[col.key] = bestVal(rows, col.key)
    }
    return b
  }, [rows])

  const sorted = useMemo(() => {
    return [...rows].sort((a, b) => {
      const av = a[sortKey]
      const bv = b[sortKey]
      if (av === null || av === undefined) return 1
      if (bv === null || bv === undefined) return -1
      const cmp = av < bv ? -1 : av > bv ? 1 : 0
      return sortAsc ? cmp : -cmp
    })
  }, [rows, sortKey, sortAsc])

  function handleSort(key: SortKey) {
    if (key === sortKey) setSortAsc(a => !a)
    else { setSortKey(key); setSortAsc(false) }
  }

  return (
    <div>

      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-lg font-bold">Evaluate</h2>
          <p className="text-zinc-500 text-sm mt-0.5">
            Compare completed jobs. Best value per column is highlighted in green.
          </p>
        </div>
      </div>

      {isLoading && <p className="text-zinc-500 text-sm">Loading…</p>}

      {!isLoading && rows.length === 0 && (
        <div className="p-8 border border-zinc-800 rounded bg-zinc-900 text-center">
          <p className="text-zinc-500 text-sm">No evaluations yet.</p>
          <p className="text-zinc-600 text-xs mt-1">
            Complete a training job, then run auto evaluation from the job monitor.
          </p>
        </div>
      )}

      {sorted.length > 0 && (
        <div className="overflow-x-auto rounded border border-zinc-800">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-zinc-800 bg-zinc-900">
                {COLUMNS.map(col => (
                  <th
                    key={col.key}
                    onClick={() => handleSort(col.key)}
                    className="px-3 py-2.5 text-left text-xs font-medium text-zinc-500 cursor-pointer hover:text-zinc-300 whitespace-nowrap select-none"
                  >
                    {col.label}
                    {sortKey === col.key && (
                      <span className="ml-1 text-zinc-400">{sortAsc ? "↑" : "↓"}</span>
                    )}
                  </th>
                ))}
                <th className="px-3 py-2.5 text-left text-xs font-medium text-zinc-500">Actions</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map(row => (
                <tr key={row.job_id} className="border-b border-zinc-800 hover:bg-zinc-900/50">
                  {COLUMNS.map(col => {
                    const val = row[col.key]
                    const isBest = col.numeric && best[col.key] !== null && val === best[col.key]
                    return (
                      <td
                        key={col.key}
                        className={`px-3 py-2.5 font-mono text-xs whitespace-nowrap ${
                          isBest ? "text-green-400 font-bold" : "text-zinc-300"
                        }`}
                      >
                        {fmt(val as number | string | null, col.key)}
                      </td>
                    )
                  })}
                  <td className="px-3 py-2.5">
                    <a
                      href={`/projects/${id}/jobs/${row.job_id}/evaluate`}
                      className="text-zinc-500 text-xs hover:text-zinc-300"
                    >
                      Details →
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

    </div>
  )
}
