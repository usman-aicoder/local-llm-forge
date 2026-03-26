"use client"

import { useState } from "react"
import { useParams } from "next/navigation"
import { useQuery } from "@tanstack/react-query"
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend,
  ResponsiveContainer, BarChart, Bar, Cell,
} from "recharts"
import { api } from "@/lib/api"
import type { Experiment } from "@/lib/types"

// ── Hooks ─────────────────────────────────────────────────────────────────────

function useExperiments(projectId: string) {
  return useQuery({
    queryKey: ["experiments", projectId],
    queryFn: () => api.projects.experimentsSummary(projectId),
    enabled: !!projectId,
  })
}

// ── Metric bar chart ──────────────────────────────────────────────────────────

const METRIC_COLORS = ["#a1a1aa", "#71717a", "#52525b", "#3f3f46", "#27272a"]
const CHART_METRICS = [
  { key: "rouge_1", label: "ROUGE-1" },
  { key: "rouge_2", label: "ROUGE-2" },
  { key: "rouge_l", label: "ROUGE-L" },
  { key: "bleu",    label: "BLEU" },
] as const

function MetricChart({ experiments, metric, label }: {
  experiments: Experiment[]
  metric: keyof Experiment["metrics"]
  label: string
}) {
  const data = experiments
    .filter(e => e.metrics[metric] !== null)
    .map((e, i) => ({
      name: e.name.length > 12 ? e.name.slice(0, 12) + "…" : e.name,
      value: e.metrics[metric] as number,
      color: METRIC_COLORS[i % METRIC_COLORS.length],
    }))

  if (!data.length) {
    return (
      <div className="flex items-center justify-center h-28 text-zinc-600 text-xs">
        No {label} data
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={150}>
      <BarChart data={data} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
        <XAxis dataKey="name" tick={{ fontSize: 10, fill: "#71717a" }} />
        <YAxis tick={{ fontSize: 10, fill: "#71717a" }} domain={[0, 1]} />
        <Tooltip
          contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", fontSize: 11 }}
          formatter={(v) => (typeof v === "number" ? v.toFixed(3) : v)}
        />
        <Bar dataKey="value" name={label} radius={[2, 2, 0, 0]}>
          {data.map((entry, i) => (
            <Cell key={i} fill={i === 0 ? "#ffffff" : "#52525b"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── Comparison line chart (LR vs metric) ─────────────────────────────────────

function LRvsMetricChart({ experiments }: { experiments: Experiment[] }) {
  const data = experiments
    .filter(e => e.metrics.rouge_l !== null)
    .map(e => ({
      name: e.name.length > 10 ? e.name.slice(0, 10) + "…" : e.name,
      lr: e.hyperparams.learning_rate,
      rouge_l: e.metrics.rouge_l,
    }))
    .sort((a, b) => a.lr - b.lr)

  if (!data.length) return null

  return (
    <div className="p-4 border border-zinc-800 rounded bg-zinc-900">
      <p className="text-zinc-400 text-xs font-mono mb-3">learning rate vs ROUGE-L</p>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={data} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
          <XAxis
            dataKey="lr"
            tick={{ fontSize: 10, fill: "#71717a" }}
            tickFormatter={v => typeof v === "number" ? v.toExponential(0) : v}
          />
          <YAxis tick={{ fontSize: 10, fill: "#71717a" }} domain={[0, 1]} width={35} />
          <Tooltip
            contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", fontSize: 11 }}
            formatter={(v) => (typeof v === "number" ? v.toFixed(3) : v)}
          />
          <Line type="monotone" dataKey="rouge_l" stroke="#ffffff" strokeWidth={2} dot={{ r: 4 }} name="ROUGE-L" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── Experiment row ────────────────────────────────────────────────────────────

const METHOD_STYLE: Record<string, string> = {
  sft:  "bg-zinc-800 text-zinc-300",
  dpo:  "bg-purple-900 text-purple-300",
  orpo: "bg-blue-900 text-blue-300",
  fft:  "bg-amber-900 text-amber-300",
}

function ExperimentRow({ exp, projectId }: { exp: Experiment; projectId: string }) {
  const m = exp.metrics

  function fmt(v: number | null) {
    return v === null ? <span className="text-zinc-700">—</span> : v.toFixed(3)
  }

  return (
    <tr className="border-b border-zinc-900 hover:bg-zinc-900/50">
      <td className="p-3">
        <p className="text-white text-xs font-medium">{exp.name}</p>
        <p className="text-zinc-600 text-xs mt-0.5">{exp.base_model}</p>
      </td>
      <td className="p-3 text-center">
        <span className={`px-1.5 py-0.5 rounded text-xs uppercase ${METHOD_STYLE[exp.training_method] ?? "bg-zinc-800 text-zinc-400"}`}>
          {exp.training_method}
        </span>
      </td>
      <td className="p-3 text-center font-mono text-xs text-zinc-300">{exp.hyperparams.learning_rate.toExponential(0)}</td>
      <td className="p-3 text-center font-mono text-xs text-zinc-300">{exp.hyperparams.epochs}</td>
      <td className="p-3 text-center font-mono text-xs text-zinc-300">{exp.hyperparams.lora_r}</td>
      <td className="p-3 text-center font-mono text-xs text-zinc-300">{fmt(m.rouge_1)}</td>
      <td className="p-3 text-center font-mono text-xs text-zinc-300">{fmt(m.rouge_l)}</td>
      <td className="p-3 text-center font-mono text-xs text-zinc-300">{fmt(m.bleu)}</td>
      <td className="p-3 text-center">
        <a
          href={`/projects/${projectId}/jobs/${exp.id}`}
          className="text-xs text-zinc-500 hover:text-zinc-300 underline mr-3"
        >
          View
        </a>
        <a
          href={`/projects/${projectId}/jobs/new?clone=${exp.id}`}
          className="text-xs text-zinc-500 hover:text-zinc-300 underline"
        >
          Clone
        </a>
      </td>
    </tr>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function ExperimentsPage() {
  const { id } = useParams<{ id: string }>()
  const { data, isLoading } = useExperiments(id)
  const experiments = data?.experiments ?? []

  const hasMetrics = experiments.some(e =>
    Object.values(e.metrics).some(v => v !== null)
  )

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-6">
        <a href={`/projects/${id}`} className="text-zinc-600 text-xs hover:text-zinc-400">← Project</a>
        <h2 className="text-lg font-bold mt-1">Experiment Comparison</h2>
        <p className="text-zinc-500 text-xs mt-1">
          Completed jobs with their hyperparameters and evaluation metrics.
        </p>
      </div>

      {isLoading && (
        <div className="text-zinc-500 text-sm">Loading experiments…</div>
      )}

      {!isLoading && experiments.length === 0 && (
        <div className="p-8 border border-zinc-800 rounded bg-zinc-900 text-center">
          <p className="text-zinc-500 text-sm">No completed jobs yet.</p>
          <a href={`/projects/${id}/jobs/new`} className="text-xs text-zinc-400 underline mt-2 block">
            Create a training job →
          </a>
        </div>
      )}

      {experiments.length > 0 && (
        <>
          {/* Metric charts */}
          {hasMetrics && (
            <div className="mb-6">
              <p className="text-zinc-400 text-xs font-medium uppercase tracking-wider mb-3">Metric Comparison</p>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
                {CHART_METRICS.map(({ key, label }) => (
                  <div key={key} className="p-3 border border-zinc-800 rounded bg-zinc-900">
                    <p className="text-zinc-500 text-xs font-mono mb-2">{label}</p>
                    <MetricChart experiments={experiments} metric={key} label={label} />
                  </div>
                ))}
              </div>
              <LRvsMetricChart experiments={experiments} />
            </div>
          )}

          {!hasMetrics && (
            <div className="mb-6 p-4 border border-zinc-800 rounded bg-zinc-900 text-center text-zinc-500 text-sm">
              No evaluation metrics yet. Run{" "}
              <a href={`/projects/${id}/evaluate`} className="underline">Evaluate</a>
              {" "}on your completed jobs to see charts here.
            </div>
          )}

          {/* Comparison table */}
          <div className="border border-zinc-800 rounded overflow-hidden">
            <div className="p-3 border-b border-zinc-800 flex items-center justify-between">
              <p className="text-zinc-400 text-xs font-medium uppercase tracking-wider">
                All Experiments ({experiments.length})
              </p>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="bg-zinc-950 border-b border-zinc-800">
                  <tr className="text-zinc-500">
                    <th className="text-left p-3 font-normal">Job</th>
                    <th className="text-center p-3 font-normal">Method</th>
                    <th className="text-center p-3 font-normal">LR</th>
                    <th className="text-center p-3 font-normal">Epochs</th>
                    <th className="text-center p-3 font-normal">LoRA r</th>
                    <th className="text-center p-3 font-normal">ROUGE-1</th>
                    <th className="text-center p-3 font-normal">ROUGE-L</th>
                    <th className="text-center p-3 font-normal">BLEU</th>
                    <th className="p-3"></th>
                  </tr>
                </thead>
                <tbody>
                  {experiments.map(exp => (
                    <ExperimentRow key={exp.id} exp={exp} projectId={id} />
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
