"use client"

import { useState } from "react"
import { useParams } from "next/navigation"
import { useJob } from "@/lib/hooks/useJobs"
import {
  useEvaluation,
  useRunAutoEval,
  useSubmitHuman,
} from "@/lib/hooks/useEvaluations"
import type { Evaluation } from "@/lib/types"

// ── Metric card ───────────────────────────────────────────────────────────────

function MetricCard({ label, value, best }: { label: string; value: number | null; best?: boolean }) {
  return (
    <div className={`p-4 border rounded bg-zinc-900 ${best ? "border-green-700" : "border-zinc-800"}`}>
      <p className="text-zinc-500 text-xs">{label}</p>
      <p className={`font-mono text-lg mt-1 font-bold ${best ? "text-green-400" : "text-white"}`}>
        {value !== null && value !== undefined ? value.toFixed(4) : "—"}
      </p>
    </div>
  )
}

// ── Slider row ────────────────────────────────────────────────────────────────

function SliderRow({
  label,
  value,
  onChange,
}: {
  label: string
  value: number
  onChange: (v: number) => void
}) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-zinc-400 text-xs w-28 shrink-0">{label}</span>
      <input
        type="range"
        min={1}
        max={5}
        step={1}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="flex-1 accent-white"
      />
      <span className="text-white font-mono text-xs w-4 text-right">{value}</span>
    </div>
  )
}

// ── Human eval UI ─────────────────────────────────────────────────────────────

type SampleScores = {
  accuracy: number
  relevance: number
  fluency: number
  completeness: number
}

function HumanEvalUI({
  samples,
  onSubmit,
  submitting,
}: {
  samples: Evaluation["sample_results"]
  onSubmit: (results: Evaluation["sample_results"]) => void
  submitting: boolean
}) {
  const [cursor, setCursor] = useState(0)
  const [scores, setScores] = useState<SampleScores[]>(
    samples.map(() => ({ accuracy: 3, relevance: 3, fluency: 3, completeness: 3 }))
  )
  const [done, setDone] = useState(false)

  if (!samples.length) {
    return <p className="text-zinc-600 text-sm">No samples available for human evaluation.</p>
  }

  if (done) {
    return (
      <div className="p-4 border border-green-800 rounded bg-green-950 text-green-300 text-sm">
        Human evaluation submitted.
      </div>
    )
  }

  const sample = samples[cursor]
  const current = scores[cursor]

  function updateScore(field: keyof SampleScores, val: number) {
    setScores(prev => {
      const next = [...prev]
      next[cursor] = { ...next[cursor], [field]: val }
      return next
    })
  }

  function handleNext() {
    if (cursor < samples.length - 1) {
      setCursor(c => c + 1)
    } else {
      // Merge scores into sample data
      const merged = samples.map((s, i) => ({ ...s, ...scores[i] }))
      setDone(true)
      onSubmit(merged)
    }
  }

  const progress = Math.round(((cursor + 1) / samples.length) * 100)

  return (
    <div className="space-y-4">
      {/* Progress */}
      <div>
        <div className="flex justify-between mb-1">
          <span className="text-zinc-500 text-xs">Human evaluation</span>
          <span className="text-zinc-500 text-xs">{cursor + 1} / {samples.length}</span>
        </div>
        <div className="w-full bg-zinc-800 rounded-full h-1.5">
          <div
            className="h-full bg-white rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Prompt */}
      <div className="p-3 border border-zinc-800 rounded bg-zinc-900">
        <p className="text-zinc-500 text-xs mb-1">Prompt</p>
        <pre className="text-zinc-300 text-xs whitespace-pre-wrap break-words max-h-32 overflow-auto">
          {sample.prompt}
        </pre>
      </div>

      {/* Response */}
      <div className="p-3 border border-zinc-800 rounded bg-zinc-900">
        <p className="text-zinc-500 text-xs mb-1">Model response</p>
        <pre className="text-zinc-300 text-xs whitespace-pre-wrap break-words max-h-32 overflow-auto">
          {sample.response || "—"}
        </pre>
      </div>

      {/* Sliders */}
      <div className="p-4 border border-zinc-800 rounded bg-zinc-900 space-y-3">
        <SliderRow label="Accuracy"     value={current.accuracy}     onChange={v => updateScore("accuracy", v)} />
        <SliderRow label="Relevance"    value={current.relevance}    onChange={v => updateScore("relevance", v)} />
        <SliderRow label="Fluency"      value={current.fluency}      onChange={v => updateScore("fluency", v)} />
        <SliderRow label="Completeness" value={current.completeness} onChange={v => updateScore("completeness", v)} />
      </div>

      <button
        onClick={handleNext}
        disabled={submitting}
        className="px-4 py-2 bg-white text-black text-sm font-medium rounded hover:bg-zinc-200 disabled:opacity-50"
      >
        {cursor < samples.length - 1 ? "Next →" : submitting ? "Submitting…" : "Finish & Submit"}
      </button>
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function EvaluatePage() {
  const { id, jobId } = useParams<{ id: string; jobId: string }>()
  const { data: job } = useJob(jobId)
  const { data: ev, isLoading: evLoading, error: evError } = useEvaluation(jobId)
  const runAuto = useRunAutoEval(jobId)
  const submitHuman = useSubmitHuman(jobId)

  const [tab, setTab] = useState<"auto" | "human">("auto")
  const [autoDispatched, setAutoDispatched] = useState(false)

  function handleRunAuto() {
    setAutoDispatched(true)
    runAuto.mutate()
  }

  if (!job) return <div className="text-zinc-500 text-sm p-4">Loading…</div>

  return (
    <div className="max-w-4xl mx-auto">

      {/* Header */}
      <div className="mb-5">
        <a href={`/projects/${id}/jobs/${jobId}`} className="text-zinc-600 text-xs hover:text-zinc-400">← Job Monitor</a>
        <h2 className="text-lg font-bold mt-1">Evaluate — {job.name}</h2>
        <p className="text-zinc-500 text-xs mt-0.5">{job.base_model}</p>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-5 border-b border-zinc-800">
        {(["auto", "human"] as const).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm border-b-2 transition-colors ${
              tab === t
                ? "border-white text-white"
                : "border-transparent text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {t === "auto" ? "Auto Metrics" : "Human Eval"}
          </button>
        ))}
      </div>

      {/* ── Auto Metrics tab ── */}
      {tab === "auto" && (
        <div className="space-y-5">
          {/* Run button */}
          {!ev && (
            <div className="flex items-center gap-3">
              <button
                onClick={handleRunAuto}
                disabled={runAuto.isPending || autoDispatched}
                className="px-4 py-2 bg-white text-black text-sm font-medium rounded hover:bg-zinc-200 disabled:opacity-50"
              >
                {runAuto.isPending || autoDispatched ? "Running evaluation…" : "Run Auto Evaluation"}
              </button>
              {autoDispatched && !runAuto.isPending && (
                <span className="text-zinc-500 text-xs">
                  Evaluating up to 50 samples — this may take several minutes. Refresh to check.
                </span>
              )}
            </div>
          )}

          {/* Loading */}
          {evLoading && <p className="text-zinc-500 text-sm">Loading evaluation…</p>}

          {/* Results */}
          {ev && (
            <>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                <MetricCard label="ROUGE-L"     value={ev.rouge_l} />
                <MetricCard label="ROUGE-1"     value={ev.rouge_1} />
                <MetricCard label="ROUGE-2"     value={ev.rouge_2} />
                <MetricCard label="BLEU"        value={ev.bleu} />
                <MetricCard label="Perplexity"  value={ev.perplexity} />
                {ev.human_avg_score !== null && (
                  <MetricCard label="Human Score" value={ev.human_avg_score} best />
                )}
              </div>

              <button
                onClick={handleRunAuto}
                disabled={runAuto.isPending || autoDispatched}
                className="px-3 py-1.5 border border-zinc-700 text-zinc-400 text-xs rounded hover:bg-zinc-800 disabled:opacity-50"
              >
                Re-run Evaluation
              </button>
            </>
          )}

          {/* Not found and not dispatched */}
          {evError && !autoDispatched && (
            <p className="text-zinc-500 text-sm">
              No evaluation yet. Click <em>Run Auto Evaluation</em> to start.
            </p>
          )}
        </div>
      )}

      {/* ── Human Eval tab ── */}
      {tab === "human" && (
        <div className="space-y-4">
          {!ev ? (
            <p className="text-zinc-500 text-sm">Run auto evaluation first to generate sample prompts.</p>
          ) : ev.human_avg_score !== null ? (
            <div className="p-4 border border-green-800 rounded bg-green-950">
              <p className="text-green-300 text-sm font-medium mb-2">Human evaluation complete</p>
              <p className="text-zinc-400 text-xs">
                Average score: <span className="text-white font-mono">{ev.human_avg_score.toFixed(2)} / 5</span>
                {" "}across {ev.sample_results.length} samples
              </p>
              <button
                onClick={() => setTab("auto")}
                className="mt-3 px-3 py-1.5 border border-zinc-700 text-zinc-400 text-xs rounded hover:bg-zinc-800"
              >
                View Auto Metrics
              </button>
            </div>
          ) : (
            <HumanEvalUI
              samples={ev.sample_results}
              onSubmit={results => submitHuman.mutate(results)}
              submitting={submitHuman.isPending}
            />
          )}
        </div>
      )}

    </div>
  )
}
