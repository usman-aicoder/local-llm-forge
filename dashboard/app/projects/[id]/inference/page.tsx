"use client"

import { useState } from "react"
import { useParams } from "next/navigation"
import { useInferenceModels } from "@/lib/hooks/useInference"
import { ChatUI } from "@/components/inference/ChatUI"
import { SideBySide } from "@/components/inference/SideBySide"

// ── Generation params panel ───────────────────────────────────────────────────

interface Params {
  temperature: number
  maxTokens: number
  repeatPenalty: number
}

function ParamsPanel({ params, onChange }: { params: Params; onChange: (p: Params) => void }) {
  return (
    <div className="flex flex-wrap gap-4 items-center p-3 border border-zinc-800 rounded bg-zinc-900">
      <div className="flex items-center gap-2">
        <label className="text-zinc-500 text-xs w-24">Temperature</label>
        <input
          type="range" min={0.1} max={2} step={0.05}
          value={params.temperature}
          onChange={e => onChange({ ...params, temperature: Number(e.target.value) })}
          className="w-24 accent-white"
        />
        <span className="text-zinc-300 font-mono text-xs w-8">{params.temperature.toFixed(2)}</span>
      </div>
      <div className="flex items-center gap-2">
        <label className="text-zinc-500 text-xs w-24">Max Tokens</label>
        <input
          type="number" min={64} max={4096} step={64}
          value={params.maxTokens}
          onChange={e => onChange({ ...params, maxTokens: Number(e.target.value) })}
          className="w-20 bg-zinc-800 border border-zinc-700 rounded px-2 py-0.5 text-xs text-white outline-none focus:border-zinc-500"
        />
      </div>
      <div className="flex items-center gap-2">
        <label className="text-zinc-500 text-xs w-24">Repeat Penalty</label>
        <input
          type="range" min={1} max={1.5} step={0.05}
          value={params.repeatPenalty}
          onChange={e => onChange({ ...params, repeatPenalty: Number(e.target.value) })}
          className="w-24 accent-white"
        />
        <span className="text-zinc-300 font-mono text-xs w-8">{params.repeatPenalty.toFixed(2)}</span>
      </div>
    </div>
  )
}

// ── Model select (single-model chat mode) ─────────────────────────────────────

function ModelSelect({
  value,
  onChange,
  data,
}: {
  value: string
  onChange: (v: string) => void
  data: ReturnType<typeof useInferenceModels>["data"]
}) {
  return (
    <select
      value={value}
      onChange={e => onChange(e.target.value)}
      className="bg-zinc-900 border border-zinc-700 rounded px-3 py-2 text-sm text-white outline-none focus:border-zinc-500"
    >
      <option value="">— select model —</option>
      {data?.base_models.length ? (
        <optgroup label="Base Models (Ollama)">
          {data.base_models.map(m => (
            <option key={m.name} value={m.name}>{m.name}</option>
          ))}
        </optgroup>
      ) : null}
      {data?.fine_tuned.length ? (
        <optgroup label="Fine-Tuned Models">
          {data.fine_tuned.map(m => (
            <option key={m.ollama_model_name} value={m.ollama_model_name}>
              {m.job_name} ({m.ollama_model_name})
            </option>
          ))}
        </optgroup>
      ) : null}
    </select>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function InferencePage() {
  const { id } = useParams<{ id: string }>()
  const { data: modelsData, isLoading, error } = useInferenceModels()

  const [tab, setTab] = useState<"chat" | "compare">("chat")
  const [model, setModel] = useState("")
  const [params, setParams] = useState<Params>({
    temperature: 0.7,
    maxTokens: 512,
    repeatPenalty: 1.1,
  })

  return (
    <div className="flex flex-col" style={{ height: "calc(100vh - 120px)" }}>

      {/* Header */}
      <div className="mb-4 shrink-0">
        <h2 className="text-lg font-bold">Inference</h2>
        <p className="text-zinc-500 text-sm mt-0.5">
          Chat with any Ollama model or fine-tuned export.
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-zinc-800 mb-4 shrink-0">
        {(["chat", "compare"] as const).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm border-b-2 transition-colors ${
              tab === t
                ? "border-white text-white"
                : "border-transparent text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {t === "chat" ? "Chat" : "Side by Side"}
          </button>
        ))}
      </div>

      {/* Error / loading */}
      {isLoading && <p className="text-zinc-500 text-sm shrink-0">Loading models…</p>}
      {error && (
        <p className="text-red-400 text-sm shrink-0">
          Could not load models. Is Ollama running?
        </p>
      )}

      {/* Generation params — shared across tabs */}
      <div className="mb-3 shrink-0">
        <ParamsPanel params={params} onChange={setParams} />
      </div>

      {/* ── Chat tab ── */}
      {tab === "chat" && (
        <div className="flex flex-col flex-1 min-h-0 gap-3">
          <div className="shrink-0">
            <ModelSelect value={model} onChange={setModel} data={modelsData} />
          </div>
          <div className="flex-1 min-h-0">
            <ChatUI
              model={model}
              temperature={params.temperature}
              maxTokens={params.maxTokens}
              repeatPenalty={params.repeatPenalty}
              showInput
            />
          </div>
        </div>
      )}

      {/* ── Compare tab ── */}
      {tab === "compare" && modelsData && (
        <div className="flex-1 min-h-0">
          <SideBySide
            models={modelsData}
            temperature={params.temperature}
            maxTokens={params.maxTokens}
            repeatPenalty={params.repeatPenalty}
          />
        </div>
      )}
      {tab === "compare" && !modelsData && !isLoading && (
        <p className="text-zinc-600 text-sm">No models available.</p>
      )}

    </div>
  )
}
