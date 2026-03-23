"use client"

import { useState, useRef, useEffect } from "react"
import { useQuery } from "@tanstack/react-query"
import { api } from "@/lib/api"
import { useInferenceModels } from "@/lib/hooks/useInference"
import { useTaskStatus } from "@/lib/hooks/useTasks"

type Tab = "upload" | "pdf" | "url" | "presets"

// ── Shared model selector ─────────────────────────────────────────────────────

function OllamaModelSelect({
  value,
  onChange,
}: {
  value: string
  onChange: (v: string) => void
}) {
  const { data } = useInferenceModels()
  const models = data?.base_models ?? []

  return (
    <div>
      <label className="text-zinc-400 text-xs block mb-1">
        Ollama model for Q&A generation
        <span className="text-zinc-600 ml-1">(smaller = faster, e.g. gemma2:2b)</span>
      </label>
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-white outline-none focus:border-zinc-500"
      >
        <option value="">— select model —</option>
        {models.map(m => (
          <option key={m.name} value={m.name}>{m.name}</option>
        ))}
      </select>
    </div>
  )
}

// ── Pairs-per-chunk slider ────────────────────────────────────────────────────

function PairsSlider({ value, onChange }: { value: number; onChange: (n: number) => void }) {
  return (
    <div>
      <label className="text-zinc-400 text-xs block mb-1">
        Q&A pairs per text chunk
        <span className="text-zinc-600 ml-1">(more = richer dataset, slower generation)</span>
      </label>
      <div className="flex items-center gap-3">
        <input
          type="range" min={1} max={6} step={1}
          value={value}
          onChange={e => onChange(Number(e.target.value))}
          className="flex-1 accent-white"
        />
        <span className="text-white font-mono text-sm w-4">{value}</span>
      </div>
    </div>
  )
}

// ── Generation progress indicator ────────────────────────────────────────────

function GeneratingBanner({ taskId, onDone }: { taskId: string; onDone: () => void }) {
  const { data: task } = useTaskStatus(taskId)

  useEffect(() => {
    if (task?.status === "completed" || task?.status === "failed") {
      onDone()
    }
  }, [task?.status, onDone])

  const failed = task?.status === "failed"

  return (
    <div className={`p-4 border rounded ${failed ? "border-red-800 bg-red-950" : "border-zinc-800 bg-zinc-900"}`}>
      {failed ? (
        <>
          <p className="text-red-300 text-sm font-medium mb-1">Generation failed</p>
          <pre className="text-red-400 text-xs whitespace-pre-wrap max-h-32 overflow-auto">
            {task?.error ?? "Unknown error"}
          </pre>
        </>
      ) : (
        <div className="flex items-center gap-3">
          <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse shrink-0" />
          <div>
            <p className="text-zinc-300 text-sm">Generating dataset…</p>
            <p className="text-zinc-600 text-xs mt-0.5">
              Extracting text and calling Ollama to generate Q&A pairs.
              This may take a few minutes depending on content length.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Upload tab ────────────────────────────────────────────────────────────────

function UploadTab({ projectId, onSuccess }: { projectId: string; onSuccess: () => void }) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const BASE = process.env.NEXT_PUBLIC_API_URL

  async function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    setError(null)
    try {
      const form = new FormData()
      form.append("file", file)
      const res = await fetch(`${BASE}/projects/${projectId}/datasets/upload`, {
        method: "POST",
        body: form,
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail ?? "Upload failed")
      }
      onSuccess()
    } catch (err: unknown) {
      setError((err as Error).message)
    } finally {
      setUploading(false)
      if (fileRef.current) fileRef.current.value = ""
    }
  }

  return (
    <div className="space-y-4">
      <p className="text-zinc-400 text-sm">
        Upload a CSV, JSON, or JSONL file with <code className="text-zinc-300">instruction</code> and{" "}
        <code className="text-zinc-300">output</code> columns.
      </p>

      <label className={`flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-lg cursor-pointer transition-colors ${uploading ? "border-zinc-700 opacity-50" : "border-zinc-700 hover:border-zinc-500"}`}>
        <span className="text-zinc-500 text-sm">{uploading ? "Uploading…" : "Click to browse or drag & drop"}</span>
        <span className="text-zinc-600 text-xs mt-1">CSV · JSON · JSONL</span>
        <input
          ref={fileRef}
          type="file"
          accept=".csv,.json,.jsonl"
          className="hidden"
          onChange={handleFile}
          disabled={uploading}
        />
      </label>

      {error && <p className="text-red-400 text-xs">{error}</p>}
    </div>
  )
}

// ── PDF tab ───────────────────────────────────────────────────────────────────

function PdfTab({ projectId, onSuccess }: { projectId: string; onSuccess: () => void }) {
  const [name, setName]         = useState("")
  const [model, setModel]       = useState("gemma2:2b")
  const [pairs, setPairs]       = useState(3)
  const [file, setFile]         = useState<File | null>(null)
  const [loading, setLoading]   = useState(false)
  const [taskId, setTaskId]     = useState<string | null>(null)
  const [error, setError]       = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] ?? null
    setFile(f)
    if (f && !name) setName(f.name.replace(/\.pdf$/i, ""))
  }

  async function handleSubmit() {
    if (!file || !name.trim() || !model) return
    setLoading(true)
    setError(null)
    try {
      const form = new FormData()
      form.append("file", file)
      form.append("name", name.trim())
      form.append("ollama_model", model)
      form.append("pairs_per_chunk", String(pairs))
      const res = await api.datasets.fromPdf(projectId, form)
      setTaskId(res.task_id)
    } catch (err: unknown) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  if (taskId) {
    return <GeneratingBanner taskId={taskId} onDone={onSuccess} />
  }

  return (
    <div className="space-y-4">
      <p className="text-zinc-400 text-sm">
        Upload a PDF — text will be extracted, chunked, and an Ollama model will generate
        instruction/output Q&A pairs automatically.
      </p>

      {/* PDF file picker */}
      <div>
        <label className="text-zinc-400 text-xs block mb-1">PDF file</label>
        <label className={`flex items-center justify-center w-full h-24 border-2 border-dashed rounded-lg cursor-pointer transition-colors ${file ? "border-zinc-600" : "border-zinc-700 hover:border-zinc-500"}`}>
          <span className="text-zinc-500 text-sm">
            {file ? file.name : "Click to select PDF"}
          </span>
          <input ref={fileRef} type="file" accept=".pdf" className="hidden" onChange={handleFileChange} />
        </label>
      </div>

      {/* Dataset name */}
      <div>
        <label className="text-zinc-400 text-xs block mb-1">Dataset name</label>
        <input
          value={name}
          onChange={e => setName(e.target.value)}
          placeholder="e.g. Product Manual QA"
          className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-white outline-none focus:border-zinc-500 placeholder:text-zinc-600"
        />
      </div>

      <OllamaModelSelect value={model} onChange={setModel} />
      <PairsSlider value={pairs} onChange={setPairs} />

      {error && <p className="text-red-400 text-xs">{error}</p>}

      <button
        onClick={handleSubmit}
        disabled={!file || !name.trim() || !model || loading}
        className="w-full py-2 bg-white text-black text-sm font-medium rounded hover:bg-zinc-200 disabled:opacity-40"
      >
        {loading ? "Starting…" : "Generate Dataset from PDF"}
      </button>
    </div>
  )
}

// ── URL tab ───────────────────────────────────────────────────────────────────

function UrlTab({ projectId, onSuccess }: { projectId: string; onSuccess: () => void }) {
  const [url, setUrl]           = useState("")
  const [name, setName]         = useState("")
  const [model, setModel]       = useState("gemma2:2b")
  const [pairs, setPairs]       = useState(3)
  const [loading, setLoading]   = useState(false)
  const [taskId, setTaskId]     = useState<string | null>(null)
  const [error, setError]       = useState<string | null>(null)

  async function handleSubmit() {
    if (!url.trim() || !name.trim() || !model) return
    setLoading(true)
    setError(null)
    try {
      const res = await api.datasets.fromUrl(projectId, {
        url: url.trim(),
        name: name.trim(),
        ollama_model: model,
        pairs_per_chunk: pairs,
      })
      setTaskId(res.task_id)
    } catch (err: unknown) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  if (taskId) {
    return <GeneratingBanner taskId={taskId} onDone={onSuccess} />
  }

  return (
    <div className="space-y-4">
      <p className="text-zinc-400 text-sm">
        Enter a URL — the page content will be scraped, chunked, and Q&A pairs will be
        generated using Ollama.
      </p>

      <div>
        <label className="text-zinc-400 text-xs block mb-1">URL</label>
        <input
          value={url}
          onChange={e => setUrl(e.target.value)}
          placeholder="https://example.com/article"
          type="url"
          className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-white outline-none focus:border-zinc-500 placeholder:text-zinc-600"
        />
      </div>

      <div>
        <label className="text-zinc-400 text-xs block mb-1">Dataset name</label>
        <input
          value={name}
          onChange={e => setName(e.target.value)}
          placeholder="e.g. Blog Post QA"
          className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-white outline-none focus:border-zinc-500 placeholder:text-zinc-600"
        />
      </div>

      <OllamaModelSelect value={model} onChange={setModel} />
      <PairsSlider value={pairs} onChange={setPairs} />

      {error && <p className="text-red-400 text-xs">{error}</p>}

      <button
        onClick={handleSubmit}
        disabled={!url.trim() || !name.trim() || !model || loading}
        className="w-full py-2 bg-white text-black text-sm font-medium rounded hover:bg-zinc-200 disabled:opacity-40"
      >
        {loading ? "Starting…" : "Scrape & Generate Dataset"}
      </button>
    </div>
  )
}

// ── Presets tab ───────────────────────────────────────────────────────────────

const BASE = process.env.NEXT_PUBLIC_API_URL

interface PresetVariant {
  id: string
  label: string
  description: string
  file: string
  rows: number
  format_type: string | null
}

interface Preset {
  id: string
  name: string
  description: string
  hf_source: string
  variants: PresetVariant[]
}

function PresetsTab({ projectId, onSuccess }: { projectId: string; onSuccess: () => void }) {
  const [importing, setImporting] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [imported, setImported] = useState<Set<string>>(new Set())

  const { data, isLoading } = useQuery<{ presets: Preset[] }>({
    queryKey: ["presets"],
    queryFn: () => fetch(`${BASE}/datasets/presets`).then(r => r.json()),
  })

  async function handleImport(presetId: string, variantId: string) {
    const key = `${presetId}:${variantId}`
    setImporting(key)
    setError(null)
    try {
      const res = await fetch(`${BASE}/projects/${projectId}/datasets/from-preset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ preset_id: presetId, variant_id: variantId }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail ?? "Import failed")
      }
      setImported(prev => new Set([...prev, key]))
    } catch (err: unknown) {
      setError((err as Error).message)
    } finally {
      setImporting(null)
    }
  }

  if (isLoading) {
    return <p className="text-zinc-500 text-sm">Loading presets...</p>
  }

  if (!data?.presets.length) {
    return (
      <div className="space-y-3">
        <p className="text-zinc-400 text-sm">No preset datasets found.</p>
        <div className="p-3 bg-zinc-900 border border-zinc-800 rounded text-xs text-zinc-500">
          Run the prep script to download the demo datasets:
          <pre className="mt-2 text-zinc-400 font-mono">
{`cd backend
source venv/bin/activate
python scripts/prepare_demo_dataset.py`}
          </pre>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-5">
      <p className="text-zinc-400 text-sm">
        Pre-prepared datasets ready to import. Use these to run the SFT vs DPO vs ORPO demo.
      </p>

      {error && <p className="text-red-400 text-xs">{error}</p>}

      {data.presets.map(preset => (
        <div key={preset.id} className="border border-zinc-800 rounded overflow-hidden">
          {/* Preset header */}
          <div className="px-4 py-3 bg-zinc-900 border-b border-zinc-800">
            <div className="flex items-center justify-between">
              <p className="text-sm font-medium text-white">{preset.name}</p>
              <span className="text-xs text-zinc-500 font-mono">{preset.hf_source}</span>
            </div>
            <p className="text-xs text-zinc-500 mt-0.5">{preset.description}</p>
          </div>

          {/* Variants */}
          <div className="divide-y divide-zinc-800">
            {preset.variants.map(variant => {
              const key = `${preset.id}:${variant.id}`
              const isImporting = importing === key
              const done = imported.has(key)
              return (
                <div key={variant.id} className="flex items-start justify-between gap-4 px-4 py-3">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <p className="text-xs font-medium text-zinc-200">{variant.label}</p>
                      <span className="text-xs text-zinc-600">{variant.rows} rows</span>
                      {variant.format_type && (
                        <span className="px-1.5 py-0.5 rounded text-xs bg-zinc-800 text-zinc-400">
                          {variant.format_type}
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-zinc-600 mt-0.5">{variant.description}</p>
                  </div>
                  <button
                    onClick={() => handleImport(preset.id, variant.id)}
                    disabled={isImporting || done}
                    className={`shrink-0 px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                      done
                        ? "bg-green-900 text-green-300 cursor-default"
                        : "bg-white text-black hover:bg-zinc-200 disabled:opacity-50"
                    }`}
                  >
                    {done ? "Imported" : isImporting ? "Importing..." : "Import"}
                  </button>
                </div>
              )
            })}
          </div>
        </div>
      ))}

      {imported.size > 0 && (
        <button
          onClick={onSuccess}
          className="w-full py-2 border border-zinc-700 text-sm text-zinc-300 rounded hover:bg-zinc-800"
        >
          Done — View Datasets
        </button>
      )}
    </div>
  )
}

// ── Modal shell ───────────────────────────────────────────────────────────────

interface Props {
  projectId: string
  onClose: () => void
  onSuccess: () => void
}

const TABS: { id: Tab; label: string; desc: string }[] = [
  { id: "upload",  label: "Upload File",  desc: "CSV / JSON / JSONL" },
  { id: "pdf",     label: "From PDF",     desc: "Extract + generate Q&A" },
  { id: "url",     label: "From URL",     desc: "Scrape + generate Q&A" },
  { id: "presets", label: "Presets",      desc: "Demo datasets" },
]

export function CreateDatasetModal({ projectId, onClose, onSuccess }: Props) {
  const [tab, setTab] = useState<Tab>("upload")

  function handleSuccess() {
    onSuccess()
    onClose()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/70" onClick={onClose} />

      {/* Modal */}
      <div className="relative z-10 w-full max-w-lg mx-4 bg-zinc-950 border border-zinc-800 rounded-lg shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800">
          <h3 className="font-semibold text-sm">Create Dataset</h3>
          <button onClick={onClose} className="text-zinc-500 hover:text-white text-lg leading-none">×</button>
        </div>

        {/* Tab bar */}
        <div className="flex border-b border-zinc-800">
          {TABS.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`flex-1 px-3 py-3 text-left transition-colors ${
                tab === t.id ? "bg-zinc-900 border-b-2 border-white" : "hover:bg-zinc-900/50"
              }`}
            >
              <p className={`text-xs font-medium ${tab === t.id ? "text-white" : "text-zinc-400"}`}>{t.label}</p>
              <p className="text-zinc-600 text-xs">{t.desc}</p>
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="p-5">
          {tab === "upload"   && <UploadTab   projectId={projectId} onSuccess={handleSuccess} />}
          {tab === "pdf"      && <PdfTab     projectId={projectId} onSuccess={handleSuccess} />}
          {tab === "url"      && <UrlTab     projectId={projectId} onSuccess={handleSuccess} />}
          {tab === "presets"  && <PresetsTab projectId={projectId} onSuccess={handleSuccess} />}
        </div>
      </div>
    </div>
  )
}
