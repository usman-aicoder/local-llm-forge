"use client"

import { useState } from "react"
import { useBrowseHFModels } from "@/lib/hooks/useModels"

interface Props {
  onClose: () => void
  onSelect: (modelId: string) => void
}

export function ModelBrowserModal({ onClose, onSelect }: Props) {
  const [search, setSearch] = useState("")
  const [query, setQuery] = useState("")
  const { data: models, isLoading, isError } = useBrowseHFModels(query)

  function handleSearch(e: React.FormEvent) {
    e.preventDefault()
    setQuery(search)
  }

  function vramLabel(v: number | "unknown") {
    return v === "unknown" ? "?" : `${v} GB`
  }

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/70 pt-16 px-4">
      <div className="w-full max-w-3xl bg-zinc-950 border border-zinc-700 rounded-xl shadow-2xl flex flex-col max-h-[80vh]">

        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-zinc-800">
          <div>
            <h2 className="font-bold text-sm">HuggingFace Model Browser</h2>
            <p className="text-zinc-500 text-xs mt-0.5">Browse popular text-generation models. Select to fill the base model field.</p>
          </div>
          <button onClick={onClose} className="text-zinc-500 hover:text-white text-lg px-2">×</button>
        </div>

        {/* Search */}
        <div className="p-4 border-b border-zinc-800">
          <form onSubmit={handleSearch} className="flex gap-2">
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search models (e.g. llama, qwen, mistral)…"
              className="input flex-1"
            />
            <button type="submit" className="px-4 py-1.5 bg-white text-black text-xs font-medium rounded hover:bg-zinc-200">
              Search
            </button>
          </form>
        </div>

        {/* Results */}
        <div className="flex-1 overflow-y-auto">
          {isLoading && (
            <div className="p-8 text-center text-zinc-500 text-sm">Loading from HuggingFace…</div>
          )}
          {isError && (
            <div className="p-8 text-center text-zinc-500 text-sm">
              HuggingFace Hub unreachable. Check your internet connection.
            </div>
          )}
          {!isLoading && !isError && models && models.length === 0 && (
            <div className="p-8 text-center text-zinc-500 text-sm">No models found for "{query}".</div>
          )}
          {!isLoading && !isError && models && models.length > 0 && (
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-zinc-950 border-b border-zinc-800">
                <tr className="text-zinc-500">
                  <th className="text-left p-3 font-normal">Model</th>
                  <th className="text-right p-3 font-normal">Downloads</th>
                  <th className="text-center p-3 font-normal">QLoRA VRAM</th>
                  <th className="text-center p-3 font-normal">Full VRAM</th>
                  <th className="text-center p-3 font-normal">Status</th>
                  <th className="p-3"></th>
                </tr>
              </thead>
              <tbody>
                {models.map(m => (
                  <tr key={m.id} className="border-b border-zinc-900 hover:bg-zinc-900/50 transition-colors">
                    <td className="p-3">
                      <p className="text-white font-mono">{m.model_name}</p>
                      <p className="text-zinc-600 mt-0.5">{m.author}</p>
                    </td>
                    <td className="p-3 text-right text-zinc-400">
                      {m.downloads >= 1_000_000
                        ? `${(m.downloads / 1_000_000).toFixed(1)}M`
                        : m.downloads >= 1_000
                        ? `${(m.downloads / 1_000).toFixed(0)}K`
                        : m.downloads}
                    </td>
                    <td className="p-3 text-center">
                      <span className={`font-mono ${m.vram_estimate.qlora_gb === "unknown" ? "text-zinc-600" : "text-white"}`}>
                        {vramLabel(m.vram_estimate.qlora_gb)}
                      </span>
                    </td>
                    <td className="p-3 text-center">
                      <span className={`font-mono ${m.vram_estimate.full_lora_gb === "unknown" ? "text-zinc-600" : "text-zinc-400"}`}>
                        {vramLabel(m.vram_estimate.full_lora_gb)}
                      </span>
                    </td>
                    <td className="p-3 text-center">
                      {m.is_downloaded ? (
                        <span className="px-1.5 py-0.5 bg-green-900 text-green-400 rounded text-xs">downloaded</span>
                      ) : (
                        <span className="px-1.5 py-0.5 bg-zinc-800 text-zinc-500 rounded text-xs">hub</span>
                      )}
                    </td>
                    <td className="p-3 text-right">
                      <button
                        onClick={() => onSelect(m.id)}
                        className="px-3 py-1 border border-zinc-600 text-zinc-300 rounded hover:bg-zinc-800 text-xs"
                      >
                        Select
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {!query && !isLoading && (
            <div className="p-6 text-center text-zinc-600 text-sm">
              Enter a search term above to browse models, or click Search to see top downloads.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
