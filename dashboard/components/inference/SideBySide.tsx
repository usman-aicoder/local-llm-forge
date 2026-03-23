"use client"

import { useState, useRef } from "react"
import { ChatUI, type ChatHandle } from "./ChatUI"
import type { InferenceModels } from "@/lib/hooks/useInference"

interface Props {
  models: InferenceModels
  temperature: number
  maxTokens: number
  repeatPenalty: number
}

function ModelSelect({
  value,
  onChange,
  models,
  label,
}: {
  value: string
  onChange: (v: string) => void
  models: InferenceModels
  label: string
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-zinc-500 text-xs shrink-0">{label}</span>
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="flex-1 bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs text-white outline-none focus:border-zinc-500"
      >
        <option value="">— select —</option>
        {models.base_models.length > 0 && (
          <optgroup label="Base Models (Ollama)">
            {models.base_models.map(m => (
              <option key={m.name} value={m.name}>{m.name}</option>
            ))}
          </optgroup>
        )}
        {models.fine_tuned.length > 0 && (
          <optgroup label="Fine-Tuned Models">
            {models.fine_tuned.map(m => (
              <option key={m.ollama_model_name} value={m.ollama_model_name}>
                {m.job_name} ({m.ollama_model_name})
              </option>
            ))}
          </optgroup>
        )}
      </select>
    </div>
  )
}

export function SideBySide({ models, temperature, maxTokens, repeatPenalty }: Props) {
  const [modelLeft, setModelLeft]   = useState("")
  const [modelRight, setModelRight] = useState("")
  const [sharedInput, setSharedInput] = useState("")
  const [sending, setSending] = useState(false)

  const leftRef  = useRef<ChatHandle>(null)
  const rightRef = useRef<ChatHandle>(null)

  async function handleSend() {
    const prompt = sharedInput.trim()
    if (!prompt || sending) return
    setSending(true)
    setSharedInput("")
    // Fire both panels simultaneously — they stream independently
    leftRef.current?.send(prompt)
    rightRef.current?.send(prompt)
    // Wait a tick so streaming starts before we re-enable the button
    setTimeout(() => setSending(false), 300)
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  function handleClearAll() {
    leftRef.current?.clear()
    rightRef.current?.clear()
  }

  return (
    <div className="flex flex-col h-full gap-3">
      {/* Model selectors */}
      <div className="grid grid-cols-2 gap-3">
        <ModelSelect value={modelLeft}  onChange={setModelLeft}  models={models} label="Left" />
        <ModelSelect value={modelRight} onChange={setModelRight} models={models} label="Right" />
      </div>

      {/* Panels */}
      <div className="grid grid-cols-2 gap-3 flex-1 min-h-0">
        <ChatUI
          ref={leftRef}
          model={modelLeft}
          temperature={temperature}
          maxTokens={maxTokens}
          repeatPenalty={repeatPenalty}
          showInput={false}
          showHeader
        />
        <ChatUI
          ref={rightRef}
          model={modelRight}
          temperature={temperature}
          maxTokens={maxTokens}
          repeatPenalty={repeatPenalty}
          showInput={false}
          showHeader
        />
      </div>

      {/* Shared input */}
      <div className="flex gap-2 items-end shrink-0">
        <textarea
          value={sharedInput}
          onChange={e => setSharedInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Send the same prompt to both models… (Enter to send)"
          rows={2}
          className="flex-1 bg-zinc-900 border border-zinc-700 rounded px-3 py-2 text-sm text-white outline-none focus:border-zinc-500 resize-none placeholder:text-zinc-600"
        />
        <div className="flex flex-col gap-1 shrink-0">
          <button
            onClick={handleSend}
            disabled={!sharedInput.trim() || (!modelLeft && !modelRight) || sending}
            className="px-4 py-2 bg-white text-black text-xs font-medium rounded hover:bg-zinc-200 disabled:opacity-40"
          >
            Send Both
          </button>
          <button
            onClick={handleClearAll}
            className="px-4 py-1 border border-zinc-700 text-zinc-500 text-xs rounded hover:bg-zinc-800"
          >
            Clear All
          </button>
        </div>
      </div>
    </div>
  )
}
