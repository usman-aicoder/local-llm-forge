"use client"

import { useRef, useState } from "react"
import { useParams } from "next/navigation"
import {
  useCollections,
  useCreateCollection,
  useDeleteCollection,
  useDocuments,
  useUploadDocument,
  useDeleteDocument,
} from "@/lib/hooks/useRag"
import { useInferenceModels } from "@/lib/hooks/useInference"
import type { RAGCollection, RAGDocument } from "@/lib/types"

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8010"

// ── Status badge ─────────────────────────────────────────────────────────────
function StatusBadge({ status }: { status: RAGDocument["status"] }) {
  const styles: Record<string, string> = {
    uploaded:   "bg-zinc-700 text-zinc-300",
    processing: "bg-yellow-900 text-yellow-300 animate-pulse",
    indexed:    "bg-green-900 text-green-300",
    failed:     "bg-red-900 text-red-300",
  }
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${styles[status] ?? styles.uploaded}`}>
      {status}
    </span>
  )
}

// ── Source citation ───────────────────────────────────────────────────────────
function Source({ text, score, idx }: { text: string; score: number; idx: number }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="border border-zinc-700 rounded text-xs">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-3 py-1.5 text-left text-zinc-400 hover:text-white"
      >
        <span>Source {idx + 1} — score {score}</span>
        <span>{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <p className="px-3 pb-2 text-zinc-500 leading-relaxed whitespace-pre-wrap border-t border-zinc-800">
          {text}
        </p>
      )}
    </div>
  )
}

// ── Chat message ─────────────────────────────────────────────────────────────
interface Message {
  role: "user" | "assistant"
  text: string
  sources?: Array<{ text: string; score: number; document_id: string; chunk_index: number }>
}

// ── Chat panel ────────────────────────────────────────────────────────────────
function ChatPanel({ col, models }: { col: RAGCollection; models: string[] }) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [model, setModel] = useState(models[0] ?? "llama3.2:latest")
  const [streaming, setStreaming] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  async function send() {
    const q = input.trim()
    if (!q || streaming) return
    setInput("")
    setMessages(m => [...m, { role: "user", text: q }])

    const ab = new AbortController()
    abortRef.current = ab
    setStreaming(true)

    let sources: Message["sources"] = []
    let answer = ""
    const idx = messages.length + 1

    setMessages(m => [...m, { role: "assistant", text: "", sources: [] }])

    try {
      const res = await fetch(`${BASE}/rag/collections/${col.id}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, model, top_k: 5 }),
        signal: ab.signal,
      })

      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let buf = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        const lines = buf.split("\n")
        buf = lines.pop() ?? ""
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue
          try {
            const ev = JSON.parse(line.slice(6))
            if (ev.sources) {
              sources = ev.sources
              setMessages(m => m.map((msg, i) => i === idx ? { ...msg, sources } : msg))
            }
            if (ev.token) {
              answer += ev.token
              setMessages(m => m.map((msg, i) => i === idx ? { ...msg, text: answer } : msg))
              bottomRef.current?.scrollIntoView({ behavior: "smooth" })
            }
            if (ev.done || ev.error) break
          } catch {}
        }
      }
    } catch {}
    setStreaming(false)
  }

  return (
    <div className="flex flex-col h-full">
      {/* Model selector */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-zinc-500 text-xs">Model:</span>
        <select
          value={model}
          onChange={e => setModel(e.target.value)}
          className="input text-xs"
        >
          {models.map(m => <option key={m} value={m}>{m}</option>)}
        </select>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-1 min-h-0">
        {messages.length === 0 && (
          <p className="text-zinc-600 text-sm text-center pt-12">
            Ask a question about your documents.
          </p>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`space-y-2 ${msg.role === "user" ? "text-right" : "text-left"}`}>
            <div className={`inline-block max-w-[85%] px-3 py-2 rounded-lg text-sm text-left whitespace-pre-wrap ${
              msg.role === "user"
                ? "bg-zinc-700 text-white"
                : "bg-zinc-900 border border-zinc-800 text-zinc-200"
            }`}>
              {msg.text || (streaming && i === messages.length - 1 ? (
                <span className="inline-block w-2 h-4 bg-zinc-400 animate-pulse rounded-sm" />
              ) : "...")}
            </div>
            {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (
              <div className="space-y-1 max-w-[85%]">
                {msg.sources.map((s, si) => (
                  <Source key={si} text={s.text} score={s.score} idx={si} />
                ))}
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="flex gap-2 mt-3 pt-3 border-t border-zinc-800">
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send() } }}
          placeholder="Ask a question... (Enter to send)"
          rows={2}
          className="input flex-1 resize-none text-sm"
          disabled={streaming}
        />
        {streaming ? (
          <button
            onClick={() => { abortRef.current?.abort(); setStreaming(false) }}
            className="px-3 py-1 bg-red-900 text-red-300 border border-red-700 rounded text-xs"
          >
            Stop
          </button>
        ) : (
          <button
            onClick={send}
            disabled={!input.trim()}
            className="px-4 py-1 bg-white text-black rounded text-sm font-medium disabled:opacity-40"
          >
            Send
          </button>
        )}
      </div>
    </div>
  )
}

// ── Document list ─────────────────────────────────────────────────────────────
function DocList({ col }: { col: RAGCollection }) {
  const { data, isLoading } = useDocuments(col.id)
  const upload = useUploadDocument(col.id)
  const del = useDeleteDocument(col.id)
  const fileRef = useRef<HTMLInputElement>(null)

  const docs = data?.documents ?? []

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-zinc-400 text-xs font-medium uppercase tracking-wider">
          Documents ({docs.length})
        </span>
        <button
          onClick={() => fileRef.current?.click()}
          disabled={upload.isPending}
          className="text-xs px-2 py-1 border border-zinc-700 rounded hover:border-zinc-500 text-zinc-400 hover:text-white transition-colors disabled:opacity-50"
        >
          {upload.isPending ? "Uploading..." : "+ Upload"}
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.txt,.md"
          className="hidden"
          onChange={e => {
            const f = e.target.files?.[0]
            if (f) upload.mutate(f)
            e.target.value = ""
          }}
        />
      </div>

      {isLoading && <p className="text-zinc-600 text-xs">Loading...</p>}

      {docs.length === 0 && !isLoading && (
        <p className="text-zinc-600 text-xs">No documents yet. Upload a PDF, TXT, or MD file.</p>
      )}

      <div className="space-y-1">
        {docs.map(doc => (
          <div
            key={doc.id}
            className="flex items-center justify-between gap-2 px-3 py-2 bg-zinc-900 border border-zinc-800 rounded"
          >
            <div className="flex items-center gap-2 min-w-0">
              <StatusBadge status={doc.status} />
              <span className="text-xs text-zinc-300 truncate">{doc.filename}</span>
              {doc.chunk_count !== null && (
                <span className="text-zinc-600 text-xs shrink-0">{doc.chunk_count} chunks</span>
              )}
            </div>
            <button
              onClick={() => {
                if (confirm(`Delete "${doc.filename}"? This will remove the file and its vectors from the collection.`))
                  del.mutate(doc.id)
              }}
              className="text-zinc-600 hover:text-red-400 text-xs shrink-0"
              title="Delete document"
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function RagPage() {
  const { id } = useParams<{ id: string }>()
  const { data: colData, isLoading: loadingCols } = useCollections(id)
  const { data: modelsData } = useInferenceModels()
  const createCol = useCreateCollection(id)
  const deleteCol = useDeleteCollection(id)

  const [selectedColId, setSelectedColId] = useState<string | null>(null)
  const [newColName, setNewColName] = useState("")
  const [showNew, setShowNew] = useState(false)

  const collections = colData?.collections ?? []
  const selectedCol = collections.find(c => c.id === selectedColId) ?? collections[0] ?? null

  const allModels = [
    ...(modelsData?.base_models?.map((m: { name: string }) => m.name) ?? []),
    ...(modelsData?.fine_tuned?.map((m: { ollama_model_name: string }) => m.ollama_model_name) ?? []),
  ].filter(Boolean)

  async function handleCreate() {
    if (!newColName.trim()) return
    const col = await createCol.mutateAsync({ name: newColName.trim() })
    setSelectedColId(col.id)
    setNewColName("")
    setShowNew(false)
  }

  return (
    <div>
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold">RAG</h2>
          <p className="text-zinc-500 text-sm mt-0.5">
            Upload documents and ask questions grounded in your content.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-[220px_1fr] gap-4 h-[calc(100vh-200px)]">

        {/* ── Left: collection list ── */}
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between mb-1">
            <span className="text-zinc-500 text-xs uppercase tracking-wider font-medium">Collections</span>
            <button
              onClick={() => setShowNew(v => !v)}
              className="text-xs text-zinc-500 hover:text-white"
            >
              + New
            </button>
          </div>

          {showNew && (
            <div className="flex gap-1">
              <input
                value={newColName}
                onChange={e => setNewColName(e.target.value)}
                onKeyDown={e => e.key === "Enter" && handleCreate()}
                placeholder="Collection name"
                autoFocus
                className="input text-xs flex-1"
              />
              <button
                onClick={handleCreate}
                disabled={!newColName.trim() || createCol.isPending}
                className="px-2 py-1 bg-white text-black text-xs rounded disabled:opacity-50"
              >
                Add
              </button>
            </div>
          )}

          {loadingCols && <p className="text-zinc-600 text-xs">Loading...</p>}

          {collections.length === 0 && !loadingCols && (
            <p className="text-zinc-600 text-xs">No collections yet.</p>
          )}

          {collections.map(col => (
            <div
              key={col.id}
              onClick={() => setSelectedColId(col.id)}
              className={`group flex items-center justify-between px-3 py-2 rounded cursor-pointer border transition-colors ${
                (selectedColId === col.id || (!selectedColId && col === collections[0]))
                  ? "bg-zinc-800 border-zinc-700 text-white"
                  : "border-transparent text-zinc-400 hover:bg-zinc-900 hover:text-white"
              }`}
            >
              <div className="min-w-0">
                <p className="text-sm font-medium truncate">{col.name}</p>
                <p className="text-zinc-600 text-xs">{col.document_count} docs</p>
              </div>
              <button
                onClick={e => { e.stopPropagation(); deleteCol.mutate(col.id) }}
                className="text-zinc-700 hover:text-red-400 text-xs opacity-0 group-hover:opacity-100"
              >
                ✕
              </button>
            </div>
          ))}
        </div>

        {/* ── Right: doc list + chat ── */}
        {selectedCol ? (
          <div className="grid grid-rows-[auto_1fr] gap-4 min-h-0">
            {/* Doc list */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
              <DocList col={selectedCol} />
            </div>

            {/* Chat */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 flex flex-col min-h-0">
              <h3 className="text-zinc-400 text-xs font-medium uppercase tracking-wider mb-3">
                Chat with {selectedCol.name}
              </h3>
              {allModels.length === 0 ? (
                <p className="text-zinc-600 text-sm">
                  Start Ollama and load a model to use RAG chat.
                </p>
              ) : (
                <div className="flex-1 min-h-0">
                  <ChatPanel col={selectedCol} models={allModels} />
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center text-zinc-600 text-sm">
            {collections.length === 0
              ? "Create a collection to get started."
              : "Select a collection."}
          </div>
        )}
      </div>
    </div>
  )
}
