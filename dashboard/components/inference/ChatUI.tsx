"use client"

import {
  useState,
  useRef,
  useEffect,
  useCallback,
  forwardRef,
  useImperativeHandle,
} from "react"
import { streamGenerate } from "@/lib/hooks/useInference"

export interface Message {
  role: "user" | "assistant"
  content: string
}

export interface ChatHandle {
  send: (prompt: string) => void
  clear: () => void
  isStreaming: () => boolean
}

interface Props {
  model: string
  temperature: number
  maxTokens: number
  repeatPenalty: number
  /** Show an embedded input box (standalone mode) */
  showInput?: boolean
  /** Show model name as a header bar */
  showHeader?: boolean
}

export const ChatUI = forwardRef<ChatHandle, Props>(function ChatUI(
  { model, temperature, maxTokens, repeatPenalty, showInput = true, showHeader = false },
  ref
) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [streaming, setStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const streamingRef = useRef(false)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const send = useCallback(
    async (prompt: string) => {
      if (!prompt.trim() || !model || streamingRef.current) return
      setError(null)
      setMessages(prev => [...prev, { role: "user", content: prompt }])
      setMessages(prev => [...prev, { role: "assistant", content: "" }])
      setStreaming(true)
      streamingRef.current = true
      abortRef.current = new AbortController()

      await streamGenerate({
        model,
        prompt,
        temperature,
        maxTokens,
        repeatPenalty,
        signal: abortRef.current.signal,
        onToken: (token) => {
          setMessages(prev => {
            const next = [...prev]
            const last = next[next.length - 1]
            next[next.length - 1] = { ...last, content: last.content + token }
            return next
          })
        },
        onDone: () => { setStreaming(false); streamingRef.current = false },
        onError: (msg) => {
          setError(msg)
          setStreaming(false)
          streamingRef.current = false
        },
      })
    },
    [model, temperature, maxTokens, repeatPenalty]
  )

  const clear = useCallback(() => {
    abortRef.current?.abort()
    setMessages([])
    setStreaming(false)
    streamingRef.current = false
    setError(null)
  }, [])

  // Expose imperative handle for side-by-side parent
  useImperativeHandle(ref, () => ({
    send,
    clear,
    isStreaming: () => streamingRef.current,
  }), [send, clear])

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      const value = input.trim()
      setInput("")
      send(value)
    }
  }

  return (
    <div className="flex flex-col h-full min-h-0 border border-zinc-800 rounded bg-zinc-950">
      {/* Panel header */}
      {showHeader && (
        <div className="px-3 py-2 border-b border-zinc-800 flex items-center justify-between shrink-0">
          <span className="text-zinc-400 text-xs font-mono truncate max-w-[80%]">
            {model || <span className="text-zinc-600 italic">no model selected</span>}
          </span>
          <button onClick={clear} className="text-zinc-600 text-xs hover:text-zinc-400">
            clear
          </button>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3 min-h-0">
        {messages.length === 0 && (
          <p className="text-zinc-700 text-xs italic text-center mt-10">
            {model ? "Send a message to start" : "Select a model first"}
          </p>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[85%] rounded px-3 py-2 text-sm whitespace-pre-wrap break-words leading-relaxed ${
                msg.role === "user"
                  ? "bg-zinc-800 text-white"
                  : "bg-zinc-900 border border-zinc-800 text-zinc-200"
              }`}
            >
              {msg.content}
              {msg.role === "assistant" && streaming && i === messages.length - 1 && (
                <span className="inline-block w-1.5 h-3.5 bg-zinc-400 ml-0.5 animate-pulse align-middle" />
              )}
            </div>
          </div>
        ))}
        {error && <p className="text-red-400 text-xs text-center">{error}</p>}
        <div ref={bottomRef} />
      </div>

      {/* Embedded input (standalone mode) */}
      {showInput && (
        <div className="px-3 pb-3 pt-2 border-t border-zinc-800 shrink-0">
          <div className="flex gap-2 items-end">
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={model ? "Message… (Enter to send, Shift+Enter for newline)" : "Select a model above"}
              disabled={!model || streaming}
              rows={2}
              className="flex-1 bg-zinc-900 border border-zinc-700 rounded px-3 py-2 text-sm text-white outline-none focus:border-zinc-500 resize-none disabled:opacity-40 placeholder:text-zinc-600"
            />
            {streaming ? (
              <button
                onClick={() => { abortRef.current?.abort(); setStreaming(false); streamingRef.current = false }}
                className="px-3 py-2 border border-red-800 text-red-400 text-xs rounded hover:bg-red-950"
              >
                Stop
              </button>
            ) : (
              <button
                onClick={() => { const v = input.trim(); setInput(""); send(v) }}
                disabled={!input.trim() || !model}
                className="px-3 py-2 bg-white text-black text-xs font-medium rounded hover:bg-zinc-200 disabled:opacity-40"
              >
                Send
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
})
