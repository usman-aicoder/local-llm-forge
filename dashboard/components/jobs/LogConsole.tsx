"use client"

import { useEffect, useRef } from "react"

interface Props {
  lines: string[]
  isLive?: boolean
  maxHeight?: string
}

export function LogConsole({ lines, isLive = false, maxHeight = "h-64" }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const autoScroll = useRef(true)

  // Auto-scroll only when user is near the bottom
  useEffect(() => {
    if (!autoScroll.current) return
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [lines])

  function handleScroll() {
    const el = containerRef.current
    if (!el) return
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80
    autoScroll.current = nearBottom
  }

  return (
    <div className="border border-zinc-800 rounded bg-zinc-950">
      {/* Header */}
      <div className="px-3 py-2 border-b border-zinc-800 flex items-center justify-between">
        <p className="text-zinc-500 text-xs font-mono">training logs</p>
        <div className="flex items-center gap-3">
          <span className="text-zinc-700 text-xs">{lines.length} lines</span>
          {isLive && (
            <span className="flex items-center gap-1 text-blue-400 text-xs">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
              live
            </span>
          )}
        </div>
      </div>

      {/* Log body */}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className={`${maxHeight} overflow-y-auto p-3 font-mono text-xs text-zinc-400 space-y-px`}
      >
        {lines.length === 0 ? (
          <p className="text-zinc-700 italic">
            {isLive ? "Waiting for output..." : "No logs."}
          </p>
        ) : (
          lines.map((line, i) => (
            <p
              key={i}
              className={[
                "leading-5 whitespace-pre-wrap break-all",
                line.toLowerCase().includes("error") ? "text-red-400" :
                line.toLowerCase().includes("epoch") ? "text-white" :
                line.toLowerCase().includes("warning") ? "text-yellow-400" : "",
              ].join(" ")}
            >
              {line}
            </p>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
