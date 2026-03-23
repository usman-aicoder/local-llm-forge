"use client"

import { useEffect, useRef, useCallback } from "react"

export interface ProgressPoint {
  epoch: number
  train_loss: number
  eval_loss: number
  perplexity: number
}

interface Options {
  onProgress: (p: ProgressPoint) => void
  onLog: (line: string) => void
  onDone: (status: string) => void
  onError?: (err: Event) => void
  enabled?: boolean
}

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8010"
const RECONNECT_DELAY_MS = 3000
const MAX_RECONNECTS = 10

/**
 * Opens an SSE connection to /jobs/{jobId}/stream.
 * Auto-reconnects on network drops up to MAX_RECONNECTS times.
 * Closes automatically when a "done" event is received.
 */
export function useJobStream(jobId: string | null, options: Options) {
  const { onProgress, onLog, onDone, onError, enabled = true } = options
  const esRef     = useRef<EventSource | null>(null)
  const reconnects = useRef(0)
  const closed     = useRef(false)

  // Stable callback refs so the effect doesn't re-run on every render
  const onProgressRef = useRef(onProgress)
  const onLogRef      = useRef(onLog)
  const onDoneRef     = useRef(onDone)
  const onErrorRef    = useRef(onError)

  useEffect(() => { onProgressRef.current = onProgress }, [onProgress])
  useEffect(() => { onLogRef.current = onLog }, [onLog])
  useEffect(() => { onDoneRef.current = onDone }, [onDone])
  useEffect(() => { onErrorRef.current = onError }, [onError])

  const connect = useCallback(() => {
    if (!jobId || !enabled || closed.current) return

    const url = `${BASE}/jobs/${jobId}/stream`
    const es  = new EventSource(url)
    esRef.current = es

    es.addEventListener("progress", (e: MessageEvent) => {
      try { onProgressRef.current(JSON.parse(e.data)) } catch {}
    })

    es.addEventListener("log", (e: MessageEvent) => {
      try { onLogRef.current(JSON.parse(e.data).line) } catch {}
    })

    es.addEventListener("done", (e: MessageEvent) => {
      try {
        closed.current = true
        onDoneRef.current(JSON.parse(e.data).status)
      } catch {}
      es.close()
    })

    es.onerror = (evt) => {
      es.close()
      if (onErrorRef.current) onErrorRef.current(evt)
      if (!closed.current && reconnects.current < MAX_RECONNECTS) {
        reconnects.current++
        setTimeout(connect, RECONNECT_DELAY_MS)
      }
    }
  }, [jobId, enabled])

  useEffect(() => {
    if (!jobId || !enabled) return
    closed.current = false
    reconnects.current = 0
    connect()

    return () => {
      closed.current = true
      esRef.current?.close()
    }
  }, [jobId, enabled, connect])
}
