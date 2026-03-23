"use client"

import { useTaskStatus } from "@/lib/hooks/useTasks"

interface Props {
  taskId: string | null
  onComplete?: (result: Record<string, unknown>) => void
  label?: string
}

export function TaskRunner({ taskId, onComplete, label = "Processing" }: Props) {
  const { data: task } = useTaskStatus(taskId)

  if (!taskId) return null

  if (!task || task.status === "pending" || task.status === "running") {
    return (
      <div className="flex items-center gap-2 text-sm text-zinc-400 py-4">
        <span className="animate-spin text-lg">⟳</span>
        <span>{label}...</span>
      </div>
    )
  }

  if (task.status === "failed") {
    return (
      <div className="p-3 border border-red-800 rounded bg-red-950 text-red-300 text-xs font-mono">
        <p className="font-bold mb-1">Task failed</p>
        <pre className="whitespace-pre-wrap overflow-auto max-h-40">{task.error}</pre>
      </div>
    )
  }

  // completed — caller handles display of result
  if (task.status === "completed" && onComplete && task.result) {
    onComplete(task.result as Record<string, unknown>)
  }

  return null
}
