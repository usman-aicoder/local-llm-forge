import { useQuery } from "@tanstack/react-query"
import { api } from "@/lib/api"

export interface TaskStatus {
  id: string
  status: "pending" | "running" | "completed" | "failed"
  result: Record<string, unknown> | null
  error: string | null
  created_at: string
  updated_at: string
}

export function useTaskStatus(taskId: string | null) {
  return useQuery({
    queryKey: ["tasks", taskId],
    queryFn: async () => {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/tasks/${taskId}`
      )
      if (!res.ok) throw new Error("Task not found")
      return res.json() as Promise<TaskStatus>
    },
    enabled: !!taskId,
    refetchInterval: (query) => {
      const data = query.state.data as TaskStatus | undefined
      if (!data) return 1500
      if (data.status === "completed" || data.status === "failed") return false
      return 1500   // poll every 1.5 s while running
    },
  })
}
