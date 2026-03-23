import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { api } from "@/lib/api"
import type { TrainingJob } from "@/lib/types"

export function useJobs(projectId: string) {
  return useQuery({
    queryKey: ["jobs", projectId],
    queryFn: () => api.jobs.list(projectId),
    enabled: !!projectId,
    refetchInterval: (query) => {
      const jobs = (query.state.data as { jobs: TrainingJob[] } | undefined)?.jobs ?? []
      const hasActive = jobs.some(j => j.status === "running" || j.status === "queued")
      return hasActive ? 5000 : false
    },
  })
}

export function useJob(jobId: string) {
  return useQuery({
    queryKey: ["jobs", "detail", jobId],
    queryFn: () => api.jobs.get(jobId),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const job = query.state.data as TrainingJob | undefined
      if (!job) return 3000
      return job.status === "running" || job.status === "queued" ? 5000 : false
    },
  })
}

export function useDeleteJob(projectId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api.jobs.delete(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs", projectId] }),
  })
}

export function useCheckpoints(jobId: string) {
  return useQuery({
    queryKey: ["checkpoints", jobId],
    queryFn: () => api.jobs.checkpoints(jobId),
    enabled: !!jobId,
  })
}

export function useJobStream(jobId: string | null, onProgress: (data: unknown) => void, onLog: (line: string) => void, onDone: (status: string) => void) {
  // Returns a cleanup function — call in useEffect
  if (!jobId) return () => {}

  const url = `${process.env.NEXT_PUBLIC_API_URL}/jobs/${jobId}/stream`
  const es = new EventSource(url)

  es.addEventListener("progress", (e) => {
    try { onProgress(JSON.parse(e.data)) } catch {}
  })
  es.addEventListener("log", (e) => {
    try { onLog(JSON.parse(e.data).line) } catch {}
  })
  es.addEventListener("done", (e) => {
    try { onDone(JSON.parse(e.data).status) } catch {}
    es.close()
  })
  es.onerror = () => es.close()

  return () => es.close()
}
