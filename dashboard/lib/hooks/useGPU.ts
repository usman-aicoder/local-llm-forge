import { useQuery } from "@tanstack/react-query"

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8010"

export interface GPUStats {
  gpu_util_pct: number | null
  mem_used_gb: number | null
  mem_total_gb: number | null
}

export function useGPU(jobId: string | null, active: boolean) {
  return useQuery<GPUStats>({
    queryKey: ["gpu", jobId],
    queryFn: async () => {
      const res = await fetch(`${BASE}/jobs/${jobId}/gpu`)
      if (!res.ok) throw new Error("GPU stats unavailable")
      return res.json()
    },
    enabled: !!jobId && active,
    refetchInterval: active ? 5000 : false,
    staleTime: 0,
  })
}
