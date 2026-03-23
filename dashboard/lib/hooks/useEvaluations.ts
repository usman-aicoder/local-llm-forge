import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { api } from "@/lib/api"
import type { Evaluation, EvalRow } from "@/lib/types"

export type { EvalRow }

export function useEvaluation(jobId: string | null) {
  return useQuery<Evaluation>({
    queryKey: ["evaluation", jobId],
    queryFn: () => api.evaluations.get(jobId!),
    enabled: !!jobId,
    retry: false,
  })
}

export function useRunAutoEval(jobId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => api.evaluations.runAuto(jobId),
    onSuccess: () => {
      // Poll for the evaluation result every 5s until it appears
      const interval = setInterval(async () => {
        try {
          await qc.fetchQuery({
            queryKey: ["evaluation", jobId],
            queryFn: () => api.evaluations.get(jobId),
            staleTime: 0,
          })
          qc.invalidateQueries({ queryKey: ["evaluation", jobId] })
          clearInterval(interval)
        } catch {
          // Not ready yet — keep polling
        }
      }, 5000)
    },
  })
}

export function useSubmitHuman(jobId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (results: Evaluation["sample_results"]) =>
      api.evaluations.submitHuman(jobId, results),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["evaluation", jobId] })
    },
  })
}

export function useProjectEvaluations(projectId: string) {
  return useQuery<{ evaluations: EvalRow[] }>({
    queryKey: ["evaluations", projectId],
    queryFn: () => api.evaluations.listForProject(projectId),
  })
}
