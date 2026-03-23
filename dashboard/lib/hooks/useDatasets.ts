import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { api } from "@/lib/api"
import { useRouter } from "next/navigation"

export function useDatasets(projectId: string) {
  return useQuery({
    queryKey: ["datasets", projectId],
    queryFn: () => api.datasets.list(projectId),
    enabled: !!projectId,
  })
}

export function useDataset(id: string) {
  return useQuery({
    queryKey: ["datasets", "detail", id],
    queryFn: () => api.datasets.get(id),
    enabled: !!id,
  })
}

export function useDeleteDataset(projectId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api.datasets.delete(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["datasets", projectId] }),
  })
}

export function useFormatPreview(
  datasetId: string,
  formatType: string,
  baseModel: string,
  enabled: boolean
) {
  return useQuery({
    queryKey: ["datasets", "preview", datasetId, formatType, baseModel],
    queryFn: () => api.datasets.formatPreview(datasetId),
    enabled: enabled && !!datasetId,
  })
}
