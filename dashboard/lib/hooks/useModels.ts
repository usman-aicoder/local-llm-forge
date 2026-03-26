import { useQuery } from "@tanstack/react-query"
import { api } from "@/lib/api"

export function useOllamaModels() {
  return useQuery({
    queryKey: ["models", "ollama"],
    queryFn: () => api.models.ollama(),
    staleTime: 30_000,
  })
}

export function useHFModels() {
  return useQuery({
    queryKey: ["models", "hf"],
    queryFn: () => api.models.hf(),
    staleTime: 30_000,
  })
}

export function useBrowseHFModels(search: string, enabled = true) {
  return useQuery({
    queryKey: ["models", "browse", search],
    queryFn: () => api.models.browse(search, 30),
    enabled,
    staleTime: 60_000,
    retry: 1,
  })
}

export function useSystemCapabilities() {
  return useQuery({
    queryKey: ["system", "capabilities"],
    queryFn: () => api.system.capabilities(),
    staleTime: 60_000,
  })
}
