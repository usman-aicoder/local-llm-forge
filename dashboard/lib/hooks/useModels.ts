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
