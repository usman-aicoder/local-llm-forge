import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { api } from "@/lib/api"
import type { RAGCollection, RAGDocument } from "@/lib/types"

export function useCollections(projectId: string) {
  return useQuery({
    queryKey: ["rag-collections", projectId],
    queryFn: () => api.rag.listCollections(projectId),
    enabled: !!projectId,
  })
}

export function useCreateCollection(projectId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: { name: string; embedding_model?: string }) =>
      api.rag.createCollection(projectId, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["rag-collections", projectId] }),
  })
}

export function useDeleteCollection(projectId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (colId: string) => api.rag.deleteCollection(colId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["rag-collections", projectId] }),
  })
}

export function useDocuments(colId: string | null) {
  return useQuery({
    queryKey: ["rag-documents", colId],
    queryFn: () => api.rag.listDocuments(colId!),
    enabled: !!colId,
    refetchInterval: (query) => {
      const docs = (query.state.data as { documents: RAGDocument[] } | undefined)?.documents ?? []
      return docs.some((d) => d.status === "uploaded" || d.status === "processing") ? 3000 : false
    },
  })
}

export function useUploadDocument(colId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (file: File) => {
      const form = new FormData()
      form.append("file", file)
      return api.rag.uploadDocument(colId, form)
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["rag-documents", colId] }),
  })
}

export function useDeleteDocument(colId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (docId: string) => api.rag.deleteDocument(docId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["rag-documents", colId] }),
  })
}
