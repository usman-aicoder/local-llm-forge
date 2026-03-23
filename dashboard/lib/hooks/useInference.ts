import { useQuery } from "@tanstack/react-query"

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8010"

export interface FineTunedModel {
  job_id: string
  job_name: string
  ollama_model_name: string
  base_model: string
}

export interface InferenceModels {
  base_models: { name: string; size: number; modified_at: string; digest: string }[]
  fine_tuned: FineTunedModel[]
}

export function useInferenceModels() {
  return useQuery<InferenceModels>({
    queryKey: ["inference-models"],
    queryFn: async () => {
      const res = await fetch(`${BASE}/inference/models`)
      if (!res.ok) throw new Error("Failed to load models")
      return res.json()
    },
    staleTime: 30_000,
  })
}

export interface GenerateOptions {
  model: string
  prompt: string
  temperature: number
  maxTokens: number
  repeatPenalty: number
  signal?: AbortSignal
  onToken: (token: string) => void
  onDone: () => void
  onError: (msg: string) => void
}

export async function streamGenerate(opts: GenerateOptions): Promise<void> {
  const { model, prompt, temperature, maxTokens, repeatPenalty, signal, onToken, onDone, onError } = opts

  let res: Response
  try {
    res = await fetch(`${BASE}/inference/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        prompt,
        temperature,
        max_tokens: maxTokens,
        repeat_penalty: repeatPenalty,
      }),
      signal,
    })
  } catch (e: unknown) {
    if ((e as Error).name !== "AbortError") onError(String(e))
    return
  }

  if (!res.ok) {
    onError(`HTTP ${res.status}`)
    return
  }

  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let buf = ""

  while (true) {
    let done: boolean
    let value: Uint8Array | undefined
    try {
      ;({ done, value } = await reader.read())
    } catch {
      break
    }
    if (done) break

    buf += decoder.decode(value, { stream: true })
    const lines = buf.split("\n")
    buf = lines.pop() ?? ""

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue
      const raw = line.slice(6).trim()
      try {
        const parsed = JSON.parse(raw)
        if (parsed.token) onToken(parsed.token)
        if (parsed.done) { onDone(); return }
        if (parsed.error) { onError(parsed.error); return }
      } catch {
        // ignore malformed lines
      }
    }
  }

  onDone()
}
