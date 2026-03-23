"use client"

import { useState, useEffect } from "react"
import { useParams, useRouter } from "next/navigation"
import { useDatasets } from "@/lib/hooks/useDatasets"
import { useHFModels, useOllamaModels } from "@/lib/hooks/useModels"

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8010"

const METHODS = [
  {
    id: "sft",
    label: "SFT",
    name: "Supervised Fine-Tuning",
    desc: "Train on instruction/response pairs. Requires tokenized dataset.",
    datasetStatus: "tokenized",
    defaultLR: 0.0002,
  },
  {
    id: "dpo",
    label: "DPO",
    name: "Direct Preference Optimization",
    desc: "Train on prompt/chosen/rejected pairs. Teaches model to prefer good answers over bad ones.",
    datasetStatus: "formatted",
    defaultLR: 0.00005,
  },
  {
    id: "orpo",
    label: "ORPO",
    name: "Odds Ratio Preference Optimization",
    desc: "Combines SFT + DPO in one pass. No reference model needed. Best efficiency.",
    datasetStatus: "formatted",
    defaultLR: 0.000008,
  },
] as const

type Method = typeof METHODS[number]["id"]

// ── Guide content per method ──────────────────────────────────────────────────

const GUIDES: Record<Method, {
  when: string
  how: string
  dataset: string
  lora: string
  hyperparams: string
  tips: string[]
}> = {
  sft: {
    when: "Use SFT when you want to teach the model a new task, domain knowledge, or writing style from scratch. It's the simplest and most common fine-tuning method.",
    how: "SFT works by showing the model many (instruction → correct response) pairs and training it to produce those responses. The model learns by minimizing the difference between its output and the correct answer token-by-token.",
    dataset: "Requires a tokenized dataset with instruction and output columns. Run your dataset through Clean → Format (Alpaca or Chat) → Tokenize before creating this job. Row count: 200–10,000 rows is a good range.",
    lora: "Rank 16 is a good default. Higher rank (32–64) lets the model learn more complex changes but uses more VRAM. Alpha should be 2× rank. Dropout 0.05 prevents overfitting. Enable QLoRA for <24 GB VRAM — it uses 4-bit weights to cut memory by ~4×.",
    hyperparams: "Learning rate 0.0002 is standard for SFT. Use 3 epochs for most datasets — more risks overfitting. Batch size 1–2 on 8 GB GPU. Grad accum × batch = effective batch (default 16). Max seq length 1024 is safe for 8 GB; use 512 if OOM.",
    tips: [
      "Start with default settings and only adjust if loss doesn't decrease",
      "If eval loss goes up while train loss goes down → overfitting, reduce epochs",
      "If loss barely moves → increase learning rate or check dataset quality",
      "QLoRA is almost always worth enabling — quality loss is minimal",
    ],
  },
  dpo: {
    when: "Use DPO when your model already knows the task but you want to steer its style, tone, or safety. Great for: making outputs more concise, improving refusals, fixing repetitive phrasing.",
    how: "DPO trains using pairs of (chosen answer, rejected answer) for the same prompt. It increases the probability of the chosen answer relative to the rejected one. Internally it compares outputs to a reference model (your base model via PEFT) to stay grounded.",
    dataset: "Requires a DPO-formatted dataset with prompt, chosen, and rejected columns. Run Clean → Format (DPO/Preference). Do NOT tokenize — DPO trainer handles that internally. Row count: 100–5,000 pairs is enough.",
    lora: "Same LoRA settings as SFT work well. Keep rank low (8–16) for DPO — you want subtle steering, not large changes. QLoRA recommended. The reference model is implicit (your LoRA base) so no extra memory is needed.",
    hyperparams: "Learning rate must be much lower than SFT — use 0.00005 (5e-5). Too high and the model destabilizes. 1–2 epochs is usually enough. Batch size 1 on 8 GB GPU. The beta parameter (not exposed here) controls how strongly to enforce preferences — default 0.1 is fine.",
    tips: [
      "DPO is sensitive to learning rate — if outputs get worse, halve the LR",
      "The quality of chosen/rejected pairs matters more than quantity",
      "Chosen and rejected answers should differ meaningfully — not just punctuation",
      "Run evaluation after to confirm the steered behavior improved",
    ],
  },
  orpo: {
    when: "Use ORPO when you want preference alignment but have limited data or VRAM. It combines supervised learning and preference optimization in one pass, so it's more data-efficient than DPO and doesn't need a reference model.",
    how: "ORPO adds an odds-ratio penalty to the standard language model loss. In a single forward pass, it simultaneously learns to produce the chosen answer (SFT signal) and to avoid the rejected answer (preference signal). No separate reference model is needed.",
    dataset: "Same as DPO: requires prompt, chosen, rejected columns. Run Clean → Format (DPO/Preference). Do NOT tokenize. Row count: ORPO is effective with as few as 50–500 pairs because the SFT signal helps stabilize training.",
    lora: "Same LoRA settings as DPO. Rank 8–16 recommended. QLoRA recommended. Because there's no reference model, ORPO uses slightly less VRAM than DPO for the same configuration.",
    hyperparams: "Learning rate is much lower than SFT — 0.000008 (8e-6) default. This is lower than DPO because the combined loss has a larger gradient. 1–3 epochs. Batch size 1. The orpo_alpha parameter (default 0.1) controls the strength of the preference penalty — increase to 0.5 if preference learning is weak.",
    tips: [
      "ORPO is usually better than DPO when your dataset is small (<500 pairs)",
      "If model outputs become incoherent → lower learning rate further (try 5e-6)",
      "ORPO preserves the original task capability better than DPO alone",
      "Good first choice if unsure between DPO and ORPO",
    ],
  },
}

// ── Guide panel component ─────────────────────────────────────────────────────

function GuidePanel({ method, vram }: { method: Method; vram: { total_gb: number; warning: string | null } | null }) {
  const g = GUIDES[method]
  const m = METHODS.find(x => x.id === method)!

  return (
    <div className="sticky top-6 space-y-4 text-sm">
      {/* Method header */}
      <div className="p-4 border border-zinc-700 rounded bg-zinc-900">
        <div className="flex items-center gap-2 mb-2">
          <span className="px-2 py-0.5 bg-white text-black text-xs font-bold rounded">{m.label}</span>
          <span className="text-white font-medium">{m.name}</span>
        </div>
        <p className="text-zinc-400 text-xs leading-relaxed">{g.when}</p>
      </div>

      {/* How it works */}
      <GuideSection title="How it works">
        <p className="text-zinc-400 text-xs leading-relaxed">{g.how}</p>
      </GuideSection>

      {/* Dataset */}
      <GuideSection title="Dataset requirements">
        <p className="text-zinc-400 text-xs leading-relaxed">{g.dataset}</p>
        <div className="mt-2 flex items-center gap-2 text-xs">
          <span className="text-zinc-500">Required status:</span>
          <span className="px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300 font-mono">{m.datasetStatus}</span>
        </div>
      </GuideSection>

      {/* LoRA */}
      <GuideSection title="LoRA / QLoRA settings">
        <p className="text-zinc-400 text-xs leading-relaxed">{g.lora}</p>
        <div className="mt-2 grid grid-cols-3 gap-1 text-xs">
          <div className="p-1.5 bg-zinc-800 rounded text-center">
            <div className="text-zinc-500">Rank</div>
            <div className="text-white font-mono">{method === "sft" ? "16" : "8–16"}</div>
          </div>
          <div className="p-1.5 bg-zinc-800 rounded text-center">
            <div className="text-zinc-500">Alpha</div>
            <div className="text-white font-mono">{method === "sft" ? "32" : "16–32"}</div>
          </div>
          <div className="p-1.5 bg-zinc-800 rounded text-center">
            <div className="text-zinc-500">Dropout</div>
            <div className="text-white font-mono">0.05</div>
          </div>
        </div>
      </GuideSection>

      {/* Hyperparams */}
      <GuideSection title="Hyperparameter guide">
        <p className="text-zinc-400 text-xs leading-relaxed">{g.hyperparams}</p>
        <div className="mt-2 grid grid-cols-2 gap-1 text-xs">
          <div className="p-1.5 bg-zinc-800 rounded">
            <div className="text-zinc-500">Learning Rate</div>
            <div className="text-white font-mono">{m.defaultLR}</div>
          </div>
          <div className="p-1.5 bg-zinc-800 rounded">
            <div className="text-zinc-500">Epochs</div>
            <div className="text-white font-mono">{method === "sft" ? "3–5" : "1–3"}</div>
          </div>
          <div className="p-1.5 bg-zinc-800 rounded">
            <div className="text-zinc-500">Batch Size (8 GB)</div>
            <div className="text-white font-mono">1</div>
          </div>
          <div className="p-1.5 bg-zinc-800 rounded">
            <div className="text-zinc-500">Max Seq Len</div>
            <div className="text-white font-mono">512–1024</div>
          </div>
        </div>
      </GuideSection>

      {/* Tips */}
      <GuideSection title="Tips">
        <ul className="space-y-1.5">
          {g.tips.map((tip, i) => (
            <li key={i} className="flex gap-2 text-xs text-zinc-400">
              <span className="text-zinc-600 shrink-0">→</span>
              <span className="leading-relaxed">{tip}</span>
            </li>
          ))}
        </ul>
      </GuideSection>

      {/* VRAM */}
      {vram && (
        <div className={`p-3 rounded border text-xs ${vram.warning ? "border-yellow-700 bg-yellow-950 text-yellow-300" : "border-zinc-700 bg-zinc-900 text-zinc-400"}`}>
          <div className="flex items-center justify-between mb-1">
            <span>Estimated VRAM</span>
            <span className="text-white font-bold font-mono">{vram.total_gb} GB</span>
          </div>
          {vram.warning
            ? <p>{vram.warning}</p>
            : <p className="text-green-400">Fits within available GPU memory.</p>
          }
        </div>
      )}

      {/* Compare table */}
      <GuideSection title="Method comparison">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="text-zinc-500">
              <th className="text-left pb-1 font-normal"></th>
              <th className="text-center pb-1 font-normal">SFT</th>
              <th className="text-center pb-1 font-normal">DPO</th>
              <th className="text-center pb-1 font-normal">ORPO</th>
            </tr>
          </thead>
          <tbody className="text-zinc-400">
            {[
              ["Dataset", "instruction/output", "prompt/chosen/rejected", "prompt/chosen/rejected"],
              ["Ref model", "None", "Implicit (PEFT)", "None"],
              ["VRAM", "Low", "Medium", "Low"],
              ["Data needed", "200+ rows", "100+ pairs", "50+ pairs"],
              ["Best for", "New tasks", "Style/safety", "Alignment + efficiency"],
            ].map(([label, sft, dpo, orpo]) => (
              <tr key={label} className="border-t border-zinc-800">
                <td className="py-1 text-zinc-500 pr-2">{label}</td>
                <td className={`py-1 text-center ${method === "sft" ? "text-white font-medium" : ""}`}>{sft}</td>
                <td className={`py-1 text-center ${method === "dpo" ? "text-white font-medium" : ""}`}>{dpo}</td>
                <td className={`py-1 text-center ${method === "orpo" ? "text-white font-medium" : ""}`}>{orpo}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </GuideSection>
    </div>
  )
}

function GuideSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="p-3 border border-zinc-800 rounded bg-zinc-900">
      <p className="text-zinc-500 text-xs font-medium uppercase tracking-wider mb-2">{title}</p>
      {children}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function NewJobPage() {
  const { id } = useParams<{ id: string }>()
  const router = useRouter()
  const { data: datasetsData } = useDatasets(id)
  const { data: hfData } = useHFModels()
  const { data: ollamaData } = useOllamaModels()

  const [method, setMethod] = useState<Method>("sft")

  const selectedMethod = METHODS.find(m => m.id === method)!
  const allDatasets = datasetsData?.datasets ?? []
  const eligibleDatasets = allDatasets.filter(d =>
    method === "sft" ? d.status === "tokenized" : ["formatted", "tokenized"].includes(d.status)
  )

  const [form, setForm] = useState({
    name: "",
    dataset_id: "",
    base_model: "mistral:7b",
    model_path: "",
    use_qlora: true,
    lora_r: 16,
    lora_alpha: 32,
    lora_dropout: 0.05,
    learning_rate: 0.0002,
    epochs: 3,
    batch_size: 2,
    grad_accum: 8,
    max_seq_len: 2048,
    bf16: true,
  })

  const [vram, setVram] = useState<{ total_gb: number; warning: string | null } | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    set("learning_rate", selectedMethod.defaultLR)
    set("dataset_id", "")
  }, [method])

  useEffect(() => {
    if (!form.base_model) return
    fetch(`${BASE}/models/vram-estimate?base_model=${encodeURIComponent(form.base_model)}&lora_r=${form.lora_r}&use_qlora=${form.use_qlora}`)
      .then(r => r.json())
      .then(setVram)
      .catch(() => {})
  }, [form.base_model, form.lora_r, form.use_qlora])

  function set(key: string, value: unknown) {
    setForm(f => ({ ...f, [key]: value }))
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!form.dataset_id) { setError("Select a dataset"); return }
    if (!form.model_path) { setError("Select or enter a model path"); return }
    setSubmitting(true)
    setError(null)
    try {
      const res = await fetch(`${BASE}/projects/${id}/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...form, training_method: method }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail ?? "Failed to create job")
      }
      const job = await res.json()
      router.push(`/projects/${id}/jobs/${job.id}`)
    } catch (err: unknown) {
      setError((err as Error).message)
      setSubmitting(false)
    }
  }

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-6">
        <a href={`/projects/${id}/jobs`} className="text-zinc-600 text-xs hover:text-zinc-400">← Jobs</a>
        <h2 className="text-lg font-bold mt-1">New Training Job</h2>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-6 items-start">

        {/* ── Left: Form ── */}
        <form onSubmit={handleSubmit} className="space-y-5">

          {/* Training Method */}
          <Section title="Training Method">
            <div className="grid grid-cols-3 gap-2">
              {METHODS.map(m => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => setMethod(m.id)}
                  className={`p-3 rounded border text-left transition-colors ${
                    method === m.id
                      ? "border-white bg-zinc-800 text-white"
                      : "border-zinc-700 text-zinc-400 hover:border-zinc-500 hover:text-white"
                  }`}
                >
                  <p className="font-bold text-sm">{m.label}</p>
                  <p className="text-xs mt-0.5 text-zinc-500">{m.name}</p>
                </button>
              ))}
            </div>
            <p className="text-zinc-500 text-xs mt-2">{selectedMethod.desc}</p>
          </Section>

          {/* Job name */}
          <Section title="Job Name">
            <input
              value={form.name}
              onChange={e => set("name", e.target.value)}
              placeholder="e.g. qwen-dpo-v1"
              required
              className="input w-full"
            />
          </Section>

          {/* Dataset */}
          <Section title="Dataset">
            {eligibleDatasets.length === 0 ? (
              <p className="text-zinc-500 text-sm">
                No {method === "sft" ? "tokenized" : "DPO-formatted"} datasets.{" "}
                <a href={`/projects/${id}/datasets`} className="underline">Prepare one first →</a>
              </p>
            ) : (
              <select value={form.dataset_id} onChange={e => set("dataset_id", e.target.value)} className="input w-full">
                <option value="">Select dataset...</option>
                {eligibleDatasets.map(d => (
                  <option key={d.id} value={d.id}>
                    {d.name} ({d.row_count?.toLocaleString()} rows) — {d.format_type ?? d.status}
                  </option>
                ))}
              </select>
            )}
          </Section>

          {/* Model */}
          <Section title="Base Model (HuggingFace weights)">
            {hfData?.models.length === 0 ? (
              <div>
                <p className="text-zinc-500 text-xs mb-2">
                  No HF models downloaded. Download weights to <code className="text-zinc-400">storage/models/hf/</code>.
                </p>
                <input
                  value={form.model_path}
                  onChange={e => { set("model_path", e.target.value); set("base_model", e.target.value) }}
                  placeholder="Absolute path to HF model directory"
                  className="input w-full"
                />
              </div>
            ) : (
              <select value={form.model_path} onChange={e => {
                const m = hfData?.models.find(m => m.path === e.target.value)
                set("model_path", e.target.value)
                if (m) set("base_model", m.name)
              }} className="input w-full">
                <option value="">Select model...</option>
                {hfData?.models.map(m => (
                  <option key={m.path} value={m.path}>{m.name} ({m.size_gb} GB)</option>
                ))}
              </select>
            )}
            <div className="flex items-center gap-2 mt-2">
              <span className="text-zinc-500 text-xs">Ollama name:</span>
              {ollamaData?.models && ollamaData.models.length > 0 ? (
                <select value={form.base_model} onChange={e => set("base_model", e.target.value)} className="input text-xs">
                  {ollamaData.models.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}
                </select>
              ) : (
                <input
                  value={form.base_model}
                  onChange={e => set("base_model", e.target.value)}
                  placeholder="e.g. qwen2.5:1.5b"
                  className="input text-xs w-40"
                />
              )}
            </div>
          </Section>

          {/* LoRA */}
          <Section title="LoRA Configuration">
            <div className="grid grid-cols-3 gap-3">
              <Field label="Rank (r)">
                <input type="number" value={form.lora_r} onChange={e => set("lora_r", +e.target.value)} className="input w-full" />
              </Field>
              <Field label="Alpha">
                <input type="number" value={form.lora_alpha} onChange={e => set("lora_alpha", +e.target.value)} className="input w-full" />
              </Field>
              <Field label="Dropout">
                <input type="number" step="0.01" value={form.lora_dropout} onChange={e => set("lora_dropout", +e.target.value)} className="input w-full" />
              </Field>
            </div>
            <label className="flex items-center gap-2 mt-3 cursor-pointer">
              <input type="checkbox" checked={form.use_qlora} onChange={e => set("use_qlora", e.target.checked)} className="accent-white" />
              <span className="text-sm">Use QLoRA (4-bit) — recommended for &lt;24 GB VRAM</span>
            </label>
          </Section>

          {/* Training Hyperparameters */}
          <Section title="Training Hyperparameters">
            <div className="grid grid-cols-2 gap-3">
              <Field label="Learning Rate">
                <input type="number" step="0.000001" value={form.learning_rate} onChange={e => set("learning_rate", +e.target.value)} className="input w-full" />
              </Field>
              <Field label="Epochs">
                <input type="number" value={form.epochs} onChange={e => set("epochs", +e.target.value)} className="input w-full" />
              </Field>
              <Field label="Batch Size">
                <input type="number" value={form.batch_size} onChange={e => set("batch_size", +e.target.value)} className="input w-full" />
              </Field>
              <Field label="Grad Accum Steps">
                <input type="number" value={form.grad_accum} onChange={e => set("grad_accum", +e.target.value)} className="input w-full" />
              </Field>
              <Field label="Max Seq Length">
                <input type="number" value={form.max_seq_len} onChange={e => set("max_seq_len", +e.target.value)} className="input w-full" />
              </Field>
            </div>
            <p className="text-zinc-600 text-xs mt-2">
              Effective batch = {form.batch_size} × {form.grad_accum} = {form.batch_size * form.grad_accum}
            </p>
          </Section>

          {error && (
            <div className="p-3 border border-red-800 rounded bg-red-950 text-red-300 text-sm">{error}</div>
          )}

          <button
            type="submit"
            disabled={submitting}
            className="w-full py-3 bg-white text-black font-medium rounded hover:bg-zinc-200 disabled:opacity-50"
          >
            {submitting ? "Launching..." : `Launch ${selectedMethod.label} Training Job`}
          </button>
        </form>

        {/* ── Right: Guide ── */}
        <GuidePanel method={method} vram={vram} />

      </div>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="p-4 border border-zinc-800 rounded bg-zinc-900">
      <h3 className="text-zinc-400 text-xs font-medium mb-3 uppercase tracking-wider">{title}</h3>
      {children}
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="text-zinc-500 text-xs block mb-1">{label}</label>
      {children}
    </div>
  )
}
