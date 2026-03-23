import type { Dataset } from "@/lib/types"

const STAGES = ["uploaded", "inspected", "cleaned", "formatted", "tokenized"] as const
type Stage = typeof STAGES[number]

const STAGE_LABELS: Record<Stage, string> = {
  uploaded:  "Uploaded",
  inspected: "EDA",
  cleaned:   "Cleaned",
  formatted: "Formatted",
  tokenized: "Tokenized",
}

const STAGE_HREF: Record<Stage, (id: string, dsId: string) => string> = {
  uploaded:  (id, dsId) => `/projects/${id}/datasets/${dsId}/eda`,
  inspected: (id, dsId) => `/projects/${id}/datasets/${dsId}/clean`,
  cleaned:   (id, dsId) => `/projects/${id}/datasets/${dsId}/format`,
  formatted: (id, dsId) => `/projects/${id}/datasets/${dsId}/tokenize`,
  tokenized: (id, dsId) => `/projects/${id}/datasets/${dsId}/tokenize`,
}

interface Props {
  dataset: Dataset
  projectId: string
}

export function PipelineStatus({ dataset, projectId }: Props) {
  const currentIdx = STAGES.indexOf(dataset.status as Stage)

  return (
    <div className="flex items-center gap-1 flex-wrap">
      {STAGES.map((stage, idx) => {
        const done    = idx <= currentIdx
        const current = idx === currentIdx
        const href    = STAGE_HREF[stage](projectId, dataset.id)

        return (
          <a key={stage} href={href} className="flex items-center gap-1">
            <span
              className={[
                "px-2 py-0.5 rounded text-xs font-mono transition-colors",
                done && !current ? "bg-zinc-700 text-zinc-300" : "",
                current         ? "bg-white text-black" : "",
                !done           ? "bg-zinc-900 text-zinc-600 border border-zinc-800" : "",
              ].join(" ")}
            >
              {STAGE_LABELS[stage]}
            </span>
            {idx < STAGES.length - 1 && (
              <span className="text-zinc-700 text-xs">→</span>
            )}
          </a>
        )
      })}
    </div>
  )
}
