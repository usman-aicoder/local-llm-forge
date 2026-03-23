"use client"

import { useParams } from "next/navigation"
import { useJobs, useDeleteJob } from "@/lib/hooks/useJobs"
import type { TrainingJob } from "@/lib/types"

const STATUS_STYLE: Record<TrainingJob["status"], string> = {
  queued:    "bg-zinc-800 text-zinc-400",
  running:   "bg-blue-900 text-blue-300",
  completed: "bg-green-900 text-green-300",
  failed:    "bg-red-900 text-red-300",
  cancelled: "bg-zinc-800 text-zinc-500",
}

export default function JobsPage() {
  const { id } = useParams<{ id: string }>()
  const { data, isLoading } = useJobs(id)
  const deleteJob = useDeleteJob(id)

  function handleDelete(e: React.MouseEvent, job: TrainingJob) {
    e.preventDefault()
    e.stopPropagation()
    if (confirm(`Delete job "${job.name}"? This will also remove the saved adapter/checkpoint files.`)) {
      deleteJob.mutate(job.id)
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-lg font-bold">Training Jobs</h2>
          <p className="text-zinc-500 text-sm mt-0.5">Fine-tune models with SFT, DPO, or ORPO.</p>
        </div>
        <a
          href={`/projects/${id}/jobs/new`}
          className="px-4 py-2 bg-white text-black text-sm font-medium rounded-lg hover:bg-zinc-200 transition-colors"
        >
          + New Job
        </a>
      </div>

      {isLoading && (
        <div className="grid gap-3">
          {[1,2,3].map(i => <div key={i} className="skeleton h-20 rounded-lg" />)}
        </div>
      )}

      {!isLoading && data?.jobs.length === 0 && (
        <div className="text-center py-16 text-zinc-600 animate-fade-in">
          <p className="text-3xl mb-3">◻</p>
          <p className="text-sm">No jobs yet. Create one to start training.</p>
        </div>
      )}

      <div className="grid gap-3">
        {data?.jobs.map((job: TrainingJob) => (
          <div key={job.id} className="group relative">
            <a
              href={`/projects/${id}/jobs/${job.id}`}
              className="p-4 border border-zinc-800 rounded bg-zinc-900 hover:border-zinc-600 transition-colors block"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`px-2 py-0.5 rounded text-xs font-mono ${STATUS_STYLE[job.status]}`}>
                      {job.status}
                    </span>
                    <h3 className="font-medium text-sm truncate">{job.name}</h3>
                    {job.training_method && job.training_method !== "sft" && (
                      <span className="px-1.5 py-0.5 rounded text-xs bg-purple-900 text-purple-300">
                        {job.training_method.toUpperCase()}
                      </span>
                    )}
                  </div>
                  <p className="text-zinc-500 text-xs">
                    {job.base_model} · LoRA r={job.lora_r} · {job.epochs} epochs
                    {job.use_qlora ? " · QLoRA" : ""}
                  </p>
                  <p className="text-zinc-700 text-xs mt-1">
                    {new Date(job.created_at).toLocaleString()}
                  </p>
                </div>
                <span className="text-zinc-600 text-xs ml-4">→</span>
              </div>
              {job.error_message && (
                <p className="mt-2 text-red-400 text-xs truncate">{job.error_message.split("\n")[0]}</p>
              )}
            </a>
            <button
              onClick={e => handleDelete(e, job)}
              className="absolute top-3 right-8 opacity-0 group-hover:opacity-100 px-2 py-1 text-xs text-zinc-600 hover:text-red-400 transition-all"
            >
              delete
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}
