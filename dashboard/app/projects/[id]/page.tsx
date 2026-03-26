"use client"

import React from "react"
import { useRouter } from "next/navigation"
import { useParams } from "next/navigation"
import { useProject, useDeleteProject } from "@/lib/hooks/useProjects"
import { useDatasets } from "@/lib/hooks/useDatasets"
import { useJobs } from "@/lib/hooks/useJobs"

// ── Stat card ─────────────────────────────────────────────────────────────────

function StatCard({ label, value, href }: { label: string; value: number | string; href: string }) {
  return (
    <a
      href={href}
      className="p-4 border border-zinc-800 rounded-xl bg-zinc-900 hover:border-zinc-600 hover:bg-zinc-800/50 transition-all group"
    >
      <p className="text-2xl font-bold text-white group-hover:text-white">{value}</p>
      <p className="text-zinc-500 text-xs mt-1">{label}</p>
    </a>
  )
}

// ── Section link ──────────────────────────────────────────────────────────────

function SectionLink({ href, label, description, icon }: {
  href: string; label: string; description: string; icon: React.ReactNode
}) {
  return (
    <a
      href={href}
      className="flex items-start gap-3 p-4 border border-zinc-800 rounded-xl bg-zinc-900 hover:border-zinc-600 hover:bg-zinc-800/50 transition-all group"
    >
      <span className="mt-0.5 text-zinc-500 group-hover:text-zinc-300 transition-colors shrink-0">
        {icon}
      </span>
      <div>
        <p className="text-sm font-medium text-zinc-200 group-hover:text-white">{label}</p>
        <p className="text-zinc-600 text-xs mt-0.5">{description}</p>
      </div>
      <svg className="ml-auto mt-1 text-zinc-700 group-hover:text-zinc-400 shrink-0" width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M6 4l4 4-4 4" />
      </svg>
    </a>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function ProjectPage() {
  const { id } = useParams<{ id: string }>()
  const router = useRouter()
  const { data: project, isLoading } = useProject(id)
  const deleteProject = useDeleteProject()
  const { data: datasetsData } = useDatasets(id)
  const { data: jobsData } = useJobs(id)

  if (isLoading) {
    return (
      <div className="space-y-4 animate-pulse">
        <div className="skeleton h-6 w-48" />
        <div className="skeleton h-4 w-64" />
        <div className="grid grid-cols-2 gap-3 mt-6">
          {[1,2].map(i => <div key={i} className="skeleton h-20 rounded-xl" />)}
        </div>
      </div>
    )
  }

  if (!project) return <div className="text-red-400 text-sm">Project not found.</div>

  async function handleDelete() {
    if (confirm(`Delete project "${project!.name}"? All datasets and jobs will be removed.`)) {
      await deleteProject.mutateAsync(project!.id)
      router.push("/")
    }
  }

  const datasetCount = datasetsData?.datasets.length ?? "—"
  const jobCount     = jobsData?.jobs.length ?? "—"
  const completedJobs = jobsData?.jobs.filter(j => j.status === "completed").length ?? 0

  const sections = [
    {
      href: `/projects/${id}/datasets`,
      label: "Datasets",
      description: "Upload files, scrape web pages, or generate synthetic data.",
      icon: (
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <ellipse cx="8" cy="4" rx="6" ry="2.5" />
          <path d="M2 4v4c0 1.38 2.69 2.5 6 2.5s6-1.12 6-2.5V4" />
          <path d="M2 8v4c0 1.38 2.69 2.5 6 2.5s6-1.12 6-2.5V8" />
        </svg>
      ),
    },
    {
      href: `/projects/${id}/jobs`,
      label: "Training Jobs",
      description: "Fine-tune models with SFT, DPO, or ORPO on your datasets.",
      icon: (
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <rect x="4" y="4" width="8" height="8" rx="1" />
          <path d="M6 1v3M10 1v3M6 12v3M10 12v3M1 6h3M1 10h3M12 6h3M12 10h3" />
        </svg>
      ),
    },
    {
      href: `/projects/${id}/evaluate`,
      label: "Evaluate",
      description: "Compare model performance with ROUGE, BLEU, and human scores.",
      icon: (
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M2 12l4-4 3 3 5-7" /><path d="M2 14h12" />
        </svg>
      ),
    },
    {
      href: `/projects/${id}/inference`,
      label: "Inference",
      description: "Chat with base or fine-tuned models, compare side by side.",
      icon: (
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14 9a2 2 0 01-2 2H5l-3 3V4a2 2 0 012-2h8a2 2 0 012 2v5z" />
        </svg>
      ),
    },
    {
      href: `/projects/${id}/rag`,
      label: "RAG",
      description: "Upload documents and ask questions over your content.",
      icon: (
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
          <circle cx="6.5" cy="6.5" r="4.5" /><path d="M10.5 10.5l3 3" />
        </svg>
      ),
    },
    {
      href: `/projects/${id}/experiments`,
      label: "Experiments",
      description: "Compare completed jobs: hyperparameters, ROUGE, BLEU metrics side by side.",
      icon: (
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M2 13l3-4 3 2 3-5 3 2" /><path d="M2 15h12" />
        </svg>
      ),
    },
  ]

  return (
    <div className="max-w-2xl animate-fade-in">

      {/* Header */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-xl font-bold text-white">{project.name}</h1>
          {project.description && (
            <p className="text-zinc-500 text-sm mt-1">{project.description}</p>
          )}
          <p className="text-zinc-700 text-xs mt-1">
            Created {new Date(project.created_at).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" })}
          </p>
        </div>
        <button
          onClick={handleDelete}
          className="text-xs text-zinc-600 hover:text-red-400 border border-zinc-800 hover:border-red-900/60 rounded-md px-3 py-1.5 transition-colors mt-1"
        >
          Delete
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-3 mb-8">
        <StatCard label="Datasets"         value={datasetCount}  href={`/projects/${id}/datasets`} />
        <StatCard label="Training Jobs"    value={jobCount}      href={`/projects/${id}/jobs`} />
        <StatCard label="Completed Jobs"   value={completedJobs} href={`/projects/${id}/evaluate`} />
      </div>

      {/* Section links */}
      <div>
        <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-3">Sections</p>
        <div className="grid grid-cols-1 gap-2">
          {sections.map(s => (
            <SectionLink key={s.href} {...s} />
          ))}
        </div>
      </div>

    </div>
  )
}
