"use client"

import { useState, useMemo } from "react"
import { useProjects, useCreateProject, useDeleteProject } from "@/lib/hooks/useProjects"
import type { Project } from "@/lib/types"

// ── Icons ─────────────────────────────────────────────────────────────────────

function PlusIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M8 2v12M2 8h12" />
    </svg>
  )
}

function SearchIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
      <circle cx="6.5" cy="6.5" r="4.5" />
      <path d="M10.5 10.5l3 3" />
    </svg>
  )
}

function ChevronRightIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M6 4l4 4-4 4" />
    </svg>
  )
}

function FolderIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-zinc-700">
      <path d="M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V7z" />
    </svg>
  )
}

// ── Skeleton card ─────────────────────────────────────────────────────────────

function SkeletonCard() {
  return (
    <div className="p-5 border border-zinc-800 rounded-xl bg-zinc-900">
      <div className="skeleton h-4 w-32 mb-3" />
      <div className="skeleton h-3 w-48 mb-4" />
      <div className="flex items-center justify-between">
        <div className="skeleton h-3 w-24" />
        <div className="skeleton h-6 w-16 rounded-md" />
      </div>
    </div>
  )
}

// ── Project card ──────────────────────────────────────────────────────────────

function ProjectCard({ project, onDelete }: { project: Project; onDelete: (id: string) => void }) {
  const updated = new Date(project.updated_at)
  const now = new Date()
  const diffDays = Math.floor((now.getTime() - updated.getTime()) / 86_400_000)
  const relativeDate =
    diffDays === 0 ? "Today" :
    diffDays === 1 ? "Yesterday" :
    diffDays < 7  ? `${diffDays}d ago` :
    updated.toLocaleDateString(undefined, { month: "short", day: "numeric" })

  return (
    <div className="group relative p-5 border border-zinc-800 rounded-xl bg-zinc-900 hover:border-zinc-600 hover:bg-zinc-900/80 transition-all animate-fade-in">
      {/* Delete button */}
      <button
        onClick={e => { e.preventDefault(); if (confirm(`Delete "${project.name}"?`)) onDelete(project.id) }}
        className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 p-1.5 rounded-md text-zinc-600 hover:text-red-400 hover:bg-red-950/40 transition-all"
        title="Delete project"
      >
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <path d="M2 2l12 12M14 2L2 14" />
        </svg>
      </button>

      <a href={`/projects/${project.id}`} className="block">
        {/* Header */}
        <div className="mb-3">
          <h2 className="font-semibold text-sm text-white leading-snug pr-6">{project.name}</h2>
          {project.description ? (
            <p className="text-zinc-500 text-xs mt-1 line-clamp-2">{project.description}</p>
          ) : (
            <p className="text-zinc-700 text-xs mt-1 italic">No description</p>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between">
          <span className="text-zinc-600 text-xs">Updated {relativeDate}</span>
          <span className="flex items-center gap-1 text-xs text-zinc-500 group-hover:text-white transition-colors font-medium">
            Open <ChevronRightIcon />
          </span>
        </div>
      </a>
    </div>
  )
}

// ── Create form (inline slide-down) ───────────────────────────────────────────

function CreateForm({ onClose }: { onClose: () => void }) {
  const createProject = useCreateProject()
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!name.trim()) return
    await createProject.mutateAsync({ name: name.trim(), description: description.trim() || undefined })
    onClose()
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="mb-6 p-5 border border-zinc-700 rounded-xl bg-zinc-900 animate-fade-in"
    >
      <h3 className="text-sm font-semibold mb-4">New Project</h3>
      <div className="flex flex-col gap-3">
        <input
          value={name}
          onChange={e => setName(e.target.value)}
          placeholder="Project name"
          className="input"
          autoFocus
        />
        <input
          value={description}
          onChange={e => setDescription(e.target.value)}
          placeholder="Description (optional)"
          className="input"
        />
        <div className="flex gap-2 pt-1">
          <button
            type="submit"
            disabled={!name.trim() || createProject.isPending}
            className="btn-primary disabled:opacity-40"
          >
            {createProject.isPending ? "Creating…" : "Create Project"}
          </button>
          <button type="button" onClick={onClose} className="btn-ghost">
            Cancel
          </button>
        </div>
      </div>
    </form>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function HomePage() {
  const { data, isLoading, error } = useProjects()
  const deleteProject = useDeleteProject()
  const [showForm, setShowForm] = useState(false)
  const [search, setSearch] = useState("")

  const allProjects: Project[] = data?.projects ?? []

  const filtered = useMemo(() => {
    const q = search.toLowerCase().trim()
    if (!q) return allProjects
    return allProjects.filter(p =>
      p.name.toLowerCase().includes(q) || (p.description ?? "").toLowerCase().includes(q)
    )
  }, [allProjects, search])

  return (
    <div className="max-w-5xl mx-auto">

      {/* ── Header ── */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-xl font-bold">Projects</h1>
          <p className="text-zinc-500 text-sm mt-1">
            Fine-tune models, build RAG pipelines, and run evaluations.
          </p>
        </div>
        <button
          onClick={() => setShowForm(v => !v)}
          className="flex items-center gap-1.5 px-4 py-2 bg-white text-black text-sm font-medium rounded-lg hover:bg-zinc-200 transition-colors shrink-0"
        >
          <PlusIcon />
          New Project
        </button>
      </div>

      {/* ── Stats strip ── */}
      {!isLoading && !error && allProjects.length > 0 && (
        <div className="flex items-center gap-6 mb-5 p-3 rounded-lg border border-zinc-800 bg-zinc-900/50">
          <div className="text-center">
            <p className="text-lg font-bold text-white">{allProjects.length}</p>
            <p className="text-zinc-500 text-xs">Projects</p>
          </div>
          <div className="h-8 w-px bg-zinc-800" />
          <p className="text-zinc-600 text-xs">
            Select a project to manage datasets, run training jobs, and evaluate results.
          </p>
        </div>
      )}

      {/* ── Create form ── */}
      {showForm && <CreateForm onClose={() => setShowForm(false)} />}

      {/* ── Search ── */}
      {allProjects.length > 3 && (
        <div className="relative mb-5">
          <span className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500 pointer-events-none">
            <SearchIcon />
          </span>
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search projects…"
            className="input w-full pl-9"
          />
        </div>
      )}

      {/* ── Loading skeletons ── */}
      {isLoading && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {[1, 2, 3, 4].map(i => <SkeletonCard key={i} />)}
        </div>
      )}

      {/* ── Error ── */}
      {error && (
        <div className="p-4 border border-red-800 rounded-lg bg-red-950/40 text-red-300 text-sm">
          Failed to load projects. Is the backend running on port 8010?
        </div>
      )}

      {/* ── Empty state ── */}
      {!isLoading && !error && allProjects.length === 0 && (
        <div className="text-center py-20 animate-fade-in">
          <div className="flex justify-center mb-4"><FolderIcon /></div>
          <p className="text-zinc-500 text-sm font-medium">No projects yet</p>
          <p className="text-zinc-700 text-xs mt-1 mb-5">Create your first project to get started.</p>
          <button
            onClick={() => setShowForm(true)}
            className="inline-flex items-center gap-1.5 px-4 py-2 bg-white text-black text-sm font-medium rounded-lg hover:bg-zinc-200 transition-colors"
          >
            <PlusIcon />
            New Project
          </button>
        </div>
      )}

      {/* ── No search results ── */}
      {!isLoading && !error && allProjects.length > 0 && filtered.length === 0 && (
        <p className="text-zinc-600 text-sm text-center py-10">No projects match "{search}"</p>
      )}

      {/* ── Project grid ── */}
      {filtered.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {filtered.map(project => (
            <ProjectCard
              key={project.id}
              project={project}
              onDelete={id => deleteProject.mutate(id)}
            />
          ))}
        </div>
      )}

    </div>
  )
}
