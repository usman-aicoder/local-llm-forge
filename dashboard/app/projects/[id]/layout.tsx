"use client"

import { useParams, usePathname } from "next/navigation"
import { useProject } from "@/lib/hooks/useProjects"
import React from "react"

// ── Nav icons ─────────────────────────────────────────────────────────────────

function IconGrid() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="1" y="1" width="6" height="6" rx="1" /><rect x="9" y="1" width="6" height="6" rx="1" />
      <rect x="1" y="9" width="6" height="6" rx="1" /><rect x="9" y="9" width="6" height="6" rx="1" />
    </svg>
  )
}

function IconDatabase() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="8" cy="4" rx="6" ry="2.5" />
      <path d="M2 4v4c0 1.38 2.69 2.5 6 2.5s6-1.12 6-2.5V4" />
      <path d="M2 8v4c0 1.38 2.69 2.5 6 2.5s6-1.12 6-2.5V8" />
    </svg>
  )
}

function IconCpu() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="4" y="4" width="8" height="8" rx="1" />
      <path d="M6 1v3M10 1v3M6 12v3M10 12v3M1 6h3M1 10h3M12 6h3M12 10h3" />
    </svg>
  )
}

function IconChart() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M2 12l4-4 3 3 5-7" />
      <path d="M2 14h12" />
    </svg>
  )
}

function IconChat() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 9a2 2 0 01-2 2H5l-3 3V4a2 2 0 012-2h8a2 2 0 012 2v5z" />
    </svg>
  )
}

function IconSearch() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
      <circle cx="6.5" cy="6.5" r="4.5" />
      <path d="M10.5 10.5l3 3" />
    </svg>
  )
}

function IconBack() {
  return (
    <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10 4L6 8l4 4" />
    </svg>
  )
}

// ── Sidebar nav item ──────────────────────────────────────────────────────────

function NavItem({
  href,
  icon,
  label,
  active,
}: {
  href: string
  icon: React.ReactNode
  label: string
  active: boolean
}) {
  return (
    <a
      href={href}
      className={active ? "nav-item-active" : "nav-item"}
    >
      <span className="shrink-0">{icon}</span>
      {label}
    </a>
  )
}

// ── Layout ────────────────────────────────────────────────────────────────────

export default function ProjectLayout({ children }: { children: React.ReactNode }) {
  const { id } = useParams<{ id: string }>()
  const pathname = usePathname()
  const { data: project, isLoading } = useProject(id)

  const base = `/projects/${id}`

  const navItems = [
    { label: "Overview",   href: base,                  icon: <IconGrid />,     exact: true },
    { label: "Datasets",   href: `${base}/datasets`,    icon: <IconDatabase /> },
    { label: "Jobs",       href: `${base}/jobs`,        icon: <IconCpu />      },
    { label: "Evaluate",   href: `${base}/evaluate`,    icon: <IconChart />    },
    { label: "Inference",  href: `${base}/inference`,   icon: <IconChat />     },
    { label: "RAG",        href: `${base}/rag`,         icon: <IconSearch />   },
  ]

  return (
    <div className="flex gap-6 min-h-[calc(100vh-3.5rem-3rem)]">

      {/* ── Sidebar ── */}
      <aside className="w-48 shrink-0 flex flex-col">

        {/* Back link */}
        <a
          href="/"
          className="flex items-center gap-1.5 text-xs text-zinc-600 hover:text-zinc-400 mb-4 transition-colors"
        >
          <IconBack /> Projects
        </a>

        {/* Project identity */}
        <div className="mb-4 px-1">
          {isLoading ? (
            <div className="skeleton h-4 w-28" />
          ) : (
            <>
              <p className="font-semibold text-sm text-white truncate" title={project?.name}>
                {project?.name}
              </p>
              {project?.description && (
                <p className="text-zinc-600 text-xs mt-0.5 truncate" title={project.description}>
                  {project.description}
                </p>
              )}
            </>
          )}
        </div>

        {/* Divider */}
        <div className="h-px bg-zinc-800 mb-3" />

        {/* Nav items */}
        <nav className="flex flex-col gap-0.5">
          {navItems.map(item => {
            const active = item.exact
              ? pathname === item.href
              : pathname.startsWith(item.href)
            return (
              <NavItem
                key={item.href}
                href={item.href}
                icon={item.icon}
                label={item.label}
                active={active}
              />
            )
          })}
        </nav>

      </aside>

      {/* ── Main content ── */}
      <div className="flex-1 min-w-0">
        {children}
      </div>

    </div>
  )
}
