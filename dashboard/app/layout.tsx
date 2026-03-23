"use client"

import "./globals.css"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { useState, useEffect } from "react"
import { api } from "@/lib/api"

// ── FOUC-prevention script (runs synchronously before first paint) ─────────────
// Inlined as a string so Next.js doesn't defer it.
const THEME_SCRIPT = `
(function(){
  try {
    var t = localStorage.getItem('theme');
    var dark = t === 'dark' || (!t && window.matchMedia('(prefers-color-scheme: dark)').matches);
    if (dark) document.documentElement.classList.add('dark');
  } catch(e){}
})();
`

// ── Theme toggle ──────────────────────────────────────────────────────────────

function ThemeToggle() {
  const [dark, setDark] = useState(true)

  useEffect(() => {
    // Sync state with whatever the FOUC script already set
    setDark(document.documentElement.classList.contains("dark"))
  }, [])

  function toggle() {
    const next = !dark
    setDark(next)
    document.documentElement.classList.toggle("dark", next)
    try { localStorage.setItem("theme", next ? "dark" : "light") } catch {}
  }

  return (
    <button
      onClick={toggle}
      aria-label="Toggle theme"
      className="w-8 h-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
    >
      {dark ? (
        // Sun icon
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>
        </svg>
      ) : (
        // Moon icon
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 12.79A9 9 0 1 1 11.21 3a7 7 0 0 0 9.79 9.79z"/>
        </svg>
      )}
    </button>
  )
}

// ── Backend status pill ───────────────────────────────────────────────────────

function BackendStatus() {
  const [status, setStatus] = useState<"ok" | "err" | "loading">("loading")

  useEffect(() => {
    api.health()
      .then(() => setStatus("ok"))
      .catch(() => setStatus("err"))
  }, [])

  const cfg = {
    ok:      { dot: "bg-green-400",  ring: "border-green-800 bg-green-950/40",  text: "text-green-400",  label: "API online"  },
    err:     { dot: "bg-red-400",    ring: "border-red-800 bg-red-950/40",      text: "text-red-400",    label: "API offline" },
    loading: { dot: "bg-zinc-500",   ring: "border-zinc-700 bg-zinc-900",       text: "text-zinc-500",   label: "checking…"   },
  }[status]

  return (
    <span className={`hidden sm:flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full border ${cfg.ring} ${cfg.text} select-none`}>
      <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${cfg.dot} ${status === "loading" ? "animate-pulse" : ""}`} />
      {cfg.label}
    </span>
  )
}

// ── Nav link ──────────────────────────────────────────────────────────────────

function NavLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a
      href={href}
      className="text-zinc-400 hover:text-white text-sm transition-colors px-1"
    >
      {children}
    </a>
  )
}

// ── Root layout ───────────────────────────────────────────────────────────────

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(() => new QueryClient({
    defaultOptions: {
      queries: { retry: 1, staleTime: 10_000 },
    },
  }))

  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
        {/* FOUC prevention — must be synchronous */}
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body>
        <QueryClientProvider client={queryClient}>
          <div className="min-h-screen bg-zinc-950 text-zinc-50 flex flex-col">

            {/* ── Navbar ── */}
            <nav className="sticky top-0 z-40 border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-md px-4 sm:px-6">
              <div className="max-w-7xl mx-auto flex items-center gap-4 h-14">

                {/* Logo mark + brand */}
                <a href="/" className="flex items-center gap-2.5 shrink-0 group">
                  <span className="w-7 h-7 rounded-md bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center shadow-glow group-hover:shadow-none transition-shadow">
                    <svg width="13" height="13" viewBox="0 0 16 16" fill="white">
                      <path d="M8 1L2 5v6l6 4 6-4V5L8 1z" opacity=".9"/>
                      <path d="M8 1v14M2 5l6 4 6-4" stroke="white" strokeWidth="1.2" fill="none" opacity=".5"/>
                    </svg>
                  </span>
                  <span className="font-semibold text-sm tracking-tight text-white">
                    LLM Platform
                  </span>
                </a>

                {/* Nav links */}
                <div className="hidden sm:flex items-center gap-1 ml-4">
                  <NavLink href="/">Projects</NavLink>
                </div>

                {/* Right side */}
                <div className="ml-auto flex items-center gap-2">
                  <BackendStatus />
                  <ThemeToggle />
                </div>

              </div>
            </nav>

            {/* ── Page content ── */}
            <main className="flex-1 p-4 sm:p-6 max-w-7xl mx-auto w-full">
              {children}
            </main>

          </div>
        </QueryClientProvider>
      </body>
    </html>
  )
}
