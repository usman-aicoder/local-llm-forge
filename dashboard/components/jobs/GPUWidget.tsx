"use client"

import { useGPU } from "@/lib/hooks/useGPU"

interface Props {
  jobId: string | null
  active: boolean
}

function Bar({ pct, color }: { pct: number; color: string }) {
  return (
    <div className="w-full bg-zinc-800 rounded-full h-1.5 overflow-hidden">
      <div
        className={`h-full rounded-full transition-all duration-500 ${color}`}
        style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
      />
    </div>
  )
}

export function GPUWidget({ jobId, active }: Props) {
  const { data: gpu, isError } = useGPU(jobId, active)

  if (isError || !gpu) {
    return (
      <div className="p-3 border border-zinc-800 rounded bg-zinc-900">
        <p className="text-zinc-600 text-xs">GPU stats unavailable</p>
        <p className="text-zinc-700 text-xs mt-0.5">nvidia-smi not found</p>
      </div>
    )
  }

  const utilPct  = gpu.gpu_util_pct ?? 0
  const memUsed  = gpu.mem_used_gb ?? 0
  const memTotal = gpu.mem_total_gb ?? 1
  const memPct   = (memUsed / memTotal) * 100

  const utilColor = utilPct > 80 ? "bg-green-400" : utilPct > 40 ? "bg-yellow-400" : "bg-zinc-500"
  const memColor  = memPct  > 85 ? "bg-red-400"   : memPct  > 60 ? "bg-yellow-400" : "bg-blue-400"

  return (
    <div className="p-3 border border-zinc-800 rounded bg-zinc-900 space-y-3">
      <p className="text-zinc-500 text-xs font-mono uppercase tracking-wider">GPU</p>

      {/* Utilization */}
      <div>
        <div className="flex justify-between mb-1">
          <span className="text-zinc-500 text-xs">Utilization</span>
          <span className={`text-xs font-mono font-bold ${utilPct > 80 ? "text-green-400" : "text-zinc-400"}`}>
            {gpu.gpu_util_pct !== null ? `${utilPct}%` : "—"}
          </span>
        </div>
        <Bar pct={utilPct} color={utilColor} />
      </div>

      {/* VRAM */}
      <div>
        <div className="flex justify-between mb-1">
          <span className="text-zinc-500 text-xs">VRAM</span>
          <span className={`text-xs font-mono ${memPct > 85 ? "text-red-400" : "text-zinc-400"}`}>
            {gpu.mem_used_gb !== null
              ? `${memUsed.toFixed(1)} / ${memTotal.toFixed(1)} GB`
              : "—"}
          </span>
        </div>
        <Bar pct={memPct} color={memColor} />
      </div>

      {active && (
        <p className="text-zinc-700 text-xs">refreshes every 5s</p>
      )}
    </div>
  )
}
