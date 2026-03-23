"use client"

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts"

interface Props {
  buckets: number[]
  counts: number[]
  maxLength?: number
}

export function TokenHistogram({ buckets, counts, maxLength = 2048 }: Props) {
  if (!buckets.length) return <div className="text-zinc-600 text-xs">No data</div>

  const data = buckets.map((b, i) => ({ tokens: b, count: counts[i] ?? 0 }))

  return (
    <ResponsiveContainer width="100%" height={180}>
      <BarChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
        <XAxis
          dataKey="tokens"
          tick={{ fontSize: 10, fill: "#71717a" }}
          tickFormatter={(v) => `${v}`}
        />
        <YAxis tick={{ fontSize: 10, fill: "#71717a" }} width={36} />
        <Tooltip
          contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", fontSize: 12 }}
          labelFormatter={(v) => `~${v} tokens`}
          formatter={(v) => [v ?? 0, "samples"]}
        />
        <Bar dataKey="count" fill="#3f3f46" radius={[2, 2, 0, 0]} />
        {maxLength && (
          <ReferenceLine
            x={maxLength}
            stroke="#ef4444"
            strokeDasharray="4 2"
            label={{ value: "max", fontSize: 10, fill: "#ef4444" }}
          />
        )}
      </BarChart>
    </ResponsiveContainer>
  )
}
