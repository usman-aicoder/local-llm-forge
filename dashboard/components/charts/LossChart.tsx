"use client"

import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts"

interface DataPoint {
  epoch: number
  train_loss: number
  eval_loss: number
}

export function LossChart({ data }: { data: DataPoint[] }) {
  if (!data.length) {
    return (
      <div className="flex items-center justify-center h-40 text-zinc-600 text-sm">
        Waiting for first epoch...
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
        <XAxis
          dataKey="epoch"
          tick={{ fontSize: 11, fill: "#71717a" }}
          label={{ value: "Epoch", position: "insideBottom", offset: -2, fontSize: 11, fill: "#71717a" }}
        />
        <YAxis tick={{ fontSize: 11, fill: "#71717a" }} width={40} />
        <Tooltip
          contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", fontSize: 12 }}
          formatter={(v) => (typeof v === "number" ? v.toFixed(4) : v)}
        />
        <Legend wrapperStyle={{ fontSize: 12, color: "#a1a1aa" }} />
        <Line
          type="monotone"
          dataKey="train_loss"
          stroke="#a1a1aa"
          strokeWidth={2}
          dot={{ r: 3 }}
          name="Train Loss"
        />
        <Line
          type="monotone"
          dataKey="eval_loss"
          stroke="#ffffff"
          strokeWidth={2}
          dot={{ r: 3 }}
          name="Eval Loss"
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
