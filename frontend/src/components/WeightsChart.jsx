/**
 * components/WeightsChart.jsx
 *
 * Visualises EXP3's probability weight distribution across arms.
 * Shows each arm's selection probability (sums to 1.0).
 */
import React from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell,
} from 'recharts'

const TICK_STYLE = { fontSize: 10, fontFamily: 'var(--font-mono)', fill: 'var(--color-text-secondary)' }

const ARM_COLORS = ['#E89320', '#378ADD', '#1D9E75', '#D85A30', '#9B59B6', '#E74C3C', '#F39C12', '#2ECC71']

export default function WeightsChart({ probabilities, labels }) {
  if (!probabilities || probabilities.length === 0) return null

  const data = probabilities.map((p, i) => ({
    arm: labels && labels[i] ? labels[i] : String.fromCharCode(65 + i),
    probability: +p.toFixed(4),
  }))

  return (
    <div>
      <div style={{ fontSize: 11, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)', marginBottom: 6 }}>
        selection probability weights
      </div>
      <ResponsiveContainer width="100%" height={160}>
        <BarChart data={data} barSize={28}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" vertical={false} />
          <XAxis dataKey="arm" tick={TICK_STYLE} axisLine={false} tickLine={false} />
          <YAxis
            tick={TICK_STYLE}
            axisLine={false}
            tickLine={false}
            width={40}
            domain={[0, 1]}
            tickFormatter={v => v.toFixed(2)}
          />
          <Tooltip
            contentStyle={{
              background: 'var(--color-background-primary)',
              border: '0.5px solid var(--color-border-tertiary)',
              borderRadius: 8,
              fontSize: 11,
              fontFamily: 'var(--font-mono)',
            }}
            formatter={(v) => [v.toFixed(4), 'P(select)']}
          />
          <Bar dataKey="probability" radius={[3, 3, 0, 0]}>
            {data.map((_, i) => (
              <Cell key={i} fill={ARM_COLORS[i % ARM_COLORS.length]} opacity={0.8} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
