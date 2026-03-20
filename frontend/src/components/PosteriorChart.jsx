/**
 * components/PosteriorChart.jsx
 * Animated Beta posterior distributions per arm using Recharts AreaChart.
 */
import React, { useMemo } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts'
import { computePDFCurves } from '../lib/beta'

const ARM_COLORS = [
  '#E89320', '#378ADD', '#1D9E75', '#D85A30', '#9B59B6',
  '#E74C3C', '#2ECC71', '#3498DB', '#F39C12', '#1ABC9C',
]

const TICK_STYLE = { fontSize: 10, fontFamily: 'var(--font-mono)', fill: 'var(--color-text-secondary)' }

export default function PosteriorChart({ betaParams, labels }) {
  const data = useMemo(() => computePDFCurves(betaParams, 80), [betaParams])

  return (
    <div>
      <div style={{ fontSize: 11, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)', marginBottom: 6 }}>
        posterior distributions (Beta)
      </div>
      <ResponsiveContainer width="100%" height={140}>
        <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" vertical={false} />
          <XAxis
            dataKey="x" type="number" domain={[0, 1]}
            tick={TICK_STYLE} axisLine={false} tickLine={false}
            tickFormatter={v => v.toFixed(1)}
            label={{ value: 'p(reward)', position: 'insideBottomRight', offset: -4, style: TICK_STYLE }}
          />
          <YAxis tick={TICK_STYLE} axisLine={false} tickLine={false} width={30} />
          <Tooltip
            contentStyle={{ background: 'var(--color-background-primary)', border: '0.5px solid var(--color-border-tertiary)', borderRadius: 8, fontSize: 11, fontFamily: 'var(--font-mono)' }}
            formatter={(v, name) => {
              const idx = parseInt(name.replace('arm', ''))
              const label = labels && labels[idx] ? labels[idx] : `Arm ${String.fromCharCode(65 + idx)}`
              return [v.toFixed(2), label]
            }}
            labelFormatter={v => `p = ${Number(v).toFixed(2)}`}
          />
          {betaParams.map((_, i) => (
            <Area
              key={i}
              type="monotone"
              dataKey={`arm${i}`}
              stroke={ARM_COLORS[i % ARM_COLORS.length]}
              fill={ARM_COLORS[i % ARM_COLORS.length]}
              fillOpacity={0.12}
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
