/**
 * components/Charts.jsx
 * Recharts visualisations:
 *   1. Arm selection frequency (BarChart)
 *   2. Cumulative reward + regret + random baseline over time (LineChart)
 */
import React from 'react'
import {
  BarChart, Bar, LineChart, Line,
  XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from 'recharts'

const TICK_STYLE = { fontSize: 10, fontFamily: 'var(--font-mono)', fill: 'var(--color-text-secondary)' }

export default function Charts({ state, values, counts, color, labels }) {
  const armLabels = counts.map((_, i) => labels && labels[i] ? labels[i] : String.fromCharCode(65 + i))
  const freqData = counts.map((n, i) => ({ arm: armLabels[i], pulls: n, value: values[i] }))

  // Subsample history to at most 100 points for performance
  const hist = state.hist
  const step = Math.max(1, Math.floor(hist.length / 100))
  const histData = hist.filter((_, i) => i % step === 0 || i === hist.length - 1)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Arm frequency */}
      <div>
        <div style={{ fontSize: 11, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)', marginBottom: 6 }}>
          arm selection frequency
        </div>
        <ResponsiveContainer width="100%" height={140}>
          <BarChart data={freqData} barSize={28}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" vertical={false} />
            <XAxis dataKey="arm" tick={TICK_STYLE} axisLine={false} tickLine={false} />
            <YAxis tick={TICK_STYLE} axisLine={false} tickLine={false} width={30} />
            <Tooltip
              contentStyle={{ background: 'var(--color-background-primary)', border: '0.5px solid var(--color-border-tertiary)', borderRadius: 8, fontSize: 11, fontFamily: 'var(--font-mono)' }}
              formatter={(v, name) => [v, name === 'pulls' ? 'pulls' : 'est. value']}
            />
            <Bar dataKey="pulls" fill={color} opacity={0.75} radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Cumulative reward vs regret vs random baseline */}
      <div>
        <div style={{ fontSize: 11, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)', marginBottom: 6 }}>
          cumulative reward vs. regret
        </div>
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={histData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" vertical={false} />
            <XAxis dataKey="t" tick={TICK_STYLE} axisLine={false} tickLine={false} label={{ value: 'step', position: 'insideBottomRight', offset: -4, style: TICK_STYLE }} />
            <YAxis tick={TICK_STYLE} axisLine={false} tickLine={false} width={36} />
            <Tooltip
              contentStyle={{ background: 'var(--color-background-primary)', border: '0.5px solid var(--color-border-tertiary)', borderRadius: 8, fontSize: 11, fontFamily: 'var(--font-mono)' }}
              formatter={v => [typeof v === 'number' ? v.toFixed(2) : v]}
            />
            <Line type="monotone" dataKey="cumR" name="reward" stroke="#1D9E75" dot={false} strokeWidth={1.5} />
            <Line type="monotone" dataKey="cumReg" name="regret" stroke="#E24B4A" dot={false} strokeWidth={1.5} strokeDasharray="4 2" />
            <Line type="monotone" dataKey="cumRandom" name="random (A/B)" stroke="#8b949e" dot={false} strokeWidth={1} strokeDasharray="2 4" opacity={0.6} />
            <Legend
              wrapperStyle={{ fontSize: 10, fontFamily: 'var(--font-mono)', paddingTop: 4 }}
              formatter={v => {
                if (v === 'cumR') return 'cum. reward'
                if (v === 'cumReg') return 'cum. regret'
                if (v === 'cumRandom') return 'random (A/B)'
                return v
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
