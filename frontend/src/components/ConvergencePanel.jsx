/**
 * components/ConvergencePanel.jsx
 * Shows P(best arm) per arm, expected loss, and a "Test Complete" badge.
 */
import React, { useRef } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell, ReferenceLine,
} from 'recharts'
import { computePBest, computeExpectedLoss } from '../lib/convergence'

const ARM_COLORS = [
  '#E89320', '#378ADD', '#1D9E75', '#D85A30', '#9B59B6',
  '#E74C3C', '#2ECC71', '#3498DB', '#F39C12', '#1ABC9C',
]

const TICK_STYLE = { fontSize: 10, fontFamily: 'var(--font-mono)', fill: 'var(--color-text-secondary)' }

export default function ConvergencePanel({ betaParams, pulls, labels, threshold = 0.95, onThresholdChange }) {
  // Recompute every 10 pulls using a ref to hold the last result
  const lastComputed = useRef({ at: -1, pBest: [], loss: [] })
  const bucket = Math.floor(pulls / 10)

  if (pulls < 1) {
    lastComputed.current = { at: -1, pBest: betaParams.map(() => 1 / betaParams.length), loss: betaParams.map(() => 0) }
  } else if (bucket !== lastComputed.current.at) {
    lastComputed.current = { at: bucket, pBest: computePBest(betaParams), loss: computeExpectedLoss(betaParams) }
  }

  const { pBest, loss } = lastComputed.current

  const leaderIdx = pBest.indexOf(Math.max(...pBest))
  const converged = pBest[leaderIdx] >= threshold

  const data = pBest.map((p, i) => {
    const label = labels && labels[i] ? labels[i] : `Arm ${String.fromCharCode(65 + i)}`
    return { arm: label, pBest: p, idx: i }
  })

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
        <span style={{ fontSize: 11, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)' }}>
          convergence — P(best arm)
        </span>
        {converged && (
          <span style={{
            fontSize: 9, fontFamily: 'var(--font-mono)', padding: '2px 8px',
            borderRadius: 'var(--border-radius-md)',
            background: '#1D9E7522', color: '#1D9E75', border: '0.5px solid #1D9E75',
          }}>
            TEST COMPLETE
          </span>
        )}
        {pulls > 0 && (
          <span style={{ fontSize: 9, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)', marginLeft: 'auto' }}>
            leader loss: {loss[leaderIdx]?.toFixed(4) || '—'}
          </span>
        )}
      </div>

      <ResponsiveContainer width="100%" height={Math.max(80, data.length * 28 + 20)}>
        <BarChart data={data} layout="vertical" barSize={16} margin={{ top: 0, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" horizontal={false} />
          <XAxis type="number" domain={[0, 1]} tick={TICK_STYLE} axisLine={false} tickLine={false} tickFormatter={v => (v * 100).toFixed(0) + '%'} />
          <YAxis type="category" dataKey="arm" tick={TICK_STYLE} axisLine={false} tickLine={false} width={80} />
          <Tooltip
            contentStyle={{ background: 'var(--color-background-primary)', border: '0.5px solid var(--color-border-tertiary)', borderRadius: 8, fontSize: 11, fontFamily: 'var(--font-mono)' }}
            formatter={(v) => [(v * 100).toFixed(1) + '%', 'P(best)']}
          />
          <ReferenceLine x={threshold} stroke="#1D9E75" strokeDasharray="4 2" strokeWidth={1.5} />
          <Bar dataKey="pBest" radius={[0, 3, 3, 0]}>
            {data.map((d, i) => (
              <Cell key={i} fill={ARM_COLORS[d.idx % ARM_COLORS.length]} opacity={d.pBest >= threshold ? 1 : 0.5} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Threshold slider */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
        <span style={{ fontSize: 9, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)' }}>threshold</span>
        <input
          type="range" min="0.80" max="0.99" step="0.01"
          value={threshold}
          onChange={e => onThresholdChange(parseFloat(e.target.value))}
          style={{ width: 80, height: 3 }}
        />
        <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>
          {(threshold * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  )
}
