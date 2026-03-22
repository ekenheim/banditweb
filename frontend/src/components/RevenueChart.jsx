/**
 * components/RevenueChart.jsx
 * Bar chart showing expected revenue per price point.
 * Used in the Dynamic Pricing scenario.
 */
import React, { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell,
} from 'recharts'

const TICK_STYLE = { fontSize: 10, fontFamily: 'var(--font-mono)', fill: 'var(--color-text-secondary)' }

export default function RevenueChart({ scenario, estimatedValues, pullCounts }) {
  const [showTruth, setShowTruth] = useState(false)
  const { labels, prices, purchaseP } = scenario

  // Build chart data
  const data = labels.map((label, i) => {
    const trueRevenue = prices[i] * purchaseP[i]
    const estRevenue = estimatedValues[i] ?? 0
    return {
      arm: label,
      estimated: +estRevenue.toFixed(2),
      true: +trueRevenue.toFixed(2),
      pulls: pullCounts[i] ?? 0,
      purchaseRate: purchaseP[i],
    }
  })

  const bestEstIdx = estimatedValues.indexOf(Math.max(...estimatedValues))
  const bestTrueIdx = data.reduce((best, d, i) => d.true > data[best].true ? i : best, 0)

  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <span style={{ fontSize: 11, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)' }}>
          expected revenue per price point
        </span>
        <button
          onClick={() => setShowTruth(!showTruth)}
          style={{
            fontSize: 10, fontFamily: 'var(--font-mono)', padding: '2px 8px',
            borderRadius: 'var(--border-radius-md)',
            border: '0.5px solid var(--color-border-secondary)',
            background: showTruth ? '#1D9E7518' : 'transparent',
            color: showTruth ? '#1D9E75' : 'var(--color-text-secondary)',
            cursor: 'pointer',
          }}
        >
          {showTruth ? 'hide truth' : 'reveal demand curve'}
        </button>
      </div>

      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={data} barSize={24} barGap={4}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" vertical={false} />
          <XAxis dataKey="arm" tick={TICK_STYLE} axisLine={false} tickLine={false} />
          <YAxis tick={TICK_STYLE} axisLine={false} tickLine={false} width={36} label={{ value: '$', position: 'insideTopLeft', style: TICK_STYLE }} />
          <Tooltip
            contentStyle={{ background: 'var(--color-background-primary)', border: '0.5px solid var(--color-border-tertiary)', borderRadius: 8, fontSize: 11, fontFamily: 'var(--font-mono)' }}
            formatter={(v, name) => [`$${v}`, name === 'estimated' ? 'est. revenue' : 'true revenue']}
          />
          <Bar dataKey="estimated" fill="#378ADD" opacity={0.8} radius={[3, 3, 0, 0]}>
            {data.map((_, i) => (
              <Cell key={i} fill={i === bestEstIdx ? '#1D9E75' : '#378ADD'} />
            ))}
          </Bar>
          {showTruth && (
            <Bar dataKey="true" fill="#8b949e" opacity={0.5} radius={[3, 3, 0, 0]}>
              {data.map((_, i) => (
                <Cell key={i} fill={i === bestTrueIdx ? '#1D9E75' : '#8b949e'} opacity={0.5} />
              ))}
            </Bar>
          )}
        </BarChart>
      </ResponsiveContainer>

      {/* Demand curve table */}
      {showTruth && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 10, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)', marginBottom: 4 }}>
            true demand curve (revealed)
          </div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {labels.map((label, i) => (
              <div key={i} style={{
                background: 'var(--color-background-secondary)',
                borderRadius: 'var(--border-radius-md)',
                padding: '4px 8px',
                fontSize: 10,
                fontFamily: 'var(--font-mono)',
                border: i === bestTrueIdx ? '1px solid #1D9E75' : '1px solid transparent',
              }}>
                {label}: p={purchaseP[i].toFixed(2)} rev=${(prices[i] * purchaseP[i]).toFixed(2)}
                {i === bestTrueIdx && <span style={{ color: '#1D9E75', marginLeft: 4 }}>optimal</span>}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
