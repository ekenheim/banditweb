/**
 * components/CartBreakdownChart.jsx
 * Side-by-side bars for low-cart and high-cart estimated recovery rates.
 * Used in the Checkout Recovery scenario.
 */
import React from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from 'recharts'

const TICK_STYLE = { fontSize: 10, fontFamily: 'var(--font-mono)', fill: 'var(--color-text-secondary)' }

export default function CartBreakdownChart({ scenario, estimatedValues, pullCounts, lastMeta }) {
  const { labels, baseP, cartMultipliers } = scenario

  // Build data showing true low/high cart recovery rates
  const data = labels.map((label, i) => ({
    arm: label.length > 12 ? label.slice(0, 11) + '\u2026' : label,
    fullLabel: label,
    lowCart: +(baseP[i] * cartMultipliers[i][0]).toFixed(3),
    highCart: +(baseP[i] * cartMultipliers[i][1]).toFixed(3),
    estimated: estimatedValues[i] ?? 0,
    pulls: pullCounts[i] ?? 0,
  }))

  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <span style={{ fontSize: 11, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)' }}>
          true recovery rate by cart value
        </span>
        {lastMeta?.cartCategory && (
          <span style={{
            fontSize: 10, fontFamily: 'var(--font-mono)', padding: '2px 8px',
            borderRadius: 'var(--border-radius-md)',
            background: lastMeta.cartCategory === 'high' ? '#D85A3018' : '#378ADD18',
            color: lastMeta.cartCategory === 'high' ? '#D85A30' : '#378ADD',
          }}>
            cart: {lastMeta.cartCategory} (${Math.round(lastMeta.cartValue * 200)})
          </span>
        )}
      </div>

      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={data} barSize={14} barGap={2}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" vertical={false} />
          <XAxis dataKey="arm" tick={TICK_STYLE} axisLine={false} tickLine={false} />
          <YAxis tick={TICK_STYLE} axisLine={false} tickLine={false} width={36} domain={[0, 'auto']} />
          <Tooltip
            contentStyle={{ background: 'var(--color-background-primary)', border: '0.5px solid var(--color-border-tertiary)', borderRadius: 8, fontSize: 11, fontFamily: 'var(--font-mono)' }}
            formatter={(v, name) => [v.toFixed(3), name === 'lowCart' ? 'low cart p(recover)' : 'high cart p(recover)']}
          />
          <Legend
            wrapperStyle={{ fontSize: 10, fontFamily: 'var(--font-mono)', paddingTop: 4 }}
          />
          <Bar dataKey="lowCart" name="low cart" fill="#378ADD" opacity={0.7} radius={[3, 3, 0, 0]} />
          <Bar dataKey="highCart" name="high cart" fill="#D85A30" opacity={0.7} radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>

      {/* Intervention cards */}
      <div style={{ fontSize: 11, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)', marginTop: 12, marginBottom: 6 }}>
        intervention performance
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 6 }}>
        {labels.map((label, i) => (
          <div key={i} style={{
            background: 'var(--color-background-secondary)',
            borderRadius: 'var(--border-radius-md)',
            padding: '8px 10px',
            fontSize: 10,
            fontFamily: 'var(--font-mono)',
          }}>
            <div style={{ fontWeight: 600, marginBottom: 4, fontSize: 11 }}>{label}</div>
            <div>est: <strong>{(estimatedValues[i] ?? 0).toFixed(3)}</strong></div>
            <div style={{ color: 'var(--color-text-secondary)' }}>{pullCounts[i] ?? 0} pulls</div>
          </div>
        ))}
      </div>
    </div>
  )
}
