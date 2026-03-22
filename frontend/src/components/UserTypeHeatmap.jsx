/**
 * components/UserTypeHeatmap.jsx
 * 5x4 grid showing estimated vs true p(click) for each arm/user-type combination.
 * Used in the Product Recommendations scenario.
 */
import React from 'react'

const CELL_SIZE = 56

function colorScale(value) {
  // Green intensity based on value (0 to 1)
  const g = Math.round(80 + value * 140)
  const r = Math.round(40 + (1 - value) * 60)
  return `rgb(${r}, ${g}, 80)`
}

export default function UserTypeHeatmap({ scenario, estimatedValues, pullCounts, lastMeta }) {
  const { userTypes, labels, getEffectiveP } = scenario
  if (!userTypes || !getEffectiveP) return null

  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ fontSize: 11, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)', marginBottom: 8 }}>
        true p(click) by user type
        {lastMeta?.userType && (
          <span style={{ marginLeft: 8, color: '#D85A30' }}>
            current user: {lastMeta.userType}
          </span>
        )}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ borderCollapse: 'collapse', fontFamily: 'var(--font-mono)', fontSize: 10 }}>
          <thead>
            <tr>
              <th style={{ padding: '4px 8px', textAlign: 'left', color: 'var(--color-text-secondary)' }}></th>
              {userTypes.map((ut, j) => (
                <th key={j} style={{
                  padding: '4px 8px',
                  textAlign: 'center',
                  color: lastMeta?.userType === ut ? '#D85A30' : 'var(--color-text-secondary)',
                  fontWeight: lastMeta?.userType === ut ? 700 : 400,
                }}>
                  {ut}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {labels.map((armLabel, arm) => (
              <tr key={arm}>
                <td style={{ padding: '4px 8px', color: 'var(--color-text-secondary)', whiteSpace: 'nowrap' }}>
                  {armLabel}
                </td>
                {userTypes.map((ut, typeIdx) => {
                  const p = getEffectiveP(arm, typeIdx)
                  const isHighlighted = lastMeta?.userType === ut
                  return (
                    <td key={typeIdx} style={{
                      padding: 2,
                      textAlign: 'center',
                    }}>
                      <div style={{
                        width: CELL_SIZE,
                        height: 32,
                        borderRadius: 4,
                        background: colorScale(p),
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: '#fff',
                        fontWeight: 600,
                        fontSize: 11,
                        opacity: isHighlighted ? 1 : 0.7,
                        border: isHighlighted ? '2px solid #D85A30' : '2px solid transparent',
                        transition: 'all .15s',
                      }}>
                        {p.toFixed(2)}
                      </div>
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Estimated values from the bandit */}
      <div style={{ fontSize: 11, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)', marginTop: 12, marginBottom: 6 }}>
        bandit's estimated value per arm (averaged over contexts)
      </div>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {labels.map((armLabel, i) => (
          <div key={i} style={{
            background: 'var(--color-background-secondary)',
            borderRadius: 'var(--border-radius-md)',
            padding: '4px 10px',
            fontSize: 10,
            fontFamily: 'var(--font-mono)',
          }}>
            <span style={{ color: 'var(--color-text-secondary)' }}>{armLabel}:</span>{' '}
            <strong>{estimatedValues[i]?.toFixed(3) ?? '—'}</strong>
            <span style={{ color: 'var(--color-text-secondary)', marginLeft: 4 }}>
              ({pullCounts[i] ?? 0} pulls)
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
