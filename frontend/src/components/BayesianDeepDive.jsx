/**
 * components/BayesianDeepDive.jsx
 *
 * Modal/panel triggered by an "Analyze" button on each policy panel.
 * Sends the current bandit state to the PyMC analysis service and
 * renders trace plots, posterior distributions, convergence diagnostics,
 * and posterior predictive checks.
 */
import { useState, useCallback } from 'react'

const ANALYSIS_BASE = import.meta.env.VITE_ANALYSIS_URL || '/analysis'

const CONTEXTUAL_POLICIES = ['linucb', 'lints']

export default function BayesianDeepDive({ policyId, state }) {
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  const analyze = useCallback(async () => {
    setLoading(true)
    setError(null)

    try {
      let endpoint, body

      if (CONTEXTUAL_POLICIES.includes(policyId)) {
        endpoint = `${ANALYSIS_BASE}/analyze/linear`
        body = {
          A_matrices: state.A,
          b_vectors: state.b,
          n_samples: 1000,
        }
      } else {
        // Derive alpha/beta from state
        let alpha, beta
        if (state.alpha && Array.isArray(state.alpha)) {
          alpha = state.alpha
          beta = state.beta
        } else {
          // For epsilon-greedy/ucb/exp3: derive from counts
          const n = state.counts?.length || 5
          alpha = Array.from({ length: n }, (_, i) => {
            const s = Math.max(0, Math.round(state.rewards?.[i] || (state.counts[i] * (state.values?.[i] || 0))))
            return s + 1
          })
          beta = Array.from({ length: n }, (_, i) => {
            const s = alpha[i] - 1
            return (state.counts[i] - s) + 1
          })
        }

        endpoint = `${ANALYSIS_BASE}/analyze/beta`
        body = { alpha, beta, n_samples: 1000 }
      }

      const resp = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      if (!resp.ok) {
        throw new Error(`Analysis service returned ${resp.status}`)
      }

      setResult(await resp.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [policyId, state])

  const handleOpen = () => {
    setOpen(true)
    if (!result && !loading) analyze()
  }

  return (
    <div style={{ marginTop: 12 }}>
      <button
        onClick={handleOpen}
        style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 11,
          padding: '5px 12px',
          borderRadius: 'var(--border-radius-md)',
          border: '0.5px solid #9B59B6',
          background: open ? '#9B59B622' : 'transparent',
          color: '#9B59B6',
          cursor: 'pointer',
        }}
      >
        {open ? '\u25BC' : '\u25B6'} Bayesian Deep Dive
      </button>

      {open && (
        <div style={{
          marginTop: 8,
          border: '1px solid var(--color-border-secondary)',
          borderRadius: 'var(--border-radius-md)',
          background: 'var(--color-background-secondary)',
          padding: 16,
        }}>
          {loading && (
            <div style={{ textAlign: 'center', padding: 24, color: 'var(--color-text-secondary)', fontSize: 12 }}>
              Running MCMC sampling... this may take a few seconds.
            </div>
          )}

          {error && (
            <div style={{ color: 'var(--color-text-danger)', fontSize: 12, padding: 12 }}>
              Analysis service unavailable: {error}
              <br />
              <span style={{ fontSize: 10, color: 'var(--color-text-secondary)' }}>
                Start the service: docker run -p 8090:8090 bandit-analysis
              </span>
            </div>
          )}

          {result && !loading && (
            <div>
              {/* Diagnostics summary */}
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--color-text-primary)', marginBottom: 8, fontFamily: 'var(--font-mono)' }}>
                  Convergence Diagnostics
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 8 }}>
                  {result.diagnostics && Object.entries(result.diagnostics.rhat || {}).map(([name, rhat]) => {
                    const ess = result.diagnostics.ess_bulk?.[name] || 0
                    const rhatOk = rhat < 1.01
                    const essOk = ess > 400
                    return (
                      <div key={name} style={{
                        background: 'var(--color-background-primary)',
                        borderRadius: 'var(--border-radius-md)',
                        padding: '6px 10px',
                        border: `1px solid ${rhatOk && essOk ? '#30363d' : '#E74C3C44'}`,
                      }}>
                        <div style={{ fontSize: 10, color: 'var(--color-text-secondary)', marginBottom: 2, fontFamily: 'var(--font-mono)' }}>
                          {name}
                        </div>
                        <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)' }}>
                          <span style={{ color: rhatOk ? '#1D9E75' : '#E74C3C' }}>
                            R-hat: {rhat.toFixed(3)}
                          </span>
                          {' | '}
                          <span style={{ color: essOk ? '#1D9E75' : '#E74C3C' }}>
                            ESS: {Math.round(ess)}
                          </span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Trace plot */}
              {result.plots?.trace && (
                <div style={{ marginBottom: 16 }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--color-text-primary)', marginBottom: 8, fontFamily: 'var(--font-mono)' }}>
                    Trace Plot
                  </div>
                  <img
                    src={`data:image/png;base64,${result.plots.trace}`}
                    alt="Trace plot"
                    style={{ width: '100%', borderRadius: 'var(--border-radius-md)' }}
                  />
                </div>
              )}

              {/* Posterior plot */}
              {result.plots?.posterior && (
                <div style={{ marginBottom: 16 }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--color-text-primary)', marginBottom: 8, fontFamily: 'var(--font-mono)' }}>
                    Posterior Distributions
                  </div>
                  <img
                    src={`data:image/png;base64,${result.plots.posterior}`}
                    alt="Posterior distributions"
                    style={{ width: '100%', borderRadius: 'var(--border-radius-md)' }}
                  />
                </div>
              )}

              {/* PPC plot */}
              {result.plots?.ppc && (
                <div>
                  <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--color-text-primary)', marginBottom: 8, fontFamily: 'var(--font-mono)' }}>
                    Posterior Predictive Check
                  </div>
                  <img
                    src={`data:image/png;base64,${result.plots.ppc}`}
                    alt="Posterior predictive check"
                    style={{ width: '100%', borderRadius: 'var(--border-radius-md)' }}
                  />
                </div>
              )}

              {/* Re-analyze button */}
              <button
                onClick={analyze}
                disabled={loading}
                style={{
                  marginTop: 12,
                  fontFamily: 'var(--font-mono)',
                  fontSize: 10,
                  padding: '4px 10px',
                  borderRadius: 'var(--border-radius-md)',
                  border: '0.5px solid var(--color-border-secondary)',
                  background: 'transparent',
                  color: 'var(--color-text-secondary)',
                  cursor: 'pointer',
                }}
              >
                {'\u27F3'} Re-analyze with current state
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
