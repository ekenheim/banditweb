/**
 * hooks/useBandit.js
 *
 * Central state hook for one bandit policy.
 *
 * In LIVE mode  (VITE_SIMULATE !== 'true'):
 *   - Calls Seldon /infer for arm selection
 *   - Simulates Bernoulli reward client-side
 *   - Posts reward back to Seldon for state update + MLflow logging
 *
 * In SIMULATE mode:
 *   - Runs the JS simulation entirely in the browser
 *   - Identical UI, no cluster required
 */
import { useState, useCallback, useRef } from 'react'
import * as api from '../lib/api'
import * as sim from '../lib/simulation'

const SIMULATE = import.meta.env.VITE_SIMULATE === 'true'

export function useBandit(policy) {
  const [state, setState] = useState(() => sim.makeState(policy))
  const [status, setStatus] = useState('idle')  // 'idle' | 'loading' | 'error'
  const [error, setError] = useState(null)
  const lastContext = useRef(null)

  const makeContext = useCallback(() => {
    if (policy !== 'linucb') return null
    // Simulate 4-dim context: [time_bucket, user_segment, recency, frequency]
    // In production the frontend would derive these from real user signals
    return [
      Math.round((new Date().getHours() / 23) * 10) / 10,  // time of day
      Math.round(Math.random() * 10) / 10,                  // simulated segment
      Math.round(Math.random() * 10) / 10,                  // recency
      Math.round(Math.random() * 10) / 10,                  // frequency
    ]
  }, [policy])

  const pull = useCallback(async (forceArm = null) => {
    setStatus('loading')
    setError(null)

    try {
      let arm, reward

      if (SIMULATE) {
        const ctx = makeContext()
        lastContext.current = ctx
        arm = forceArm ?? sim.selectArm(state, ctx)
        reward = sim.drawReward(arm)
        setState(prev => sim.updateState(prev, arm, reward))
      } else {
        // 1. Select arm from Seldon
        const ctx = makeContext()
        lastContext.current = ctx
        if (forceArm !== null) {
          arm = forceArm
        } else {
          const selection = await api.selectArm(policy, ctx)
          arm = selection.arm
        }

        // 2. Simulate reward (Bernoulli draw based on hidden true probabilities)
        reward = sim.drawReward(arm)

        // 3. Submit reward to Seldon (updates model state + logs to MLflow)
        await api.submitReward(policy, arm, reward)

        // 4. Mirror state locally for chart rendering
        setState(prev => sim.updateState(prev, arm, reward))
      }

      setStatus('idle')
      return { arm, reward }
    } catch (err) {
      setStatus('error')
      setError(err.message)
      console.error(`[${policy}] pull error:`, err)
      return null
    }
  }, [policy, state, makeContext])

  const reset = useCallback(async () => {
    setState(sim.makeState(policy))
    setStatus('idle')
    setError(null)
    if (!SIMULATE) {
      try { await api.resetPolicy(policy) } catch (e) { console.warn('reset failed', e) }
    }
  }, [policy])

  return {
    state,
    status,
    error,
    pull,
    reset,
    estimatedValues: sim.getEstimatedValues(state),
    pullCounts: sim.getPullCounts(state),
    isSimulating: SIMULATE,
  }
}
