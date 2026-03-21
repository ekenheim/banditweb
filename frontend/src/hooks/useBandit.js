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
import { useState, useCallback, useRef, useEffect } from 'react'
import * as api from '../lib/api'
import * as sim from '../lib/simulation'

const SIMULATE = import.meta.env.VITE_SIMULATE === 'true'

export function useBandit(policy, armConfig = sim.DEFAULT_CONFIG, driftFn = null) {
  const [state, setState] = useState(() => sim.makeState(policy, armConfig))
  const [status, setStatus] = useState('idle')  // 'idle' | 'loading' | 'error'
  const [error, setError] = useState(null)
  const lastContext = useRef(null)
  const prevNArms = useRef(armConfig.nArms)

  // Auto-reset when arm count changes
  useEffect(() => {
    if (armConfig.nArms !== prevNArms.current) {
      setState(sim.makeState(policy, armConfig))
      setStatus('idle')
      setError(null)
      prevNArms.current = armConfig.nArms
    }
  }, [armConfig.nArms, policy, armConfig])

  const makeContext = useCallback(() => {
    if (policy !== 'linucb' && policy !== 'lints') return null
    return [
      Math.round((new Date().getHours() / 23) * 10) / 10,
      Math.round(Math.random() * 10) / 10,
      Math.round(Math.random() * 10) / 10,
      Math.round(Math.random() * 10) / 10,
    ]
  }, [policy])

  const pull = useCallback(async (forceArm = null) => {
    setStatus('loading')
    setError(null)

    try {
      let arm, reward

      // Use simulation when explicitly set, when arm config differs from backend default,
      // or when drift is active (backend doesn't support drift)
      const useLocalSim = SIMULATE || armConfig.nArms !== sim.DEFAULT_CONFIG.nArms || driftFn !== null

      if (useLocalSim) {
        const ctx = makeContext()
        lastContext.current = ctx
        arm = forceArm ?? sim.selectArm(state, ctx, armConfig)
        reward = sim.drawReward(arm, armConfig, state.pulls, driftFn)
        setState(prev => sim.updateState(prev, arm, reward, armConfig, driftFn))
      } else {
        const ctx = makeContext()
        lastContext.current = ctx
        if (forceArm !== null) {
          arm = forceArm
        } else {
          const selection = await api.selectArm(policy, ctx)
          arm = selection.arm
        }
        reward = sim.drawReward(arm, armConfig, state.pulls, driftFn)
        await api.submitReward(policy, arm, reward)
        setState(prev => sim.updateState(prev, arm, reward, armConfig, driftFn))
      }

      setStatus('idle')
      return { arm, reward }
    } catch (err) {
      setStatus('error')
      setError(err.message)
      console.error(`[${policy}] pull error:`, err)
      return null
    }
  }, [policy, state, makeContext, armConfig, driftFn])

  const reset = useCallback(async () => {
    setState(sim.makeState(policy, armConfig))
    setStatus('idle')
    setError(null)
    if (!SIMULATE) {
      try { await api.resetPolicy(policy) } catch (e) { console.warn('reset failed', e) }
    }
  }, [policy, armConfig])

  return {
    state,
    status,
    error,
    pull,
    reset,
    estimatedValues: sim.getEstimatedValues(state),
    pullCounts: sim.getPullCounts(state),
    betaParams: sim.getBetaParams(state),
    isSimulating: SIMULATE || armConfig.nArms !== sim.DEFAULT_CONFIG.nArms || driftFn !== null,
  }
}
