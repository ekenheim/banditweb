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
 *
 * Scenario hooks:
 *   - scenarioContextFn: () => { context, meta } — custom context generator
 *   - scenarioRewardFn: (arm, armConfig, step, driftFn, meta) => reward
 */
import { useState, useCallback, useRef, useEffect } from 'react'
import * as api from '../lib/api'
import * as sim from '../lib/simulation'

const SIMULATE = import.meta.env.VITE_SIMULATE === 'true'

export function useBandit(policy, armConfig = sim.DEFAULT_CONFIG, driftFn = null, scenarioContextFn = null, scenarioRewardFn = null) {
  const [state, setState] = useState(() => sim.makeState(policy, armConfig))
  const [status, setStatus] = useState('idle')  // 'idle' | 'loading' | 'error'
  const [error, setError] = useState(null)
  const [lastMeta, setLastMeta] = useState(null)  // scenario context metadata
  const lastContext = useRef(null)
  const prevNArms = useRef(armConfig.nArms)
  const prevPolicy = useRef(policy)

  // Auto-reset when arm count or policy changes
  useEffect(() => {
    if (armConfig.nArms !== prevNArms.current || policy !== prevPolicy.current) {
      setState(sim.makeState(policy, armConfig))
      setStatus('idle')
      setError(null)
      setLastMeta(null)
      prevNArms.current = armConfig.nArms
      prevPolicy.current = policy
    }
  }, [armConfig.nArms, policy, armConfig])

  const makeContext = useCallback(() => {
    // Scenario-provided context takes precedence
    if (scenarioContextFn) {
      const result = scenarioContextFn()
      return result  // { context, meta }
    }
    // Default context for linucb/lints (existing behavior)
    if (policy !== 'linucb' && policy !== 'lints') return { context: null, meta: null }
    return {
      context: [
        Math.round((new Date().getHours() / 23) * 10) / 10,
        Math.round(Math.random() * 10) / 10,
        Math.round(Math.random() * 10) / 10,
        Math.round(Math.random() * 10) / 10,
      ],
      meta: null,
    }
  }, [policy, scenarioContextFn])

  const pull = useCallback(async (forceArm = null) => {
    setStatus('loading')
    setError(null)

    try {
      let arm, reward

      // Use simulation when explicitly set, when arm config differs from backend default,
      // when drift is active, or when scenario has custom reward logic
      const useLocalSim = SIMULATE || armConfig.nArms !== sim.DEFAULT_CONFIG.nArms || driftFn !== null || scenarioRewardFn !== null

      const { context: ctx, meta } = makeContext()
      lastContext.current = ctx
      setLastMeta(meta)

      if (useLocalSim) {
        arm = forceArm ?? sim.selectArm(state, ctx, armConfig)
        reward = scenarioRewardFn
          ? scenarioRewardFn(arm, armConfig, state.pulls, driftFn, meta)
          : sim.drawReward(arm, armConfig, state.pulls, driftFn)
        setState(prev => sim.updateState(prev, arm, reward, armConfig, driftFn))
      } else {
        if (forceArm !== null) {
          arm = forceArm
        } else {
          const selection = await api.selectArm(policy, ctx)
          arm = selection.arm
        }
        reward = scenarioRewardFn
          ? scenarioRewardFn(arm, armConfig, state.pulls, driftFn, meta)
          : sim.drawReward(arm, armConfig, state.pulls, driftFn)
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
  }, [policy, state, makeContext, armConfig, driftFn, scenarioRewardFn])

  const reset = useCallback(async () => {
    setState(sim.makeState(policy, armConfig))
    setStatus('idle')
    setError(null)
    setLastMeta(null)
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
    lastMeta,
    estimatedValues: sim.getEstimatedValues(state),
    pullCounts: sim.getPullCounts(state),
    betaParams: sim.getBetaParams(state),
    isSimulating: SIMULATE || armConfig.nArms !== sim.DEFAULT_CONFIG.nArms || driftFn !== null || scenarioRewardFn !== null,
  }
}
