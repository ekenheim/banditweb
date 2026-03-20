/**
 * lib/simulation.js
 *
 * Client-side bandit implementations that mirror the Python model logic.
 * Used when VITE_SIMULATE=true or when Seldon endpoints are unreachable.
 * Arm reward probabilities are fixed per-session; operators can override
 * via the VITE_ARM_PROBS env var (comma-separated, e.g. "0.55,0.35,0.7,0.25,0.45").
 */

const RAW = import.meta.env.VITE_ARM_PROBS
export const DEFAULT_TRUE_P = RAW
  ? RAW.split(',').map(Number)
  : [0.55, 0.35, 0.7, 0.25, 0.45]

// Backward-compat aliases
export const TRUE_P = DEFAULT_TRUE_P
export const N_ARMS = DEFAULT_TRUE_P.length
export const OPTIMAL = Math.max(...DEFAULT_TRUE_P)

// ── Arm config helper ────────────────────────────────────────────────────────

export function makeArmConfig(trueP) {
  return { trueP, nArms: trueP.length, optimal: Math.max(...trueP) }
}

export const DEFAULT_CONFIG = makeArmConfig(DEFAULT_TRUE_P)

// ── Helpers ──────────────────────────────────────────────────────────────────

export function betaSample(alpha, beta) {
  const x = gammaSample(alpha)
  const y = gammaSample(beta)
  return x / (x + y + 1e-10)
}

export function gammaSample(shape) {
  if (shape < 1) return gammaSample(1 + shape) * Math.pow(Math.random(), 1 / shape)
  const d = shape - 1 / 3
  const c = 1 / Math.sqrt(9 * d)
  for (;;) {
    let x, v
    do { x = (Math.random() * 2 - 1) * 3; v = 1 + c * x } while (v <= 0)
    v = v * v * v
    const u = Math.random()
    if (u < 1 - 0.0331 * x * x * x * x) return d * v
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v
  }
}

function eye(d) {
  return Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? 1 : 0)))
}

function matvec(M, v) {
  return M.map(row => row.reduce((s, x, j) => s + x * v[j], 0))
}

function outerAdd(A, x) {
  return A.map((row, i) => row.map((v, j) => v + x[i] * x[j]))
}

function vtMv(v, M, u) {
  return v.reduce((s, vi, i) => s + vi * M[i].reduce((ss, mij, j) => ss + mij * u[j], 0), 0)
}

// Gauss-Jordan inverse (good enough for d≤8 in a demo)
function inv(M) {
  const d = M.length
  const aug = M.map((r, i) => [...r, ...Array.from({ length: d }, (_, j) => (i === j ? 1 : 0))])
  for (let i = 0; i < d; i++) {
    let pivot = i
    for (let k = i + 1; k < d; k++) if (Math.abs(aug[k][i]) > Math.abs(aug[pivot][i])) pivot = k
    ;[aug[i], aug[pivot]] = [aug[pivot], aug[i]]
    const sc = aug[i][i]
    if (Math.abs(sc) < 1e-12) continue
    for (let j = i; j < 2 * d; j++) aug[i][j] /= sc
    for (let k = 0; k < d; k++) {
      if (k === i) continue
      const f = aug[k][i]
      for (let j = i; j < 2 * d; j++) aug[k][j] -= f * aug[i][j]
    }
  }
  return aug.map(r => r.slice(d))
}

// ── Initial states ────────────────────────────────────────────────────────────

export function makeState(policy, armConfig = DEFAULT_CONFIG) {
  const n = armConfig.nArms
  const base = { policy, pulls: 0, cumR: 0, cumReg: 0, hist: [], lastArm: null }
  switch (policy) {
    case 'epsilon-greedy':
      return { ...base, counts: new Array(n).fill(0), values: new Array(n).fill(0) }
    case 'ucb':
      return { ...base, counts: new Array(n).fill(0), values: new Array(n).fill(0), total: 0 }
    case 'thompson-sampling':
      return { ...base, alpha: new Array(n).fill(1), beta: new Array(n).fill(1) }
    case 'linucb': {
      const d = 4
      return {
        ...base,
        d,
        alpha: 1.0,
        A: Array.from({ length: n }, () => eye(d)),
        b: Array.from({ length: n }, () => new Array(d).fill(0)),
        lastCtx: null,
      }
    }
    default:
      throw new Error(`Unknown policy: ${policy}`)
  }
}

// ── Arm selection ─────────────────────────────────────────────────────────────

export function selectArm(state, context = null, armConfig = DEFAULT_CONFIG) {
  switch (state.policy) {
    case 'epsilon-greedy': {
      if (Math.random() < 0.1) return Math.floor(Math.random() * armConfig.nArms)
      return state.values.indexOf(Math.max(...state.values))
    }
    case 'ucb': {
      const ucbs = state.values.map((v, i) =>
        state.counts[i] === 0 ? Infinity : v + 2 * Math.sqrt(Math.log(Math.max(state.total, 1)) / state.counts[i])
      )
      return ucbs.indexOf(Math.max(...ucbs))
    }
    case 'thompson-sampling': {
      const samples = state.alpha.map((a, i) => betaSample(a, state.beta[i]))
      return samples.indexOf(Math.max(...samples))
    }
    case 'linucb': {
      const ctx = context || Array.from({ length: state.d }, () => Math.random())
      state.lastCtx = ctx
      const scores = state.A.map((A, i) => {
        const Ainv = inv(A)
        const theta = matvec(Ainv, state.b[i])
        const exploit = ctx.reduce((s, c, j) => s + c * theta[j], 0)
        const explore = state.alpha * Math.sqrt(Math.max(0, vtMv(ctx, Ainv, ctx)))
        return exploit + explore
      })
      return scores.indexOf(Math.max(...scores))
    }
  }
}

// ── State update after reward ─────────────────────────────────────────────────

export function updateState(state, arm, reward, armConfig = DEFAULT_CONFIG, driftFn = null) {
  const next = structuredClone(state)
  const effectiveP = driftFn ? driftFn(armConfig.trueP, state.pulls) : armConfig.trueP
  const effectiveOptimal = driftFn ? Math.max(...effectiveP) : armConfig.optimal
  const meanP = effectiveP.reduce((a, b) => a + b, 0) / effectiveP.length
  next.pulls += 1
  next.cumR += reward
  next.cumReg += effectiveOptimal - effectiveP[arm]
  next.lastArm = arm
  const trimAt = 200
  next.hist = [...state.hist, { t: next.pulls, cumR: next.cumR, cumReg: next.cumReg, cumRandom: next.pulls * meanP }].slice(-trimAt)

  switch (state.policy) {
    case 'epsilon-greedy':
    case 'ucb':
      next.counts[arm] += 1
      if (state.policy === 'ucb') next.total += 1
      next.values[arm] += (reward - state.values[arm]) / next.counts[arm]
      break
    case 'thompson-sampling':
      if (reward >= 0.5) next.alpha[arm] += 1
      else next.beta[arm] += 1
      break
    case 'linucb': {
      const ctx = state.lastCtx || Array.from({ length: state.d }, () => 0.5)
      next.A[arm] = outerAdd(state.A[arm], ctx)
      next.b[arm] = state.b[arm].map((v, j) => v + reward * ctx[j])
      break
    }
  }
  return next
}

// ── Derived values for display ────────────────────────────────────────────────

export function getEstimatedValues(state) {
  switch (state.policy) {
    case 'epsilon-greedy':
    case 'ucb':
      return state.values.map(v => +v.toFixed(3))
    case 'thompson-sampling':
      return state.alpha.map((a, i) => +(a / (a + state.beta[i])).toFixed(3))
    case 'linucb':
      return state.A.map((A, i) => {
        try {
          const theta = matvec(inv(A), state.b[i])
          return +(theta.reduce((s, v) => s + v, 0) / state.d).toFixed(3)
        } catch { return 0 }
      })
  }
}

export function getPullCounts(state) {
  switch (state.policy) {
    case 'epsilon-greedy':
    case 'ucb':
      return [...state.counts]
    case 'thompson-sampling':
      return state.alpha.map((a, i) => Math.max(0, Math.round(a + state.beta[i] - 2)))
    case 'linucb':
      return new Array(state.A.length).fill(0)  // LinUCB doesn't track counts directly
  }
}

// ── Simulate a Bernoulli reward draw ─────────────────────────────────────────

export function drawReward(arm, armConfig = DEFAULT_CONFIG, step = 0, driftFn = null) {
  const effectiveP = driftFn ? driftFn(armConfig.trueP, step) : armConfig.trueP
  return Math.random() < effectiveP[arm] ? 1 : 0
}

// ── Beta posterior params for any policy ─────────────────────────────────────

export function getBetaParams(state) {
  switch (state.policy) {
    case 'thompson-sampling':
      return state.alpha.map((a, i) => ({ alpha: a, beta: state.beta[i] }))
    case 'epsilon-greedy':
    case 'ucb':
      return state.counts.map((n, i) => {
        const s = Math.max(0, Math.round(n * state.values[i]))
        return { alpha: s + 1, beta: (n - s) + 1 }
      })
    case 'linucb':
      return state.A.map(() => ({ alpha: 1, beta: 1 }))
    default:
      return []
  }
}
