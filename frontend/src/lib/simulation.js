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

// Cholesky decomposition (lower triangular L such that M = L L^T)
function cholesky(M) {
  const d = M.length
  const L = Array.from({ length: d }, () => new Array(d).fill(0))
  for (let i = 0; i < d; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0
      for (let k = 0; k < j; k++) sum += L[i][k] * L[j][k]
      if (i === j) {
        L[i][j] = Math.sqrt(Math.max(0, M[i][i] - sum))
      } else {
        L[i][j] = L[j][j] > 1e-12 ? (M[i][j] - sum) / L[j][j] : 0
      }
    }
  }
  return L
}

// Sample from N(mean, cov) using Cholesky decomposition
function mvnSample(mean, cov) {
  const d = mean.length
  const L = cholesky(cov)
  const z = Array.from({ length: d }, () => {
    // Box-Muller transform for standard normal
    const u1 = Math.random(), u2 = Math.random()
    return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2)
  })
  return mean.map((m, i) => m + L[i].reduce((s, l, j) => s + l * z[j], 0))
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
      const d = armConfig.contextDim || 4
      return {
        ...base,
        d,
        alpha: 1.0,
        A: Array.from({ length: n }, () => eye(d)),
        Ainv: Array.from({ length: n }, () => eye(d)),
        b: Array.from({ length: n }, () => new Array(d).fill(0)),
        counts: new Array(n).fill(0),
        rewards: new Array(n).fill(0),
        lastCtx: null,
      }
    }
    case 'bayesian-ucb':
      return { ...base, alpha: new Array(n).fill(1), beta: new Array(n).fill(1), c: 3.0 }
    case 'exp3':
      return { ...base, weights: new Array(n).fill(1), gamma: 0.1, counts: new Array(n).fill(0), rewards: new Array(n).fill(0) }
    case 'lints': {
      const d = armConfig.contextDim || 4
      return {
        ...base,
        d,
        v: 1.0,
        A: Array.from({ length: n }, () => eye(d)),
        Ainv: Array.from({ length: n }, () => eye(d)),
        b: Array.from({ length: n }, () => new Array(d).fill(0)),
        counts: new Array(n).fill(0),
        rewards: new Array(n).fill(0),
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
      const scores = state.Ainv.map((Ainv, i) => {
        const theta = matvec(Ainv, state.b[i])
        const exploit = ctx.reduce((s, c, j) => s + c * theta[j], 0)
        const explore = state.alpha * Math.sqrt(Math.max(0, vtMv(ctx, Ainv, ctx)))
        return exploit + explore
      })
      return scores.indexOf(Math.max(...scores))
    }
    case 'bayesian-ucb': {
      const ucbs = state.alpha.map((a, i) => {
        const b = state.beta[i]
        const mean = a / (a + b)
        const ab = a + b
        const std = Math.sqrt(a * b / (ab * ab * (ab + 1)))
        return mean + state.c * std
      })
      return ucbs.indexOf(Math.max(...ucbs))
    }
    case 'exp3': {
      const wSum = state.weights.reduce((s, w) => s + w, 0)
      const n = armConfig.nArms
      const probs = state.weights.map(w => (1 - state.gamma) * w / wSum + state.gamma / n)
      // Categorical sample
      const r = Math.random()
      let cumP = 0
      for (let i = 0; i < n; i++) {
        cumP += probs[i]
        if (r < cumP) return i
      }
      return n - 1
    }
    case 'lints': {
      const ctx = context || Array.from({ length: state.d }, () => Math.random())
      state.lastCtx = ctx
      const scores = state.Ainv.map((Ainv, i) => {
        const thetaHat = matvec(Ainv, state.b[i])
        const cov = Ainv.map(row => row.map(v => v * state.v * state.v))
        const thetaSample = mvnSample(thetaHat, cov)
        return ctx.reduce((s, c, j) => s + c * thetaSample[j], 0)
      })
      return scores.indexOf(Math.max(...scores))
    }
    default:
      return Math.floor(Math.random() * armConfig.nArms)
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
      next.Ainv[arm] = inv(next.A[arm])
      next.b[arm] = state.b[arm].map((v, j) => v + reward * ctx[j])
      next.counts[arm] += 1
      next.rewards[arm] += reward
      break
    }
    case 'bayesian-ucb':
      if (reward >= 0.5) next.alpha[arm] += 1
      else next.beta[arm] += 1
      break
    case 'exp3': {
      const wSum = state.weights.reduce((s, w) => s + w, 0)
      const n = state.weights.length
      const p = (1 - state.gamma) * state.weights[arm] / wSum + state.gamma / n
      const estReward = reward / Math.max(p, 1e-10)
      next.weights[arm] = state.weights[arm] * Math.exp(state.gamma * estReward / n)
      // Normalize to prevent overflow
      const maxW = Math.max(...next.weights)
      next.weights = next.weights.map(w => w / maxW)
      next.counts[arm] += 1
      next.rewards[arm] += reward
      break
    }
    case 'lints': {
      const ctx = state.lastCtx || Array.from({ length: state.d }, () => 0.5)
      next.A[arm] = outerAdd(state.A[arm], ctx)
      next.Ainv[arm] = inv(next.A[arm])
      next.b[arm] = state.b[arm].map((v, j) => v + reward * ctx[j])
      next.counts[arm] += 1
      next.rewards[arm] += reward
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
    case 'lints':
      return state.Ainv.map((Ainv, i) => {
        try {
          const theta = matvec(Ainv, state.b[i])
          return +(theta.reduce((s, v) => s + v, 0) / state.d).toFixed(3)
        } catch { return 0 }
      })
    case 'bayesian-ucb':
      return state.alpha.map((a, i) => +(a / (a + state.beta[i])).toFixed(3))
    case 'exp3': {
      return state.counts.map((n, i) => n > 0 ? +(state.rewards[i] / n).toFixed(3) : 0)
    }
  }
}

export function getPullCounts(state) {
  switch (state.policy) {
    case 'epsilon-greedy':
    case 'ucb':
    case 'exp3':
    case 'linucb':
    case 'lints':
      return [...state.counts]
    case 'thompson-sampling':
    case 'bayesian-ucb':
      return state.alpha.map((a, i) => Math.max(0, Math.round(a + state.beta[i] - 2)))
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
    case 'bayesian-ucb':
      return state.alpha.map((a, i) => ({ alpha: a, beta: state.beta[i] }))
    case 'epsilon-greedy':
    case 'ucb':
      return state.counts.map((n, i) => {
        const s = Math.max(0, Math.round(n * state.values[i]))
        return { alpha: s + 1, beta: (n - s) + 1 }
      })
    case 'linucb':
    case 'lints':
    case 'exp3':
      return state.counts.map((n, i) => {
        const s = Math.max(0, Math.round(state.rewards[i]))
        return { alpha: s + 1, beta: (n - s) + 1 }
      })
    default:
      return []
  }
}

// ── EXP3 probability weights (for WeightsChart) ─────────────────────────────

export function getEXP3Probabilities(state) {
  if (state.policy !== 'exp3') return null
  const wSum = state.weights.reduce((s, w) => s + w, 0)
  const n = state.weights.length
  return state.weights.map(w => (1 - state.gamma) * w / wSum + state.gamma / n)
}
