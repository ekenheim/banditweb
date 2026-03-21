/**
 * lib/api.js
 * Thin wrapper around the Seldon v2 inference protocol REST endpoints.
 *
 * All bandit models share the same request/response schema;
 * only the model name in the URL changes.
 *
 * Seldon v2 V2 protocol reference:
 *   POST /v2/models/{model_name}/infer
 *   Body: { inputs: [{ name, shape, datatype, data }] }
 */

const BASE = import.meta.env.VITE_API_BASE || '/bandit'

/** Map policy slug → Seldon model name (matches Model CRD metadata.name) */
const MODEL_NAMES = {
  'epsilon-greedy': 'epsilon-greedy',
  ucb: 'ucb',
  'thompson-sampling': 'thompson-sampling',
  linucb: 'linucb',
  'bayesian-ucb': 'bayesian-ucb',
  exp3: 'exp3',
  lints: 'lints',
}

async function post(policy, body) {
  const model = MODEL_NAMES[policy]
  const url = `${BASE}/v2/models/${model}/infer`
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Seldon error ${res.status}: ${text}`)
  }
  return res.json()
}

function parseOutputs(response) {
  return Object.fromEntries(
    (response.outputs || []).map(o => [o.name, o.data?.[0] ?? o.data])
  )
}

/**
 * Request an arm selection from the given policy.
 * For LinUCB a context vector must be supplied.
 *
 * @param {string} policy  - e.g. 'epsilon-greedy'
 * @param {number[]|null} context - feature vector (required for linucb)
 * @returns {Promise<{arm: number, step: number}>}
 */
export async function selectArm(policy, context = null) {
  const inputs = []

  if (context) {
    inputs.push({
      name: 'context',
      shape: [context.length],
      datatype: 'FP64',
      data: context,
    })
  } else {
    // Non-contextual policies: send an empty context placeholder so the
    // model knows this is a select call and not a reward update.
    inputs.push({
      name: 'context',
      shape: [0],
      datatype: 'FP64',
      data: [],
    })
  }

  const resp = await post(policy, { inputs })
  return parseOutputs(resp)
}

/**
 * Submit a reward signal for a previously selected arm.
 *
 * @param {string} policy
 * @param {number} arm     - 0-indexed arm that was pulled
 * @param {number} reward  - observed reward (typically 0 or 1)
 * @returns {Promise<object>} updated state metrics
 */
export async function submitReward(policy, arm, reward) {
  const resp = await post(policy, {
    inputs: [
      {
        name: 'reward',
        shape: [2],
        datatype: 'FP64',
        data: [arm, reward],
      },
    ],
  })
  return parseOutputs(resp)
}

/**
 * Reset a policy's state and start a new MLflow run.
 *
 * @param {string} policy
 * @returns {Promise<{status: string, step: number}>}
 */
export async function resetPolicy(policy) {
  const resp = await post(policy, {
    inputs: [
      {
        name: 'reset',
        shape: [1],
        datatype: 'BYTES',
        data: ['1'],
      },
    ],
  })
  return parseOutputs(resp)
}

/**
 * Check liveness of a model pod.
 * Uses the Seldon v2 model ready endpoint.
 */
export async function checkReady(policy) {
  const model = MODEL_NAMES[policy]
  const url = `${BASE}/v2/models/${model}/ready`
  const res = await fetch(url)
  return res.ok
}
