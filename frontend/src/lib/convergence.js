/**
 * lib/convergence.js
 * Bayesian stopping criteria via Monte Carlo sampling from Beta posteriors.
 */
import { betaSample } from './simulation'

export function computePBest(betaParams, nSamples = 5000) {
  const k = betaParams.length
  const wins = new Array(k).fill(0)

  for (let s = 0; s < nSamples; s++) {
    let bestVal = -1
    let bestArm = 0
    for (let i = 0; i < k; i++) {
      const sample = betaSample(betaParams[i].alpha, betaParams[i].beta)
      if (sample > bestVal) {
        bestVal = sample
        bestArm = i
      }
    }
    wins[bestArm]++
  }

  return wins.map(w => w / nSamples)
}

export function computeExpectedLoss(betaParams, nSamples = 5000) {
  const k = betaParams.length
  const losses = new Array(k).fill(0)

  for (let s = 0; s < nSamples; s++) {
    const samples = betaParams.map(p => betaSample(p.alpha, p.beta))
    const bestVal = Math.max(...samples)
    for (let i = 0; i < k; i++) {
      losses[i] += bestVal - samples[i]
    }
  }

  return losses.map(l => l / nSamples)
}
