/**
 * lib/beta.js
 * Beta PDF computation for posterior visualization.
 * Uses Lanczos approximation for log-gamma.
 */

const LANCZOS_G = 7
const LANCZOS_C = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028,
  771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
]

export function lnGamma(z) {
  if (z < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - lnGamma(1 - z)
  }
  z -= 1
  let x = LANCZOS_C[0]
  for (let i = 1; i < LANCZOS_G + 2; i++) {
    x += LANCZOS_C[i] / (z + i)
  }
  const t = z + LANCZOS_G + 0.5
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x)
}

export function betaPDF(x, a, b) {
  if (x <= 0 || x >= 1) return 0
  if (a <= 0 || b <= 0) return 0
  const lnB = lnGamma(a) + lnGamma(b) - lnGamma(a + b)
  return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - lnB)
}

export function computePDFCurves(betaParams, nPoints = 100) {
  const data = []
  for (let i = 1; i < nPoints; i++) {
    const x = i / nPoints
    const point = { x }
    for (let k = 0; k < betaParams.length; k++) {
      const { alpha, beta } = betaParams[k]
      point[`arm${k}`] = betaPDF(x, alpha, beta)
    }
    data.push(point)
  }
  return data
}
