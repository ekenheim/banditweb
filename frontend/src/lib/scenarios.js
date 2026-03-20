/**
 * lib/scenarios.js
 * Pre-built scenario presets and drift patterns for the bandit demo.
 */
import { DEFAULT_TRUE_P } from './simulation'

export const SCENARIOS = [
  {
    id: 'custom',
    name: 'Custom',
    description: 'Configure your own arms',
    trueP: DEFAULT_TRUE_P,
    labels: null,
  },
  {
    id: 'ad-headlines',
    name: 'Ad Headlines',
    description: 'Which headline gets more clicks?',
    trueP: [0.12, 0.08, 0.15, 0.06],
    labels: ['Summer Sale!', 'Free Shipping', 'Limited Time', 'New Arrivals'],
  },
  {
    id: 'product-recs',
    name: 'Product Recs',
    description: 'Personalize the storefront',
    trueP: [0.25, 0.18, 0.32, 0.15, 0.28, 0.10],
    labels: ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Toys'],
  },
  {
    id: 'email-subjects',
    name: 'Email Subjects',
    description: 'Optimize open rates',
    trueP: [0.22, 0.18, 0.25],
    labels: ['Urgent', 'Personal', 'Informative'],
  },
  {
    id: 'pricing',
    name: 'Pricing Tiers',
    description: 'Which price point converts?',
    trueP: [0.30, 0.45, 0.35, 0.20, 0.10],
    labels: ['$9.99', '$14.99', '$19.99', '$24.99', '$29.99'],
  },
  {
    id: 'clinical',
    name: 'Clinical Trial',
    description: 'Adaptive treatment allocation',
    trueP: [0.40, 0.55, 0.35],
    labels: ['Treatment A', 'Treatment B', 'Placebo'],
  },
]

export const DRIFT_PATTERNS = [
  {
    id: 'none',
    name: 'Stationary',
    description: 'Probabilities stay fixed',
    fn: null,
  },
  {
    id: 'best-degrades',
    name: 'Best Degrades',
    description: 'Top arm slowly loses effectiveness',
    fn: (baseP, step) => {
      const bestVal = Math.max(...baseP)
      return baseP.map(p => {
        if (p === bestVal) return Math.max(0.05, p - step * 0.0008)
        return Math.min(0.95, p + step * 0.0002)
      })
    },
  },
  {
    id: 'rotation',
    name: 'Rotating Best',
    description: 'Best arm shifts every 100 pulls',
    fn: (baseP, step) => {
      const cycle = Math.floor(step / 100) % baseP.length
      return baseP.map((_, i) => i === cycle ? 0.75 : 0.20)
    },
  },
  {
    id: 'random-walk',
    name: 'Random Walk',
    description: 'Arms drift gradually (seeded)',
    fn: (baseP, step) => {
      // Deterministic sinusoidal drift — each arm oscillates at a different frequency
      return baseP.map((p, i) => {
        const drift = Math.sin(step * 0.02 + i * 1.8) * 0.15
        return Math.max(0.05, Math.min(0.95, p + drift))
      })
    },
  },
]
