/**
 * lib/scenarioConfigs.js
 *
 * Ecommerce scenario definitions. Each scenario provides:
 *   - arm config (trueP, labels, prices)
 *   - contextFn: generates context per pull (null for non-contextual)
 *   - rewardFn: custom reward draw (default: Bernoulli)
 *   - rewardLabel: Y-axis label for charts
 *   - defaultPolicy: most instructive policy for this scenario
 *   - policies: subset of policies available in this scenario
 *   - description: scenario description for the UI
 *   - extraState: additional scenario-level display state
 */

// ── User types for recommendations ────────────────────────────────────────
const USER_TYPES = ['Tech', 'Fashion', 'Outdoors', 'Reader']

// Base click-through rates per arm (product category)
const RECS_BASE_P = [0.40, 0.30, 0.25, 0.35, 0.20]

// Multiplier table: [arm][userType] — from the doc
const RECS_MULTIPLIERS = [
  [1.6, 0.7, 0.8, 0.6],   // Electronics
  [0.7, 1.8, 0.9, 0.7],   // Apparel
  [0.8, 0.9, 1.4, 1.0],   // Home & Garden
  [0.9, 0.8, 1.7, 0.7],   // Sports
  [0.9, 0.7, 0.7, 1.9],   // Books
]

// ── Pricing demand curve ─────────────────────────────────────────────────
const PRICING_PRICES = [19.99, 29.99, 39.99, 49.99, 59.99]
const PRICING_PURCHASE_P = [0.75, 0.58, 0.42, 0.28, 0.15]

// ── Checkout recovery base rates and cart multipliers ─────────────────────
const CHECKOUT_BASE_P = [0.25, 0.15, 0.30, 0.20, 0.10]
// Cart value multiplier: [low cart, high cart] per arm
const CHECKOUT_CART_MULT = [
  [0.8, 1.4],   // 10% Discount — works better for high carts
  [1.3, 0.7],   // Email Reminder — works better for low carts
  [0.6, 1.6],   // Free Shipping — high-cart incentive
  [1.1, 1.0],   // Countdown Timer — roughly even
  [0.7, 1.3],   // Exit Survey — works better for high carts
]

// ── Scenario definitions ─────────────────────────────────────────────────

export const ECOMMERCE_SCENARIOS = {
  recommendations: {
    id: 'recommendations',
    name: 'Product Recommendations',
    shortName: 'Recs',
    description: 'Watch the bandit learn which product category to surface for each user type. LinUCB discovers user-type preferences; Thompson Sampling finds the best category on average.',
    defaultPolicy: 'linucb',
    contextDim: 4,
    policies: [
      { id: 'thompson-sampling', label: 'Thompson', meta: 'non-contextual', color: '#1D9E75' },
      { id: 'linucb', label: 'LinUCB', meta: 'contextual', color: '#D85A30' },
    ],
    labels: ['Electronics', 'Apparel', 'Home & Garden', 'Sports', 'Books'],
    // Average trueP for regret computation (averaged over user types)
    trueP: RECS_BASE_P.map((base, arm) => {
      const avgMult = RECS_MULTIPLIERS[arm].reduce((s, m) => s + m, 0) / USER_TYPES.length
      return Math.min(1, base * avgMult)
    }),
    rewardLabel: 'cum. clicks',

    contextFn: () => {
      const typeIdx = Math.floor(Math.random() * USER_TYPES.length)
      const oneHot = USER_TYPES.map((_, i) => i === typeIdx ? 1.0 : 0.0)
      return { context: oneHot, meta: { userType: USER_TYPES[typeIdx], userTypeIdx: typeIdx } }
    },

    rewardFn: (arm, _armConfig, _step, _driftFn, meta) => {
      const a = Math.max(0, Math.min(arm, RECS_BASE_P.length - 1))
      const typeIdx = meta?.userTypeIdx ?? 0
      const effectiveP = Math.min(1, RECS_BASE_P[a] * RECS_MULTIPLIERS[a][typeIdx])
      return Math.random() < effectiveP ? 1 : 0
    },

    // For the heatmap visualization
    getEffectiveP: (arm, userTypeIdx) => {
      return Math.min(1, RECS_BASE_P[arm] * RECS_MULTIPLIERS[arm][userTypeIdx])
    },
    userTypes: USER_TYPES,
    multipliers: RECS_MULTIPLIERS,
    baseP: RECS_BASE_P,
  },

  pricing: {
    id: 'pricing',
    name: 'Dynamic Pricing',
    shortName: 'Pricing',
    description: 'The bandit explores price points and learns the demand curve. It balances high-margin prices against volume at lower prices, optimizing total revenue.',
    defaultPolicy: 'thompson-sampling',
    policies: [
      { id: 'thompson-sampling', label: 'Thompson', meta: 'Beta posterior', color: '#1D9E75' },
      { id: 'ucb', label: 'UCB1', meta: 'confidence bound', color: '#378ADD' },
    ],
    labels: PRICING_PRICES.map(p => `$${p.toFixed(2)}`),
    // For regret: use expected revenue as trueP
    trueP: PRICING_PRICES.map((price, i) => price * PRICING_PURCHASE_P[i]),
    rewardLabel: 'cum. revenue ($)',
    isRevenue: true,
    prices: PRICING_PRICES,
    purchaseP: PRICING_PURCHASE_P,

    contextFn: null,  // Non-contextual

    rewardFn: (arm) => {
      const a = Math.max(0, Math.min(arm, PRICING_PURCHASE_P.length - 1))
      const purchased = Math.random() < PRICING_PURCHASE_P[a] ? 1 : 0
      return purchased * PRICING_PRICES[a]  // Revenue reward
    },
  },

  'checkout-recovery': {
    id: 'checkout-recovery',
    name: 'Checkout Recovery',
    shortName: 'Checkout',
    description: 'The bandit learns which intervention recovers abandoned checkouts. Cart value matters — LinUCB discovers that high-value carts respond to different tactics than low-value ones.',
    defaultPolicy: 'linucb',
    contextDim: 1,
    policies: [
      { id: 'epsilon-greedy', label: '\u03b5-Greedy', meta: '\u03b5=0.1', color: '#E89320' },
      { id: 'linucb', label: 'LinUCB', meta: 'contextual', color: '#D85A30' },
    ],
    labels: ['10% Discount', 'Email Reminder', 'Free Shipping', 'Countdown Timer', 'Exit Survey'],
    // Average trueP for regret
    trueP: CHECKOUT_BASE_P.map((base, arm) => {
      const avgMult = CHECKOUT_CART_MULT[arm].reduce((s, m) => s + m, 0) / 2
      return Math.min(1, base * avgMult)
    }),
    rewardLabel: 'cum. recoveries',

    contextFn: () => {
      const cartValue = Math.random()  // Normalized [0, 1] where 0=$0, 1=$200+
      return { context: [cartValue], meta: { cartValue, cartCategory: cartValue >= 0.5 ? 'high' : 'low' } }
    },

    rewardFn: (arm, _armConfig, _step, _driftFn, meta) => {
      const a = Math.max(0, Math.min(arm, CHECKOUT_BASE_P.length - 1))
      const cartValue = meta?.cartValue ?? 0.5
      // Interpolate between low-cart and high-cart multiplier
      const lowMult = CHECKOUT_CART_MULT[a][0]
      const highMult = CHECKOUT_CART_MULT[a][1]
      const mult = lowMult + (highMult - lowMult) * cartValue
      const effectiveP = Math.min(1, CHECKOUT_BASE_P[a] * mult)
      return Math.random() < effectiveP ? 1 : 0
    },

    cartMultipliers: CHECKOUT_CART_MULT,
    baseP: CHECKOUT_BASE_P,
  },
}

export const SCENARIO_LIST = Object.values(ECOMMERCE_SCENARIOS)
