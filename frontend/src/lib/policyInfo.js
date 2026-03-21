/**
 * lib/policyInfo.js
 *
 * Static reference data for each bandit policy — descriptions,
 * real-world deployment examples, and guidance on when to use each one.
 */

const POLICY_INFO = {
  'epsilon-greedy': {
    name: 'Epsilon-Greedy',
    category: 'Classic',
    description:
      'The simplest explore/exploit strategy. Picks the best-known arm most of the time, but with probability \u03b5 (default 10%) picks a random arm instead. Easy to understand and implement, but wastes exploration budget even after the best arm is known.',
    realWorld: [
      { company: 'Spotify', useCase: 'Low-stakes playlist diversification — occasionally surfaces new tracks to keep recommendations fresh' },
      { company: 'Booking.com', useCase: 'A/B testing landing page variants with a simple fallback strategy' },
    ],
    bestFor: 'Quick prototypes, low-stakes decisions, or when simplicity matters more than optimality.',
    tradeoffs: {
      strengths: ['Dead simple to implement', 'No distributional assumptions', 'Constant-time arm selection'],
      weaknesses: ['Exploration never stops (wastes pulls on bad arms)', 'No uncertainty-aware exploration', 'Epsilon is a fixed hyperparameter that needs tuning'],
    },
  },

  ucb: {
    name: 'UCB1',
    category: 'Classic',
    description:
      'Upper Confidence Bound — picks the arm with the highest optimistic estimate. Arms that haven\'t been tried much get a large uncertainty bonus, forcing exploration. As data accumulates, bonuses shrink and it naturally focuses on the best arm. Deterministic and requires no hyperparameter tuning.',
    realWorld: [
      { company: 'Yahoo', useCase: 'News article ranking on the Yahoo! front page — systematically trying under-explored stories' },
      { company: 'Adobe', useCase: 'Automated website personalization via Adobe Target' },
    ],
    bestFor: 'When you want guaranteed logarithmic regret with no randomness in selection.',
    tradeoffs: {
      strengths: ['Theoretical regret guarantees', 'No hyperparameters (classic variant)', 'Deterministic — reproducible results'],
      weaknesses: ['Can over-explore arms that are clearly bad', 'Not adaptive to reward distribution shape', 'Slower convergence than Bayesian methods in practice'],
    },
  },

  'thompson-sampling': {
    name: 'Thompson Sampling',
    category: 'Classic',
    description:
      'The gold standard Bayesian bandit algorithm. Maintains a Beta probability distribution (belief) for each arm\'s true reward rate. Each round, draws a random sample from each arm\'s distribution and picks the highest. Naturally balances exploration and exploitation — explores where uncertain, exploits where confident.',
    realWorld: [
      { company: 'Google', useCase: 'Ad click-through rate optimization — choosing which ad creative to show' },
      { company: 'Spotify', useCase: 'Podcast recommendation personalization using Bayesian reward models' },
      { company: 'Microsoft', useCase: 'Bing search result ranking experiments' },
    ],
    bestFor: 'General-purpose bandit optimization. Fastest convergence among non-contextual methods.',
    tradeoffs: {
      strengths: ['Fastest practical convergence', 'Naturally adaptive exploration', 'Elegant Bayesian interpretation'],
      weaknesses: ['Requires sampling (non-deterministic)', 'Assumes a reward distribution (Beta for Bernoulli)', 'Slightly more complex than \u03b5-greedy'],
    },
  },

  'bayesian-ucb': {
    name: 'Bayesian UCB',
    category: 'Classic',
    description:
      'Combines UCB\'s deterministic selection with Bayesian uncertainty quantification. Uses the Beta posterior\'s credible interval (mean + c \u00d7 standard deviation) as the upper bound instead of the frequentist confidence bound. Gives the best of both worlds — principled uncertainty with deterministic arm selection.',
    realWorld: [
      { company: 'Clinical Trials', useCase: 'Adaptive dosing in pharmaceutical trials — must quantify uncertainty precisely for patient safety' },
      { company: 'VWO / Optimizely', useCase: 'A/B testing platforms that need to report credible intervals alongside decisions' },
    ],
    bestFor: 'When you need interpretable confidence bounds and want deterministic (reproducible) arm selection.',
    tradeoffs: {
      strengths: ['Principled Bayesian uncertainty', 'Deterministic selection (reproducible)', 'Credible intervals are directly interpretable'],
      weaknesses: ['Requires tuning the confidence multiplier c', 'Slightly slower convergence than Thompson Sampling', 'Still assumes Bernoulli rewards'],
    },
  },

  exp3: {
    name: 'EXP3',
    category: 'Adversarial',
    description:
      'Exponential-weight algorithm for Exploration and Exploitation — the only algorithm here that makes no assumptions about reward distributions. Designed for adversarial environments where rewards can change arbitrarily (even chosen by an opponent). Maintains probability weights over arms and uses importance-weighted updates.',
    realWorld: [
      { company: 'Ad Exchanges', useCase: 'Real-time bidding where competitor strategies shift unpredictably — no stationarity assumption' },
      { company: 'Dynamic Pricing', useCase: 'Pricing against strategic buyers who adapt their behavior to your prices' },
      { company: 'Network Routing', useCase: 'Load balancing across servers where performance characteristics change adversarially' },
    ],
    bestFor: 'Non-stationary or adversarial environments where reward distributions may shift arbitrarily over time.',
    tradeoffs: {
      strengths: ['No distributional assumptions — works against adversaries', 'Worst-case regret guarantees', 'Handles non-stationarity naturally'],
      weaknesses: ['Slower convergence in stochastic settings (pays a price for adversarial robustness)', 'Exploration parameter \u03b3 needs tuning', 'Doesn\'t produce posterior distributions'],
    },
  },

  linucb: {
    name: 'LinUCB',
    category: 'Contextual',
    description:
      'Linear Upper Confidence Bound — a contextual bandit that learns that the best arm depends on who\'s asking. Fits a linear reward model per arm using ridge regression, then selects the arm with the highest upper confidence bound given the current context (user features, time of day, etc.).',
    realWorld: [
      { company: 'Netflix', useCase: 'Artwork personalization — showing different movie posters to different user segments' },
      { company: 'Microsoft (MSN)', useCase: 'Personalized news article recommendations on the MSN homepage' },
      { company: 'Yahoo', useCase: 'The original LinUCB paper evaluated on Yahoo! Today Module news recommendations' },
    ],
    bestFor: 'When the best arm depends on user features or context. Heterogeneous populations.',
    tradeoffs: {
      strengths: ['Personalizes decisions per user/context', 'UCB-style deterministic exploration', 'Handles cold-start via feature generalization'],
      weaknesses: ['Assumes linear reward model (may underfit complex relationships)', 'Requires meaningful context features', 'Higher memory per arm (stores d\u00d7d matrix)'],
    },
  },

  lints: {
    name: 'Linear Thompson Sampling',
    category: 'Contextual',
    description:
      'Combines Thompson Sampling\'s Bayesian exploration with a linear contextual reward model. Like LinUCB, it learns a weight vector per arm — but instead of using a deterministic upper bound, it samples the weight vector from its posterior distribution. This gives more adaptive exploration in contextual settings.',
    realWorld: [
      { company: 'Amazon', useCase: 'Personalized product recommendations factoring in user browsing history and demographics' },
      { company: 'LinkedIn', useCase: 'Job recommendation feeds personalized by user profile features' },
      { company: 'Alibaba', useCase: 'E-commerce display ad selection with user context features' },
    ],
    bestFor: 'Contextual settings where you want Thompson-style exploration (faster convergence) instead of UCB-style bounds.',
    tradeoffs: {
      strengths: ['Bayesian exploration adapts naturally to uncertainty', 'Often faster convergence than LinUCB in practice', 'Shares linear model benefits (feature generalization, cold-start)'],
      weaknesses: ['Non-deterministic (harder to reproduce exact runs)', 'Posterior sampling adds computational cost', 'Same linear assumption limitation as LinUCB'],
    },
  },
}

export default POLICY_INFO
