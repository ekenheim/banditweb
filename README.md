# Multi-Armed Bandit Interactive Demo

An interactive platform for exploring multi-armed bandit algorithms. Ten bandit policies run as MLServer runtimes behind Seldon Core v2 on Kubernetes, with MLflow experiment tracking and a React frontend that lets you pull arms, observe convergence, and compare policies side-by-side.

The demo includes seven core algorithm implementations and three ecommerce learning scenarios that show how bandits solve real business problems: product recommendations, dynamic pricing, and checkout recovery.

Everything runs in simulation mode without a cluster — set `VITE_SIMULATE=true` and the full demo works in-browser.

## Quick Start

```bash
cd frontend
npm install
VITE_SIMULATE=true npm run dev    # offline mode, no cluster needed
```

Open `http://localhost:3000`. All policies run client-side in JavaScript.

To connect to a live Seldon cluster:

```bash
VITE_DEV_CLUSTER=http://your-cluster:8080 npm run dev
```

## The Seven Core Policies

The Classic Demo tab lets you experiment with all seven algorithms on configurable arm probabilities.

### Classic

| Policy | How It Works | When To Use |
|--------|-------------|-------------|
| **Epsilon-Greedy** | Exploits the best-known arm most of the time, explores randomly with probability epsilon (10%). | Quick prototypes. Simple to understand and implement. |
| **UCB1** | Picks the arm with the highest optimistic estimate. Under-explored arms get a confidence bonus that shrinks with data. Deterministic. | When you want guaranteed log regret with no randomness. |
| **Thompson Sampling** | Maintains a Beta posterior per arm. Samples from each, picks the highest. Naturally explores where uncertain. | General-purpose. Fastest convergence among non-contextual methods. |
| **Bayesian UCB** | Uses the Beta posterior's credible interval (mean + c * std) as the upper bound. Bayesian uncertainty with deterministic selection. | When you need interpretable confidence bounds and reproducibility. |

### Adversarial

| Policy | How It Works | When To Use |
|--------|-------------|-------------|
| **EXP3** | Exponential-weight algorithm. No distributional assumptions. Maintains probability weights, uses importance-weighted updates. | Non-stationary or adversarial environments where rewards shift arbitrarily. |

### Contextual

| Policy | How It Works | When To Use |
|--------|-------------|-------------|
| **LinUCB** | Fits a per-arm linear reward model via ridge regression. Selects the arm with the highest UCB given the current context vector. | When the best arm depends on user features (personalization). |
| **LinTS** | Like LinUCB but samples weight vectors from the posterior instead of using a deterministic bound. | Contextual settings where you want Thompson-style adaptive exploration. |

## Ecommerce Scenarios

Three interactive scenarios demonstrate how bandits solve real business problems. Each scenario is a self-contained learning environment with domain-realistic reward distributions, custom visualizations, and a curated set of policies that best illustrate the learning dynamics.

### Product Recommendations

**What it teaches:** How a contextual bandit personalizes content for different user segments — the mechanism behind recommendation engines.

**Setup:** Five product category arms (Electronics, Apparel, Home & Garden, Sports, Books) and four simulated user types (Tech, Fashion, Outdoors, Reader). Each pull generates a random user type. Click-through probability depends on both the arm and the user type via a multiplier table:

| Category | Base CTR | Tech | Fashion | Outdoors | Reader |
|----------|---------|------|---------|----------|--------|
| Electronics | 0.40 | 1.6x | 0.7x | 0.8x | 0.6x |
| Apparel | 0.30 | 0.7x | 1.8x | 0.9x | 0.7x |
| Home & Garden | 0.25 | 0.8x | 0.9x | 1.4x | 1.0x |
| Sports | 0.35 | 0.9x | 0.8x | 1.7x | 0.7x |
| Books | 0.20 | 0.9x | 0.7x | 0.7x | 1.9x |

**Policies available:** Thompson Sampling (non-contextual) and LinUCB (contextual).

**What to watch:** Thompson Sampling converges on Electronics (best average CTR) but treats all users the same. LinUCB, fed the user type as a one-hot context vector, learns to recommend Sports to Outdoors users and Books to Readers. The regret curves visibly diverge after ~100 pulls as LinUCB starts exploiting the context signal.

**How to interpret the heatmap:** The 5x4 grid shows the true effective click probability for every arm/user-type combination. The highlighted column tracks the current user type. Watch LinUCB's estimated values converge toward the column maximums rather than the row averages — that's personalization happening in real time.

### Dynamic Pricing

**What it teaches:** How a bandit explores price points and discovers the demand curve — the core of revenue optimization.

**Setup:** Five price point arms for a fictional product ($19.99 to $59.99). Purchase probability follows a monotone demand curve — higher prices mean fewer buyers:

| Price | p(purchase) | Expected Revenue |
|-------|------------|-----------------|
| $19.99 | 0.75 | $15.00 |
| $29.99 | 0.58 | $17.39 |
| $39.99 | 0.42 | $16.80 |
| $49.99 | 0.28 | $14.00 |
| $59.99 | 0.15 | $9.00 |

The reward is **revenue** (price x purchase), not a binary signal. The $29.99 arm has the highest expected revenue despite not having the highest conversion rate.

**Policies available:** Thompson Sampling and UCB1.

**What to watch:** The bandit must balance the exploration-exploitation tension concretely: cheap arms convert often (safe revenue) but expensive arms have higher per-sale margin. Watch the algorithm discover that $29.99 is optimal — not the cheapest, not the most expensive, but the revenue sweet spot.

**How to interpret the revenue chart:** Blue bars show the bandit's current estimate of expected revenue per price point. Click "reveal demand curve" to overlay the true values. The gap between estimated and true narrows as the bandit gathers data. If the bandit's best estimate matches the true optimum, it has learned the demand curve.

### Checkout Recovery

**What it teaches:** How a contextual bandit learns that different interventions work for different customer segments — specifically that cart value predicts which recovery tactic succeeds.

**Setup:** Five intervention arms for recovering abandoned checkouts: 10% Discount, Email Reminder, Free Shipping, Countdown Timer, Exit Survey. Each pull generates a random cart value (normalized $0-$200). Recovery probability depends on both the intervention and the cart value:

| Intervention | Base Rate | Low Cart Mult | High Cart Mult |
|-------------|-----------|--------------|----------------|
| 10% Discount | 0.25 | 0.8x | 1.4x |
| Email Reminder | 0.15 | 1.3x | 0.7x |
| Free Shipping | 0.30 | 0.6x | 1.6x |
| Countdown Timer | 0.20 | 1.1x | 1.0x |
| Exit Survey | 0.10 | 0.7x | 1.3x |

Free Shipping dominates for high-value carts; Email Reminder works for low-value ones. A non-contextual policy can't see this.

**Policies available:** Epsilon-Greedy (non-contextual) and LinUCB (contextual, 1-dim cart value context).

**What to watch:** Epsilon-Greedy converges on Free Shipping (best average rate). LinUCB learns the cart-value relationship and starts picking Email Reminder for low carts and Free Shipping for high carts. The regret gap grows over time because epsilon-greedy keeps making suboptimal choices for half the population.

**How to interpret the cart breakdown chart:** The side-by-side bars show true recovery rates split by low vs high cart value. The cart badge shows the current pull's cart category and dollar amount. Watch LinUCB's intervention cards — its estimated values should diverge from the simple averages as it builds separate models for different cart values.

## Classic Demo Features

### Arm Configuration

The collapsible configurator panel lets you:

- Choose from preset scenarios (Ad Headlines, Product Recs, Pricing Tiers, Clinical Trial, or Custom)
- Add/remove arms (2-10)
- Adjust individual arm probabilities via sliders
- Enable drift patterns that change probabilities over time:
  - **Best Degrades** — the top arm slowly loses effectiveness
  - **Rotating Best** — the optimal arm shifts every 100 pulls
  - **Random Walk** — sinusoidal drift per arm

### Policy Race

The Race view runs all seven policies simultaneously on the same arm configuration, plotting cumulative regret curves on a single chart. This is the fastest way to see which algorithm converges first for a given problem structure.

### Bayesian Deep Dive

For Thompson Sampling and Bayesian UCB, a deep dive modal connects to the analysis service (PyMC) for MCMC trace plots, R-hat diagnostics, and posterior predictive checks.

## Architecture

```
bandit-demo/
├── frontend/          React + Vite SPA (Recharts visualizations)
├── models/            10 self-contained MLServer policy runtimes
│   ├── epsilon_greedy/
│   ├── ucb/
│   ├── thompson_sampling/
│   ├── bayesian_ucb/
│   ├── exp3/
│   ├── linucb/
│   ├── lints/
│   ├── recommendations/
│   ├── pricing/
│   └── checkout_recovery/
├── analysis/          PyMC Bayesian analysis microservice (FastAPI)
└── .github/workflows/ CI: model upload to S3, frontend + analysis Docker builds
```

### Model Layer

Each policy is a self-contained `model.py` with `BanditBase` inlined (MLServer caches Python modules by name, so shared imports would collide). The `predict()` endpoint multiplexes three operations via input name:

| Input Name | Operation | Returns |
|-----------|-----------|---------|
| `context` | Select an arm (optionally with context vector) | `{ arm, step }` |
| `reward` | Update state with `[arm, reward]` | Updated state metrics |
| `reset` | Zero state and start new MLflow run | `{ status: "ok" }` |

Policies implement four methods: `_init_state`, `_select_arm`, `_update`, `state_dict`.

### Frontend Layer

- **React 18** with **Recharts** for visualization
- **`useBandit` hook** — central state machine per policy. Handles live (Seldon) and simulate (in-browser JS) modes transparently.
- **`simulation.js`** — full JavaScript reimplementation of all seven core policies, used when `VITE_SIMULATE=true` or when the backend is unreachable.
- **Scenario system** — ecommerce scenarios provide custom `contextFn` (context generation) and `rewardFn` (domain-realistic rewards) that plug into `useBandit`.
- All policy panels in Classic Demo are always mounted (hidden via CSS) to preserve per-tab experiment state across tab switches.

### Production Deployment

Models deploy as Python files to MinIO S3 (not Docker images). Seldon Core v2's rclone agent fetches them. CI auto-detects changed model directories and uploads.

| Component | Stack |
|-----------|-------|
| Runtime | MLServer 1.6+ on Seldon Core v2 |
| Tracking | MLflow |
| Artifacts | MinIO S3 |
| Orchestration | Kubernetes (datasci namespace) |
| Ingress | Istio VirtualService + nginx + Authentik auth |
| CI/CD | GitHub Actions + Flux GitOps |

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `VITE_SIMULATE` | Run entirely in-browser, no cluster | `false` |
| `VITE_DEV_CLUSTER` | Proxy target for Seldon endpoints | `http://localhost:8080` |
| `VITE_ARM_PROBS` | Override default arm probabilities (comma-separated) | `0.55,0.35,0.7,0.25,0.45` |
| `VITE_MLFLOW_URL` | Link to MLflow UI | `/mlflow` |
| `VITE_ANALYSIS_URL` | PyMC analysis service URL | `http://localhost:8090` |
| `MLFLOW_TRACKING_URI` | MLflow server for model pods | `http://mlflow:5000` |

## Development

### Frontend

```bash
cd frontend
npm install
npm run dev              # dev server on :3000
npm run build            # production build to dist/
```

### Analysis Service

```bash
cd analysis
docker build -t bandit-analysis .
docker run -p 8090:8090 bandit-analysis
```

### Models (local Docker, not used in production)

```bash
docker build --build-arg POLICY=epsilon_greedy -t bandit-epsilon-greedy ./models
```

### Upload a Model to S3

```bash
mc cp models/epsilon_greedy/model.py minio/bandit-models/epsilon_greedy/model.py
mc cp models/epsilon_greedy/model-settings.json minio/bandit-models/epsilon_greedy/model-settings.json
```

### CI

Workflows trigger automatically on pushes to `main`:

- `models/**` changes: validates Python syntax, uploads to MinIO S3, triggers Flux reconcile
- `frontend/**` changes: builds Docker image, pushes to Harbor
- `analysis/**` changes: builds Docker image, pushes to Harbor

Manual dispatch: `gh workflow run build-bandit-models.yaml`

## Key Design Decisions

- **Inlined BanditBase**: Every `model.py` contains its own copy of the base class. MLServer caches Python modules by name, so a shared `_base` package would cause only the first-loaded version to be used across all models.
- **Client-side rewards**: Rewards are Bernoulli draws generated in the browser (or revenue for pricing). The backend never generates rewards — it only selects arms and updates state. This keeps the simulation/live modes symmetric.
- **In-process state**: Bandit state lives in model pod memory. Pod restart resets everything. This is intentional for a demo — no Redis or database needed.
- **Always-mounted panels**: Policy panels in Classic Demo are hidden via CSS, not unmounted. This preserves experiment state when switching tabs without needing external state management.

## Dependencies

**Frontend**: React 18, React DOM, Recharts, Vite

**Models**: MLServer 1.6+, MLflow 2.13+, NumPy, scikit-learn, boto3

**Analysis**: PyMC 5.10+, ArviZ 0.17+, FastAPI, Uvicorn
