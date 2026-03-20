# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: build the Vite SPA
FROM node:22-alpine AS builder

WORKDIR /app
COPY package.json ./
RUN npm install --frozen-lockfile

COPY . .

# Build args injected by GitHub Actions / Flux image automation
ARG VITE_API_BASE=/bandit
ARG VITE_MLFLOW_URL
ARG VITE_POLICIES=epsilon-greedy,ucb,thompson-sampling,linucb
ARG VITE_N_ARMS=5
ARG VITE_SIMULATE=false
ARG VITE_ARM_PROBS=0.55,0.35,0.7,0.25,0.45

ENV VITE_API_BASE=$VITE_API_BASE \
    VITE_MLFLOW_URL=$VITE_MLFLOW_URL \
    VITE_POLICIES=$VITE_POLICIES \
    VITE_N_ARMS=$VITE_N_ARMS \
    VITE_SIMULATE=$VITE_SIMULATE \
    VITE_ARM_PROBS=$VITE_ARM_PROBS

RUN npm run build

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: serve with NGINX
FROM nginx:1.27-alpine

COPY --from=builder /app/dist /usr/share/nginx/html

# NGINX config: serve SPA, proxy /bandit to Seldon via env-injected upstream
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Health check endpoint used by Kubernetes liveness probe
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s \
  CMD wget -qO- http://localhost:3000/healthz || exit 1

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]
