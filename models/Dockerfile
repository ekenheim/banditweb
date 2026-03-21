# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# Bandit MLServer runtime image
#
# Usage (build each policy by passing BUILD_ARG):
#   docker build --build-arg POLICY=epsilon_greedy -t bandit-epsilon-greedy .
#   docker build --build-arg POLICY=ucb            -t bandit-ucb .
#   docker build --build-arg POLICY=thompson_sampling -t bandit-thompson-sampling .
#   docker build --build-arg POLICY=linucb         -t bandit-linucb .
# ─────────────────────────────────────────────────────────────────────────────
ARG MLSERVER_VERSION=1.6.0
FROM docker.io/seldonio/mlserver:${MLSERVER_VERSION}-slim

ARG POLICY
ENV POLICY=${POLICY}

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the shared base and the specific policy module
COPY _base/          ./bandit_runtime/
COPY ${POLICY}/model.py ./bandit_runtime/${POLICY}.py

EXPOSE 8080 8081

# Map policy to class name
ENV MLSERVER_MODEL_IMPLEMENTATION=""
RUN case "${POLICY}" in \
      epsilon_greedy)     CLASS=EpsilonGreedyModel ;; \
      ucb)                CLASS=UCBModel ;; \
      thompson_sampling)  CLASS=ThompsonSamplingModel ;; \
      linucb)             CLASS=LinUCBModel ;; \
      bayesian_ucb)       CLASS=BayesianUCBModel ;; \
      exp3)               CLASS=EXP3Model ;; \
      lints)              CLASS=LinTSModel ;; \
      *) echo "Unknown policy: ${POLICY}" && exit 1 ;; \
    esac \
    && echo "MLSERVER_MODEL_IMPLEMENTATION=bandit_runtime.${POLICY}.${CLASS}" > /app/.env

CMD ["sh", "-c", ". /app/.env && exec mlserver start /app"]
