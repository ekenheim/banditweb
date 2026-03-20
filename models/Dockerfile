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
COPY _base/__init__.py  ./bandit_runtime/__init__.py 2>/dev/null || true

# MLServer model settings — tells MLServer which class to load
COPY ${POLICY}/model-settings.json /app/model-settings.json

# MLServer picks up settings from this env var
ENV MLSERVER_MODEL_IMPLEMENTATION="bandit_runtime.${POLICY}.$(python3 -c \
  "import importlib, inspect; m=importlib.import_module('bandit_runtime.${POLICY}'); \
   print([n for n,c in inspect.getmembers(m, inspect.isclass) if n != 'BanditBase'][0])")"

EXPOSE 8080 8081

CMD ["mlserver", "start", "/app"]
