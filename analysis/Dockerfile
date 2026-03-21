# ─────────────────────────────────────────────────────────────────────────────
# Bandit Bayesian Analysis service (PyMC + ArviZ + FastAPI)
#
# Usage:
#   docker build -t bandit-analysis ./analysis
#   docker run -p 8090:8090 bandit-analysis
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PyMC
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models.py .
COPY app.py .

EXPOSE 8090

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8090"]
