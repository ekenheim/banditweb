# Bandit Demo — Operator Guide & Assumption Log

## Quick Start (first deploy)

```bash
# 1. Add Bitwarden item "bandit-demo" with fields:
#    mlflow_tracking_uri, mlflow_username, mlflow_password,
#    minio_access_key, minio_secret_key, minio_endpoint,
#    harbor_user, harbor_password

# 2. Create a MinIO bucket for artifacts
mc alias set minio https://s3.${SECRET_DOMAIN} <access_key> <secret_key>
mc mb minio/bandit-demo-artifacts 

# 3. Push manifests to your GitOps repo under kubernetes/apps/datasci/bandit-demo/
#    Flux will reconcile automatically.

# 4. Build and push model images (triggers automatically on push to main)
#    Or manually: gh workflow run build-bandit-models.yaml

# 5. Verify models are ready
kubectl get model -n datasci
kubectl get server -n datasci bandit-mlserver

# 6. Open https://bandit.${SECRET_DOMAIN}
```

## Reset an experiment mid-demo

Send a reset request to any model:

```bash
curl -X POST https://bandit.${SECRET_DOMAIN}/bandit/v2/models/epsilon-greedy/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs": [{"name": "reset", "shape": [1], "datatype": "BYTES", "data": ["1"]}]}'
```

## Change epsilon / alpha / c without rebuilding

Edit the `parameters` block in `seldon/models/models.yaml` and push.
Flux will update the Model CRD; Seldon will reload the model.

---

## Assumptions & Decisions Made

| # | Assumption | Rationale | Action if wrong |
|---|-----------|-----------|-----------------|
| 1 | Seldon v2 `seldon-mesh` service is named `seldon-mesh` in `datasci` namespace | Standard Seldon v2 Helm default | `kubectl get svc -n datasci` and update VirtualService destination host |
| 2 | Istio internal gateway service is `istio-ingressgateway-internal` | Common pattern for dual internal/external setup | Update `ingress/ingress.yaml` backend service name |
| 3 | Your ClusterSecretStore is named `bitwarden-fields` | Common ESO + Bitwarden pattern | Update `secretStoreRef.name` in all ExternalSecret objects |
| 4 | Authentik outpost is at `authentik.${SECRET_DOMAIN}` | Standard Authentik deployment pattern | Update auth-url annotation in Ingress |
| 5 | nginx-internal service account is `nginx-internal` in `ingress-nginx` namespace | Matches the dual-ingress naming pattern you described | Update AuthorizationPolicy `principals` |
| 6 | MLflow is accessible in-cluster at a service in `datasci` namespace | You confirmed MLflow is in `datasci` | Populate `mlflow_tracking_uri` in Bitwarden with the in-cluster service URL |
| 7 | Bandit model images are built once and tags match Model CRD `storageUri` | Standard pattern | Update `storageUri` in models.yaml if you use sha-tagged images instead of latest |
| 8 | LinUCB context vector is 4-dim: [time_bucket, user_segment, recency, frequency] | Reasonable demo context; frontend simulates these | Change `context_dim` parameter in linucb Model CRD and update frontend |
| 9 | Binary {0,1} rewards for all policies | Simplest demo reward signal | Thompson Sampling thresholds at 0.5 for non-binary; others accept any float |

## Known Limitations (demo scope)

- **Bandit state is in-process** — pod restart zeros all arm statistics. A Redis sidecar is the path to persistence.
- **No online learning persistence** — model parameters are not written to MinIO between restarts.
- **`latest` image tags** — fine for demo; use sha-pinned tags for any production-adjacent use.
- **LinUCB stores last context per arm in-process** — concurrent requests could cause a race on `_last_context`. The `_lock` in BanditBase protects `_update` but not the context store. Add per-request context passing for production.
