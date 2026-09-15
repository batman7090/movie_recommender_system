# Operations runbook

## Serving a specific version

The API loads a model once at startup. `MODEL_DIR` overrides the `MODEL_ROOT/current.txt` pointer. Missing or corrupt artifacts fail startup. Hashes detect accidental corruption; use trusted storage because hashes are not signatures.

```powershell
$env:MODEL_DIR = "models/<known-good-version>"
python -m uvicorn recommender.api:app --host 127.0.0.1 --port 8000
```

For rollback, stop the API and start it with the previous validated `MODEL_DIR`. Check `/ready` and a known recommendation. Streamlit caches the catalog for five minutes; clear its cache or restart it after changing catalogs. Remove the environment override before switching back to the current pointer.

Promotion is explicit (`--promote`) and only happens after gates and requested MLflow logging succeed. Concurrent promotions use atomic replacement, with the last successful writer winning. Serialize release jobs if multiple operators train models.

## Drift check

```powershell
python -m recommender.drift --model-dir models/<version> --data data/incoming_movies.csv --threshold 0.2 --report reports/drift.json
```

Exit code 0 means below threshold; 2 indicates drift. The report compares normalized genre-label frequencies using total variation distance (0 means identical distributions; 1 means disjoint distributions). It also reports catalog sizes. This only detects genre composition shifts, not text drift or changing user preferences. Review the new data, retrain a candidate, compare evaluations, then promote deliberately. Drift does not automatically trigger retraining.

## Prometheus queries

Request rate:

```promql
sum(rate(recommendation_requests_total[5m]))
```

Unknown-movie request rate:

```promql
rate(recommendation_requests_total{outcome="unknown_movie"}[5m])
```

Approximate p95 inference latency:

```promql
histogram_quantile(0.95, sum by (le) (rate(recommendation_latency_seconds_bucket[5m])))
```

The histogram measures recommendation handler work; it excludes network transit and framework validation. Counters track successful and unknown-ID recommendation requests, not every HTTP route or 422 response. Compose uses one API process; multi-process serving requires Prometheus multiprocess configuration or per-instance scraping. Alert delivery is not configured.

## Full-data container serving

Train on the host with the Python environment and quality thresholds. Build the image, then mount the generated models read-only:

```powershell
docker build -t movie-recommender:local .
docker run --rm -p 127.0.0.1:8000:8000 --mount "type=bind,source=$PWD/models,target=/app/models,readonly" movie-recommender:local
```

The default Compose stack is for synthetic demos and creates its own model volume. Do not use its training service to promote a full-data release.

## Scope and release checklist

- Retain the exact source CSV and model directory in versioned storage.
- Commit the training code before a release; the recorded Git revision may otherwise precede local edits.
- Direct dependencies are pinned; transitive dependencies and the base image are not fully locked. Add platform-specific lockfiles and image digests for stricter builds.
- Test and record full-catalog quality and latency before making performance claims.
- Add authentication, TLS, request limits and managed storage before internet exposure. Local services bind to loopback by default.
- No cloud deployment, model registry service, distributed orchestration, or user-feedback pipeline is configured.

The original large pickle files remain tracked through Git LFS. CI skips LFS downloads because the pipeline regenerates its own models. Removing those historical files requires a separate repository cleanup decision.
