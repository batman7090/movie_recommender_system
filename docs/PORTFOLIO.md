# Recruiter walkthrough

## Five-minute demonstration

1. Open the README architecture and explain the path from CSV validation to serving.
2. Run demo training with MLflow. Show the source checksum, parameters and baseline comparison.
3. Show a successful quality gate, then run with `--min-lift 1.0 --promote` and show that `current.txt` stays unchanged.
4. Open `/docs`, request recommendations, and point out the returned model version. Use the Streamlit app to show the product experience.
5. Show the test results, CI workflow, and `/metrics`. Explain rollback by pinning a previous artifact directory.

Use the synthetic catalog to demonstrate engineering only. Run and document a full-dataset experiment before claiming real-catalog quality. Show the actual CI run once this branch is pushed.

## Suggested resume bullet

Built an MLOps workflow for a content-based movie recommender with validated training data, sparse TF-IDF retrieval, MLflow experiment tracking, quality-gated model promotion, FastAPI serving, Docker deployment configuration, Prometheus metrics and 16 automated tests.

## Decisions worth discussing

- Sparse query-to-catalog scoring avoids storing an N-by-N similarity matrix, while exact search still scales with catalog feature count. Approximate nearest-neighbor indexing is a possible next step after measurement.
- Genre labels are withheld from model features and used as an explicitly limited relevance proxy. User-feedback evaluation remains future work.
- Versioned artifacts separate training from inference and make rollbacks simple. JSON and sparse NPZ avoid loading the original executable pickle payloads.
- The demonstration uses a local artifact pointer and MLflow experiment store. A managed registry is a future extension, not an implemented claim.
- CI exercises both Python behavior and container startup; deploying to a cloud provider is a separate release step.
