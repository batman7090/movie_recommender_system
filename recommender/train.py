"""Validate, train, evaluate and optionally promote an immutable model version."""
import argparse
import csv
import json
import os
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from recommender.model import Recommender, digest


def read_movies(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"id", "title", "overview", "genres"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns: {sorted(required)}")
        movies, seen = [], set()
        for row_number, row in enumerate(reader, 2):
            try:
                movie_id = int(row["id"])
                title, overview = row["title"].strip(), row["overview"].strip()
                genres = json.loads(row["genres"])
                if not isinstance(genres, list):
                    raise ValueError("genres must be a JSON list")
                genres = [g["name"] if isinstance(g, dict) else g for g in genres]
                if any(not isinstance(g, str) or not g.strip() for g in genres):
                    raise ValueError("invalid genre name")
                if movie_id <= 0 or movie_id in seen or not title or not overview:
                    raise ValueError("positive unique ID, title and overview required")
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                raise ValueError(f"Invalid movie at CSV row {row_number}: {exc}") from exc
            seen.add(movie_id)
            movies.append({"id": movie_id, "title": title, "overview": overview, "genres": genres})
    if len(movies) < 3:
        raise ValueError("At least three movies are required")
    return sorted(movies, key=lambda movie: movie["id"])


def evaluate(model, k=5, seed=42, sample_size=500):
    rng = np.random.default_rng(seed)
    eligible = [m for m in model.movies if m["genres"]]
    if not eligible:
        raise ValueError("Evaluation requires movies with genre labels")
    queries = rng.choice(len(eligible), min(sample_size, len(eligible)), replace=False)
    precision, baseline, coverage = [], [], set()
    for index in queries:
        movie = eligible[index]
        recommendations = model.recommend(movie["id"], k)
        candidates = [m for m in model.movies if m["id"] != movie["id"]]
        random_indices = rng.choice(len(candidates), len(recommendations), replace=False)
        def relevant(item):
            return bool(set(movie["genres"]) & set(item["genres"]))
        precision.append(np.mean([relevant(m) for m in recommendations]))
        baseline.append(np.mean([relevant(candidates[i]) for i in random_indices]))
        coverage.update(m["id"] for m in recommendations)
    return {"genre_precision_at_k": float(np.mean(precision)),
            "random_precision_at_k": float(np.mean(baseline)),
            "catalog_coverage": len(coverage) / len(model.movies),
            "evaluated_queries": len(queries)}


def train(data, output="models", max_features=10000, seed=42, min_precision=0.0,
          min_lift=0.0, promote=False, tracking_uri=None):
    if not 0 <= min_precision <= 1 or max_features < 1 or not -1 <= min_lift <= 1:
        raise ValueError("Invalid training configuration")
    movies = read_movies(data)
    # Genres are evaluation labels only, avoiding direct label leakage.
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features,
                                 ngram_range=(1, 2), dtype=np.float32)
    matrix = vectorizer.fit_transform([m["overview"] for m in movies])
    if np.any(matrix.getnnz(axis=1) == 0):
        raise ValueError("Every movie must have at least one usable text feature")
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = Path(output) / version
    directory.mkdir(parents=True, exist_ok=False)
    catalog = [{k: m[k] for k in ("id", "title", "genres")} for m in movies]
    (directory / "movies.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    save_npz(directory / "features.npz", matrix)
    (directory / "vocabulary.json").write_text(json.dumps(vectorizer.get_feature_names_out().tolist()), encoding="utf-8")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    manifest = {"version": version, "source_sha256": digest(data), "git_revision": revision,
                "python": platform.python_version(), "sklearn": sklearn.__version__,
                "parameters": {"max_features": max_features, "seed": seed, "k": 5},
                "movie_count": len(movies), "feature_count": matrix.shape[1],
                "sha256": {name: digest(directory / name) for name in
                           ("movies.json", "features.npz", "vocabulary.json")}}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    metrics = evaluate(Recommender(directory), seed=seed)
    passed = (metrics["genre_precision_at_k"] >= min_precision and
              metrics["genre_precision_at_k"] - metrics["random_precision_at_k"] >= min_lift)
    report = {**metrics, "gate_passed": bool(passed), "min_precision": min_precision, "min_lift": min_lift}
    (directory / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if tracking_uri:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("movie-recommender")
        with mlflow.start_run(run_name=version):
            mlflow.log_params(manifest["parameters"])
            mlflow.set_tags({"source_sha256": manifest["source_sha256"], "git_revision": revision,
                             "gate_passed": str(passed)})
            mlflow.log_metrics(metrics)
            mlflow.log_artifacts(str(directory))
    if not passed:
        raise ValueError(f"Quality gate failed; candidate retained at {directory}")
    if promote:
        temporary = Path(output) / f"current-{uuid.uuid4().hex}.tmp"
        temporary.write_text(version, encoding="utf-8")
        os.replace(temporary, Path(output) / "current.txt")
    return directory, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="models")
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-precision", type=float, default=0.0)
    parser.add_argument("--min-lift", type=float, default=0.0)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--tracking-uri")
    directory, report = train(**vars(parser.parse_args()))
    print(json.dumps({"model_directory": str(directory), **report}, indent=2))
