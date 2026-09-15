"""Compare incoming catalog genre proportions with the deployed reference."""
import argparse
import json
from collections import Counter
from pathlib import Path

from recommender.model import Recommender
from recommender.train import read_movies


def distribution(movies):
    counts = Counter(g for movie in movies for g in movie["genres"])
    total = sum(counts.values())
    if not total:
        raise ValueError("Drift comparison requires genre labels")
    return {genre: count / total for genre, count in counts.items()}


def compare(model_dir, data, threshold=0.2):
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between zero and one")
    model = Recommender(model_dir)
    incoming = read_movies(data)
    reference, current = distribution(model.movies), distribution(incoming)
    distance = sum(abs(reference.get(g, 0) - current.get(g, 0))
                   for g in reference.keys() | current.keys()) / 2
    return {"model_version": model.manifest["version"], "genre_total_variation": distance,
            "threshold": threshold, "drift_detected": distance > threshold,
            "reference_movies": len(model.movies), "incoming_movies": len(incoming)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--report", default="reports/drift.json")
    args = parser.parse_args()
    result = compare(args.model_dir, args.data, args.threshold)
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(2 if result["drift_detected"] else 0)
