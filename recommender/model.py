import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Recommender:
    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        for name, expected in self.manifest["sha256"].items():
            if digest(directory / name) != expected:
                raise ValueError(f"Artifact integrity check failed: {name}")
        self.movies = json.loads((directory / "movies.json").read_text())
        self.matrix = load_npz(directory / "features.npz")
        if self.matrix.shape[0] != len(self.movies):
            raise ValueError("Movie and feature counts differ")
        self.indices = {m["id"]: i for i, m in enumerate(self.movies)}

    def recommend(self, movie_id, k=5):
        if not 1 <= k <= 20:
            raise ValueError("k must be between 1 and 20")
        index = self.indices[movie_id]
        scores = (self.matrix @ self.matrix[index].T).toarray().ravel()
        scores[index] = -np.inf
        order = np.argsort(-scores, kind="stable")[:min(k, len(self.movies) - 1)]
        return [{**self.movies[i], "score": float(scores[i])} for i in order]
