import csv
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from recommender.api import create_app
from recommender.model import Recommender
from recommender.train import read_movies, train
from scripts.make_demo_data import generate


@pytest.fixture
def data(tmp_path):
    path = tmp_path / "movies.csv"
    generate(path)
    return path


@pytest.fixture
def trained(data, tmp_path):
    return train(data, tmp_path / "models", promote=True)[0]


def test_reproducible_features_and_metrics(data, tmp_path):
    first, report1 = train(data, tmp_path / "a")
    second, report2 = train(data, tmp_path / "b")
    a, b = Recommender(first), Recommender(second)
    np.testing.assert_array_equal(a.matrix.toarray(), b.matrix.toarray())
    assert report1 == report2
    assert report1["genre_precision_at_k"] > report1["random_precision_at_k"]
    assert a.manifest["source_sha256"] == b.manifest["source_sha256"]


def test_recommendations_exclude_self_and_bound_k(trained):
    model = Recommender(trained)
    result = model.recommend(90000001, 20)
    assert len(result) == 17
    assert 90000001 not in [m["id"] for m in result]
    assert len({m["id"] for m in result}) == 17
    assert result[0]["score"] >= result[-1]["score"]


def test_failed_gate_preserves_current(data, trained):
    root = trained.parent
    before = (root / "current.txt").read_text()
    with pytest.raises(ValueError, match="Quality gate failed"):
        train(data, root, promote=True, min_lift=1.0)
    assert (root / "current.txt").read_text() == before


def test_corrupt_artifact_rejected(trained):
    (trained / "movies.json").write_text("[]")
    with pytest.raises(ValueError, match="integrity"):
        Recommender(trained)


@pytest.mark.parametrize("field,value", [
    ("id", "0"), ("title", ""), ("overview", ""), ("genres", "broken"),
    ("genres", "{}"), ("genres", "[42]"),
])
def test_invalid_rows(data, field, value):
    with data.open() as source:
        rows = list(csv.DictReader(source))
    rows[0][field] = value
    with data.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="row 2"):
        read_movies(data)


def test_missing_schema(tmp_path):
    source = tmp_path / "bad.csv"
    source.write_text("id,title\n1,Test\n")
    with pytest.raises(ValueError, match="columns"):
        read_movies(source)


def test_duplicate_id(data):
    content = data.read_text()
    data.write_text(content + content.splitlines()[1] + "\n")
    with pytest.raises(ValueError, match="unique"):
        read_movies(data)


def test_api_contract_and_monitoring(trained):
    with TestClient(create_app(trained)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").json()["version"] == trained.name
        assert len(client.get("/movies").json()) == 18
        response = client.get("/recommendations/90000001?k=3")
        assert response.status_code == 200
        assert len(response.json()["recommendations"]) == 3
        assert response.json()["model_version"] == trained.name
        assert client.get("/recommendations/1").status_code == 404
        assert client.get("/recommendations/90000001?k=0").status_code == 422
        assert client.get("/recommendations/90000001?k=21").status_code == 422
        metrics = client.get("/metrics").text
        assert 'recommendation_requests_total{outcome="success"} 1.0' in metrics
        assert 'recommendation_requests_total{outcome="unknown_movie"} 1.0' in metrics


def test_missing_model_fails_startup(tmp_path):
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(tmp_path / "missing")):
            pass


def test_drift_detects_catalog_shift(data, trained):
    from recommender.drift import compare
    assert compare(trained, data)["genre_total_variation"] == 0
    with data.open() as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        row["genres"] = json.dumps(["Documentary"])
    with data.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    report = compare(trained, data)
    assert report["genre_total_variation"] == 1
    assert report["drift_detected"]


def test_mlflow_tracking(data, tmp_path):
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient
    uri = (tmp_path / "mlruns").as_uri()
    directory, _ = train(data, tmp_path / "models", tracking_uri=uri)
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("movie-recommender")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].data.tags["mlflow.runName"] == directory.name
    assert "genre_precision_at_k" in runs[0].data.metrics
    assert any(a.path == "manifest.json" for a in client.list_artifacts(runs[0].info.run_id))
