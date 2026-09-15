import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Response
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from recommender.model import Recommender


def create_app(model_dir=None):
    registry = CollectorRegistry()
    requests = Counter("recommendation_requests_total", "Recommendation outcomes", ["outcome"], registry=registry)
    latency = Histogram("recommendation_latency_seconds", "Recommendation latency", registry=registry)

    @asynccontextmanager
    async def lifespan(app):
        root = Path(os.getenv("MODEL_ROOT", "models"))
        directory = model_dir or os.getenv("MODEL_DIR") or root / (root / "current.txt").read_text().strip()
        app.state.model = Recommender(directory)
        yield

    app = FastAPI(title="Movie Recommendation API", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        return {"status": "ready", "version": app.state.model.manifest["version"]}

    @app.get("/movies")
    def movies():
        return app.state.model.movies

    @app.get("/recommendations/{movie_id}")
    def recommendations(movie_id: int, k: int = Query(5, ge=1, le=20)):
        start = time.perf_counter()
        try:
            result = app.state.model.recommend(movie_id, k)
            requests.labels("success").inc()
            return {"model_version": app.state.model.manifest["version"], "recommendations": result}
        except KeyError:
            requests.labels("unknown_movie").inc()
            raise HTTPException(404, "Movie ID not found") from None
        finally:
            latency.observe(time.perf_counter() - start)

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
