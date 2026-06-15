"""MLflow integration for retrieval experiment tracking.

Thin wrapper around mlflow that:
  - Is a graceful no-op when mlflow is not installed
  - Auto-detects Databricks Community Edition (tracking URI is set by the runtime)
  - Works locally by logging to ./mlruns (view with: mlflow ui)

Usage:
    from app.mlflow_tracker import log_eval_run

    log_eval_run(
        run_name="hybrid_eval",
        params={"embedding_model": "e5-small-v2", "top_k": 5},
        metrics={"hit_at_5": 0.82, "mrr": 0.71, "latency_p50_ms": 45.2},
    )
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

_DEFAULT_EXPERIMENT = "agentic-rag-retrieval-eval"


def _mlflow():
    try:
        import mlflow  # noqa: PLC0415
        return mlflow
    except ImportError:
        return None


def is_available() -> bool:
    return _mlflow() is not None


def is_databricks() -> bool:
    """True when running inside a Databricks cluster."""
    mlflow = _mlflow()
    if mlflow is None:
        return False
    uri = mlflow.get_tracking_uri() or ""
    return uri.startswith("databricks") or "azuredatabricks" in uri


@contextmanager
def start_run(
    run_name: str,
    experiment: str = _DEFAULT_EXPERIMENT,
) -> Iterator[Any]:
    """Context manager: yields an mlflow active_run (or a no-op sentinel)."""
    mlflow = _mlflow()
    if mlflow is None:
        yield None
        return

    # Databricks Free Edition Serverless: set_experiment() can fail with a
    # gRPC INTERNAL error (spark.mlflow.modelRegistryUri not available).
    # Skip it — Databricks auto-attaches the run to the notebook experiment.
    try:
        mlflow.set_experiment(experiment)
    except Exception:
        pass
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_eval_run(
    run_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    artifact_path: Optional[str] = None,
    experiment: str = _DEFAULT_EXPERIMENT,
) -> Optional[str]:
    """Log a retrieval evaluation run. Returns the MLflow run_id or None."""
    mlflow = _mlflow()
    if mlflow is None:
        return None

    try:
        mlflow.set_experiment(experiment)
    except Exception:
        pass
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if artifact_path:
            mlflow.log_artifact(artifact_path, artifact_path="results")
        return run.info.run_id
