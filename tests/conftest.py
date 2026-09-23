"""Shared fixtures: isolated MLflow tracking store (SQLite in tmp_path), no network.

src/llm uses flat imports (``from tools import ...``), so both ``src`` and
``src/llm`` are added to sys.path, like when the scripts are run directly.
"""
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT / "src" / "llm", REPO_ROOT / "src", REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

EXPERIMENT_NAME = "Churn_Prediction_Basic"


@pytest.fixture(autouse=True)
def mlflow_sqlite(tmp_path, monkeypatch):
    """Point MLflow at a throw-away SQLite store and create the course experiment."""
    import mlflow

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    mlflow.set_tracking_uri(uri)
    exp_id = mlflow.create_experiment(
        EXPERIMENT_NAME, artifact_location=(tmp_path / "artifacts").as_uri()
    )
    mlflow.set_experiment(experiment_id=exp_id)
    yield uri
    while mlflow.active_run():
        mlflow.end_run()
