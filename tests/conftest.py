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

# Snapshot of the environment taken before any test module is imported.
# Several scripts call ``load_dotenv(override=True)`` at import time
# (release_decision.py, search_index.py, test_agent_trace.py): during collection
# they would inject the learner's .env (e.g. LLM_MODEL=gemma3:4b) into os.environ.
_ORIGINAL_ENV = dict(os.environ)

EXPERIMENT_NAME = "Churn_Prediction_Basic"


def _restore_env():
    # PYTEST_CURRENT_TEST is managed by pytest itself during the test: keep it.
    for key in list(os.environ):
        if key not in _ORIGINAL_ENV and not key.startswith("PYTEST_"):
            del os.environ[key]
    for key, value in _ORIGINAL_ENV.items():
        if os.environ.get(key) != value:
            os.environ[key] = value


@pytest.fixture(autouse=True)
def isolated_env():
    """Run every test with the environment of the pytest process, not the .env file."""
    _restore_env()
    yield
    _restore_env()


@pytest.fixture(autouse=True)
def mlflow_sqlite(isolated_env, tmp_path, monkeypatch):
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
