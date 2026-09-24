"""Release gate: decide_release() compares the latest eval runs of two prompt versions.

evaluate_agent.py is NOT imported: it contains the student hole
`result = # Insert your code here` and does not parse.
"""
import mlflow
import pytest

import release_decision

ALL_GOOD = {
    "discount_policy_compliance/mean": 1.0,
    "json_format_ok/mean": 1.0,
    "business_relevance/mean": 0.9,
    "risk_grounding/mean": 0.8,
    "llm_judge_business/mean": 0.8,
}


def _log_eval_run(version: int, metrics: dict):
    with mlflow.start_run(run_name=f"evaluation_v{version}"):
        mlflow.set_tags({"prompt_version": str(version), "env": "evaluation"})
        mlflow.log_metrics(metrics)


def test_no_ship_when_candidate_breaks_policy_gate():
    _log_eval_run(1, ALL_GOOD)
    _log_eval_run(2, {**ALL_GOOD, "discount_policy_compliance/mean": 0.5})
    assert release_decision.decide_release(1, 2) == "NO_SHIP"


def test_no_ship_on_regression_vs_baseline():
    _log_eval_run(1, ALL_GOOD)
    _log_eval_run(2, {**ALL_GOOD, "business_relevance/mean": 0.82})  # passes 0.80 gate, -0.08 vs baseline
    assert release_decision.decide_release(1, 2) == "NO_SHIP"


def test_ship_when_all_gates_pass():
    _log_eval_run(1, ALL_GOOD)
    _log_eval_run(2, {**ALL_GOOD, "json_format_ok/mean": 1.0, "llm_judge_business/mean": 0.9})
    assert release_decision.decide_release(1, 2) == "SHIP"


def test_no_ship_when_runs_missing():
    _log_eval_run(1, ALL_GOOD)
    assert release_decision.decide_release(1, 2) == "NO_SHIP"


@pytest.mark.parametrize("metric", sorted(release_decision.ABSOLUTE_GATES))
def test_each_absolute_gate_blocks(metric):
    _log_eval_run(1, ALL_GOOD)
    _log_eval_run(2, {**ALL_GOOD, metric: release_decision.ABSOLUTE_GATES[metric] - 0.1})
    assert release_decision.decide_release(1, 2) == "NO_SHIP"
