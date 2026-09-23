"""Tracing: RetentionAgent.invoke must produce the spans shown in the course.

No network: the model server (requests.post) and ChromaDB (RetentionSearchIndex)
are faked, the LLM is a GenericFakeChatModel. The real @tool/@mlflow.trace
functions from tools.py are executed through `.func`.
"""
import json
from types import SimpleNamespace

import mlflow
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from conftest import REPO_ROOT

FAKE_JSON = json.dumps(
    {
        "customer_id": "7590-VHVEG",
        "risk": {"score": 0.82, "label": "High"},
        "offer": {
            "name": "Loyalty Discount",
            "value": "20% off",
            "eligibility_rule_id": "RULE_policy_loyalty_discount",
        },
        "justification": "36 months tenure > 24 months threshold.",
        "email_draft": "Dear customer, ...",
        "sources": ["RULE_policy_loyalty_discount"],
    }
)


class _FakeIndex:
    def search_policy(self, query, n_results=3):
        return [
            {
                "id": "policy_loyalty_discount",
                "category": "Loyalty",
                "benefit": "20% off",
                "condition": "tenure > 24 months",
            }
        ]


@pytest.fixture
def offline_tools(monkeypatch):
    import requests
    import search_index

    monkeypatch.chdir(REPO_ROOT)  # tools.py reads data/telco_churn.csv relative to cwd

    def fake_post(url, json=None, headers=None, timeout=None):
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"predictions": [0.82]})

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(search_index, "RetentionSearchIndex", _FakeIndex)


def test_agent_invoke_produces_expected_spans(offline_tools):
    from agent import RetentionAgent

    llm = GenericFakeChatModel(messages=iter([AIMessage(content=FAKE_JSON)]))
    agent = RetentionAgent(llm=llm, system_message="You are a retention assistant. Answer in JSON.")

    query = "I am a customer since 3 years (ID: 7590-VHVEG), am I eligible for a discount?"
    with mlflow.start_run(run_name="trace_verification_test"):
        response = agent.invoke({"input": query})

    assert json.loads(response["output"])["customer_id"] == "7590-VHVEG"

    # MLflow 3.16 logs traces asynchronously: flush=True waits for the pending export
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id(), flush=True)
    assert trace is not None
    span_names = {s.name for s in trace.data.spans}
    assert {"get_churn_risk", "retrieve_retention_rules", "generate"} <= span_names

    spans = {s.name: s for s in trace.data.spans}
    # The root span comes from @mlflow.trace on RetentionAgent.invoke
    root = next(s for s in trace.data.spans if s.parent_id is None)
    assert root.name == "invoke"
    # The real tools ran (fakes only replaced the network layer)
    churn_outputs = [s.outputs for s in trace.data.spans if s.name == "get_churn_risk" and s.outputs]
    assert any("0.82" in str(o) for o in churn_outputs)
    assert "generate" in spans
