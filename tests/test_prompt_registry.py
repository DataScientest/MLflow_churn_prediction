"""Prompt Registry: the calls expected by the course solution (chapter 09/10)."""
import mlflow
from mlflow import MlflowClient

PROMPT_NAME = "retention-assistant-prompt"
BASELINE = "Baseline template. Customer Input: {{input}}"
CANDIDATE = "Candidate template with few-shot. Customer Input: {{input}}"


def _register_both():
    v1 = mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=BASELINE,
        model_config={"model_name": "model-name", "temperature": 0.0},
        commit_message="v0.1: Baseline - Strict JSON contract, no few-shot.",
    )
    v2 = mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=CANDIDATE,
        model_config={"model_name": "model-name", "temperature": 0.0},
        commit_message="v0.2: Candidate - Added few-shot examples and strict grounding.",
    )
    return v1, v2


def test_register_two_versions():
    v1, v2 = _register_both()
    assert (v1.version, v2.version) == (1, 2)
    assert v2.commit_message.startswith("v0.2")
    assert v2.model_config == {"model_name": "model-name", "temperature": 0.0}


def test_load_by_integer_version():
    _register_both()
    p1 = mlflow.genai.load_prompt(PROMPT_NAME, version=1)
    p2 = mlflow.genai.load_prompt(PROMPT_NAME, version=2)
    assert (p1.version, p1.template) == (1, BASELINE)
    assert (p2.version, p2.template) == (2, CANDIDATE)
    assert p2.variables == {"input"}


def test_load_by_alias_uses_prompts_scheme():
    _register_both()
    mlflow.genai.set_prompt_alias(name=PROMPT_NAME, alias="production", version=2)
    mlflow.genai.set_prompt_alias(name=PROMPT_NAME, alias="challenger", version=1)

    prod = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@production")
    chall = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@challenger")
    assert (prod.version, prod.template) == (2, CANDIDATE)
    assert (chall.version, chall.template) == (1, BASELINE)


def test_models_scheme_does_not_load_prompts():
    """Documents that the `models:/<name>@<alias>` URI (old course solution) is rejected."""
    import pytest
    from mlflow.exceptions import MlflowException

    _register_both()
    mlflow.genai.set_prompt_alias(name=PROMPT_NAME, alias="production", version=2)
    with pytest.raises(MlflowException, match="Not a proper prompts:/ URI"):
        mlflow.genai.load_prompt(f"models:/{PROMPT_NAME}@production")


def test_set_prompt_tag():
    _register_both()
    mlflow.genai.set_prompt_tag(name=PROMPT_NAME, key="v2_status", value="SHIPPED")
    mlflow.genai.set_prompt_tag(name=PROMPT_NAME, key="latest_release_decision", value="v2_SHIPPED")
    tags = MlflowClient().get_prompt(PROMPT_NAME).tags
    assert tags["v2_status"] == "SHIPPED"
    assert tags["latest_release_decision"] == "v2_SHIPPED"
