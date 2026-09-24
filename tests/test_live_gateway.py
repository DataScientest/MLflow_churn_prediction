"""Optional live checks against the OpenAI-compatible gateway (disabled by default).

Run with: OPENAI_API_KEY=... OPENAI_BASE_URL=https://ai-gateway.liora.tech/v1 uv run pytest -m live -q

The model is set by LIVE_LLM_MODEL (default gpt-4o-mini), not by LLM_MODEL /
JUDGE_LLM_MODEL: those come from the course .env (LLM_MODEL=gemma3:4b is an
Ollama model that the gateway rejects).
"""
import os

import pytest

LIVE_LLM_MODEL = os.getenv("LIVE_LLM_MODEL", "gpt-4o-mini")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_BASE_URL")),
        reason="OPENAI_API_KEY / OPENAI_BASE_URL not set",
    ),
]


def test_chat_openai_default_model():
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=LIVE_LLM_MODEL, temperature=0)
    out = llm.invoke("Reply with the single word: pong")
    assert "pong" in out.content.lower()


def test_openai_sdk_judge_path():
    """Same call shape as llm_judge_score() in evaluate_agent.py (openai SDK)."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"])
    resp = client.chat.completions.create(
        model=LIVE_LLM_MODEL,
        messages=[{"role": "user", "content": 'Return ONLY JSON: {"score": 1.0, "reason": "ok"}'}],
        temperature=0,
    )
    assert "score" in (resp.choices[0].message.content or "")
