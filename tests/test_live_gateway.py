"""Optional live checks against the OpenAI-compatible gateway (disabled by default).

Run with: OPENAI_API_KEY=... OPENAI_BASE_URL=https://ai-gateway.liora.tech/v1 uv run pytest -m live -q
"""
import os

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_BASE_URL")),
        reason="OPENAI_API_KEY / OPENAI_BASE_URL not set",
    ),
]


def test_chat_openai_default_model():
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)
    out = llm.invoke("Reply with the single word: pong")
    assert "pong" in out.content.lower()


def test_openai_sdk_judge_path():
    """Same call shape as llm_judge_score() in evaluate_agent.py (openai SDK)."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"])
    resp = client.chat.completions.create(
        model=os.getenv("JUDGE_LLM_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": 'Return ONLY JSON: {"score": 1.0, "reason": "ok"}'}],
        temperature=0,
    )
    assert "score" in (resp.choices[0].message.content or "")
