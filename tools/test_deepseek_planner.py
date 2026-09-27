#!/usr/bin/env python3
"""Offline contract tests for the optional DeepSeek planner."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from inference import InferenceInterface, _GenericPolicy  # noqa: E402
from inference.deepseek_planner import DeepSeekPlanner  # noqa: E402


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def main():
    previous = os.environ.get("DEEPSEEK_API_KEY")
    previous_enabled = os.environ.get("DEEPSEEK_ENABLED")
    previous_model = os.environ.get("EDA_LLM_MODEL")
    os.environ["DEEPSEEK_API_KEY"] = "test-only-not-a-secret"
    os.environ["DEEPSEEK_ENABLED"] = "1"
    os.environ.pop("EDA_LLM_MODEL", None)
    observed = {}

    def opener(request, timeout):
        observed["url"] = request.full_url
        observed["timeout"] = timeout
        observed["authorization"] = request.headers.get("Authorization")
        observed["request"] = json.loads(request.data.decode("utf-8"))
        content = json.dumps({
            "family_hint": "generic", "action_dim": 3,
            "macro_order": [3, 0, 2, 1],
            "program": [
                {"action": [0, 1, 2], "cycles": 2, "purpose": "reset"},
                {"action": [3, 4, 5], "cycles": 1, "purpose": "boundary"},
                {"action": [1, 2], "cycles": 1, "purpose": "bad dimension"},
            ], "summary": "test plan",
        })
        return FakeResponse({
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50,
                      "total_tokens": 150},
        })

    planner = DeepSeekPlanner(opener=opener)
    plan, status = planner.plan("action = [a,b,c]", "covergroup cg; endgroup")
    assert status["status"] == "ok" and status["usage"]["total_tokens"] == 150
    assert plan["family_hint"] == "generic" and plan["action_dim"] == 3
    assert len(plan["program"]) == 2  # malformed action was rejected
    assert observed["url"].endswith("/chat/completions")
    # Assert the *env chain* rather than a pinned name: the default model is a
    # measurement-driven choice and will change, and a hardcoded string turns
    # that change into a red test instead of a visible diff.
    assert observed["request"]["model"] == planner.model
    assert planner.model, "a model id must always be resolved"
    os.environ["EDA_LLM_MODEL"] = "model-under-test"
    assert DeepSeekPlanner(opener=opener).model == "model-under-test"
    os.environ.pop("EDA_LLM_MODEL", None)   # restored for real at the end
    assert observed["request"]["response_format"] == {"type": "json_object"}
    assert observed["authorization"].startswith("Bearer ")

    policy = _GenericPolicy(3, planned_program=plan["program"])
    outputs = [policy.predict(np.zeros(1), step, 10).tolist()
               for step in range(3)]
    # Unknown scalar fields are conservatively clamped to their inferred
    # one-bit range before an LLM-generated program reaches the DUT.
    assert outputs == [[0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

    if previous is None:
        os.environ.pop("DEEPSEEK_API_KEY", None)
    else:
        os.environ["DEEPSEEK_API_KEY"] = previous
    if previous_enabled is None:
        os.environ.pop("DEEPSEEK_ENABLED", None)
    else:
        os.environ["DEEPSEEK_ENABLED"] = previous_enabled
    if previous_model is None:
        os.environ.pop("EDA_LLM_MODEL", None)
    else:
        os.environ["EDA_LLM_MODEL"] = previous_model
    base = ROOT / "public_duts/spi_xfer_public/spi_xfer_public/dut"
    agent = InferenceInterface(str(base / "dut_spec.md"),
                               str(base / "covergroup.svh"))
    assert agent.llm_status["status"] == "disabled"
    action = agent.predict(np.zeros(86, dtype=np.float32), 0, 100)
    assert action.shape == (12,)
    print(json.dumps({
        "mock_api": status, "validated_program_actions": len(plan["program"]),
        "default_model": planner.model,
        "fallback_status": agent.llm_status["status"],
        "fallback_action_dim": int(action.size),
    }, indent=2))


if __name__ == "__main__":
    main()
