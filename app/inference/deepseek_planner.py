"""Optional DeepSeek-V4 semantic planner with fail-closed local fallback.

Only Python's standard library is used so the official base image needs no
extra package.  One request is made during InferenceInterface construction;
the cycle-level predict path never waits for the network.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request


KNOWN_FAMILIES = {"dma", "spi_master", "spi_xfer", "generic"}


def _first_env(*names, default=None):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _json_object(text: str):
    """Recover an object from plain, fenced, prefixed or double JSON."""
    if not isinstance(text, str):
        return None
    candidate = text.strip().lstrip("\ufeff")
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate,
                           flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    for _ in range(2):
        try:
            value = json.loads(candidate)
        except ValueError:
            value = None
        if isinstance(value, dict):
            return value
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
            return value[0]
        if isinstance(value, str):
            candidate = value.strip()
            continue
        break
    start = candidate.find("{")
    if start >= 0:
        try:
            value, _ = json.JSONDecoder().raw_decode(candidate[start:])
            return value if isinstance(value, dict) else None
        except ValueError:
            pass
    return None


class DeepSeekPlanner:
    """OpenAI-compatible DeepSeek client that returns a validated plan."""

    def __init__(self, opener=None):
        self.api_key = _first_env(
            "DEEPSEEK_API_KEY", "DEEPSEEK_API_TOKEN",
            "LLM_API_KEY", "LLM_API_TOKEN", "OPENAI_API_KEY")
        self.base_url = _first_env(
            "DEEPSEEK_BASE_URL", "LLM_BASE_URL", "OPENAI_BASE_URL",
            default="https://api.deepseek.com")
        self.model = _first_env(
            "DEEPSEEK_MODEL", "LLM_MODEL", default="deepseek-v4-pro")
        self.timeout = max(1.0, float(os.environ.get("DEEPSEEK_TIMEOUT_S", "45")))
        self.max_tokens = min(8192, max(256, int(
            os.environ.get("DEEPSEEK_MAX_TOKENS", "4096"))))
        # LLM use is opt-in.  The default competition path is fully local;
        # setting DEEPSEEK_ENABLED=1 explicitly enables the one-shot planner.
        self.enabled = bool(self.api_key) and os.environ.get(
            "DEEPSEEK_ENABLED", "0").lower() in ("1", "true", "yes")
        self._open = opener or urllib.request.urlopen

    def _prompt(self, spec: str, covergroup: str) -> str:
        # The public budget is 500k tokens/DUT.  A bounded 80k-character input
        # plus a 4k-token answer stays far below it even for non-ASCII specs.
        spec = spec[:60000]
        covergroup = covergroup[:20000]
        return f"""Analyze this hardware-verification task and create a safe stimulus plan.
Return exactly one JSON object with this schema:
{{
  "family_hint": "dma|spi_master|spi_xfer|generic",
  "action_dim": positive integer,
  "macro_order": [0,1,2,3 in a useful permutation],
  "program": [
    {{"action": [finite numeric values with exactly action_dim entries],
      "cycles": integer 1..4096,
      "purpose": "short description"}}
  ],
  "summary": "short strategy summary"
}}

Use family_hint only for an exact semantic match; choose generic for any other DUT.
Derive action_dim and every action field from the DUT stimulus mapping. Generate
legal reset, configuration, boundary, cross, temporal and recovery sequences.
The program may contain at most 32 entries and should remain useful when repeated
coverage feedback is unavailable. Do not include Markdown or additional keys.

=== DUT SPEC ===
{spec}
=== COVERGROUP ===
{covergroup}
"""

    @staticmethod
    def _validate(raw):
        if not isinstance(raw, dict):
            return None
        family = str(raw.get("family_hint", "generic")).strip().lower()
        if family not in KNOWN_FAMILIES:
            family = "generic"
        try:
            dims = int(raw.get("action_dim", 0))
        except (TypeError, ValueError):
            dims = 0
        if not 1 <= dims <= 512:
            dims = 0
        order = []
        for value in raw.get("macro_order", []):
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= value < 4 and value not in order:
                order.append(value)
        order.extend(value for value in range(4) if value not in order)
        program = []
        for item in raw.get("program", [])[:32]:
            if not isinstance(item, dict):
                continue
            action = item.get("action")
            if not isinstance(action, list) or not dims or len(action) != dims:
                continue
            try:
                action = [float(value) for value in action]
                cycles = min(4096, max(1, int(item.get("cycles", 1))))
            except (TypeError, ValueError, OverflowError):
                continue
            if not all(math.isfinite(value) and abs(value) <= 2**32
                       for value in action):
                continue
            program.append({"action": action, "cycles": cycles})
        return {
            "family_hint": family,
            "action_dim": dims,
            "macro_order": order,
            "program": program,
            "summary": str(raw.get("summary", ""))[:500],
        }

    def plan(self, spec: str, covergroup: str):
        started = time.perf_counter()
        if not self.enabled:
            return None, {"status": "disabled", "model": self.model,
                          "latency_s": 0.0, "usage": {}}
        url = self.base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": (
                    "You are an expert coverage-driven RTL verification planner. "
                    "Obey the requested JSON schema and never invent action fields.")},
                {"role": "user", "content": self._prompt(spec, covergroup)},
            ],
            "max_tokens": self.max_tokens,
            # DeepSeek-V4 enables thinking by default.  Planning needs compact
            # schema-constrained output, so disabling CoT saves wall time and
            # tokens without putting reasoning text into the returned plan.
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        request = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.api_key})
        try:
            with self._open(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            choice = payload["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            plan = self._validate(_json_object(content))
            status = "ok" if plan else "invalid_response"
            usage = payload.get("usage", {})
        except urllib.error.HTTPError as exc:
            plan, status, usage, finish_reason = None, f"http_{exc.code}", {}, None
        except Exception as exc:  # network/timeout/JSON/schema: always degrade
            plan, status, usage, finish_reason = (
                None, "error_" + type(exc).__name__, {}, None)
        return plan, {
            "status": status, "model": self.model,
            "finish_reason": finish_reason,
            "latency_s": time.perf_counter() - started,
            "usage": {key: int(value) for key, value in usage.items()
                      if key in ("prompt_tokens", "completion_tokens", "total_tokens")
                      and isinstance(value, (int, float))},
        }
