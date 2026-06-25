"""GLM LLM client for the NL layer (Approach 1, assignment §7.1).

Talks to the Zhipu GLM public-cloud API (OpenAI-compatible) at
``https://api.z.ai/api/paas/v4``. The API key is read from
``config.llm.api_key`` or, when that is empty, from the ``GLM_API_KEY``
environment variable (preferred — keep secrets out of the committed config).

CRITICAL: the whole NL layer degrades gracefully. If ``llm.enabled`` is False,
no API key is configured, the network is unreachable, or the ``openai`` package
is missing, every method returns a deterministic *template* fallback so the game
still runs end-to-end on the Q-Learning / heuristic policy alone. Missing GLM
must never break the game.
"""

from __future__ import annotations

import os
from typing import Optional

from core.config import LLMConfig
from core.observation import Observation
from . import prompts

try:  # openai python client is optional at runtime (GLM is OpenAI-compatible)
    from openai import OpenAI as _OpenAI
    _OPENAI_IMPORT_OK = True
except Exception:  # pragma: no cover - import guard
    _OpenAI = None
    _OPENAI_IMPORT_OK = False


# Environment variable holding the GLM API key (preferred over config).
GLM_ENV_KEY = "GLM_API_KEY"

# Coarse direction words used for template fallbacks and belief parsing.
_DIRECTIONS = {
    "north", "south", "east", "west",
    "northeast", "northwest", "southeast", "southwest",
    "center", "unknown",
}


class LLMClient:
    """Thin, fault-tolerant wrapper around the GLM chat-completions API."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.available = False
        self._client = None
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if config.enabled and _OPENAI_IMPORT_OK:
            self._try_connect()

    # ----- connection ------------------------------------------------------
    def _resolve_key(self) -> str:
        """Config key takes precedence; fall back to the GLM_API_KEY env var."""
        return self.config.api_key or os.environ.get(GLM_ENV_KEY, "")

    def _try_connect(self) -> None:
        key = self._resolve_key()
        if not key:
            self.available = False
            self._client = None
            return
        try:
            self._client = _OpenAI(api_key=key, base_url=self.config.base_url)
            self.available = True
        except Exception:
            self.available = False
            self._client = None

    # ----- core chat -------------------------------------------------------
    def _chat(self, system: str, user: str, max_tokens: int = 120) -> Optional[str]:
        if not self.available or self._client is None:
            return None
        try:
            resp = self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.8,
                extra_body={"thinking": {"type": "disabled"}},
            )
            if resp.usage:
                self.usage["prompt_tokens"] += resp.usage.prompt_tokens
                self.usage["completion_tokens"] += resp.usage.completion_tokens
                self.usage["total_tokens"] += resp.usage.total_tokens
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            # Any runtime failure -> mark unavailable so callers use fallbacks.
            self.available = False
            return None

    # ----- high-level helpers ---------------------------------------------
    def generate_message(self, obs: Observation, opponent_message: str = "") -> str:
        """Produce an NL message for the active agent (LLM or template)."""
        out = self._chat(
            prompts.system_prompt(obs.agent),
            prompts.message_prompt(obs, opponent_message),
        )
        if out:
            return _sanitise(out)
        return _template_message(obs)

    def infer_direction(self, obs: Observation, opponent_message: str) -> str:
        """Infer a coarse opponent direction word from their NL message."""
        out = self._chat(
            prompts.system_prompt(obs.agent),
            prompts.belief_prompt(obs, opponent_message),
            max_tokens=16,
        )
        if out:
            word = out.strip().lower().split()[0].strip(".,!?")
            if word in _DIRECTIONS:
                return word
        # Fallback: keyword scan of the message.
        return _keyword_direction(opponent_message)


# ---------------------------------------------------------------------------
# Deterministic fallbacks (used when the LLM is unavailable).
# ---------------------------------------------------------------------------
def _sanitise(text: str) -> str:
    """Trim to a couple of sentences and strip any leaked digits."""
    text = " ".join(text.split())
    # Remove standalone coordinate-like patterns; keep prose readable.
    cleaned = "".join(ch for ch in text if not ch.isdigit())
    cleaned = cleaned.replace("()", "").replace("[]", "").strip()
    # Keep it short.
    sentences = cleaned.replace("!", ".").replace("?", ".").split(".")
    short = ". ".join(s.strip() for s in sentences[:2] if s.strip())
    return (short + ".") if short else (text[:140])


def _template_message(obs: Observation) -> str:
    """A rule-based NL message used when no LLM is present (still free text)."""
    if obs.agent == "cop":
        if obs.opponent_visible:
            return "I have eyes on you and I'm closing the distance fast."
        return "I'm sweeping the board methodically; you can't hide for long."
    # thief
    if obs.opponent_visible:
        return "I spotted you, so I'm slipping away the other direction."
    return "All quiet here. I'm staying light on my feet and keeping my options open."


def _keyword_direction(message: str) -> str:
    """Very small heuristic to extract a direction from NL text."""
    if not message:
        return "unknown"
    m = message.lower()
    for d in ("northeast", "northwest", "southeast", "southwest",
              "north", "south", "east", "west", "center", "middle"):
        if d in m:
            return "center" if d == "middle" else d
    return "unknown"
