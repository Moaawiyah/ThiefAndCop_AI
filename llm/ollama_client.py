"""Ollama LLM client for the NL layer (Approach 2, assignment §7.2).

Talks to a local Ollama daemon (``http://127.0.0.1:11434`` by default) or a
remote one exposed via an authenticated ngrok tunnel (``auth_header`` set).

CRITICAL: the whole NL layer degrades gracefully. If ``ollama.enabled`` is False,
the daemon is unreachable, or the model is not pulled, every method returns a
deterministic *template* fallback so the game still runs end-to-end on the
Q-Learning / heuristic policy alone. Missing Ollama must never break the game.
"""

from __future__ import annotations

from typing import Optional

from core.config import OllamaConfig
from core.observation import Observation
from . import prompts

try:  # ollama python client is optional at runtime
    import ollama as _ollama
    _OLLAMA_IMPORT_OK = True
except Exception:  # pragma: no cover - import guard
    _ollama = None
    _OLLAMA_IMPORT_OK = False


# Coarse direction words used for template fallbacks and belief parsing.
_DIRECTIONS = {
    "north", "south", "east", "west",
    "northeast", "northwest", "southeast", "southwest",
    "center", "unknown",
}


class OllamaClient:
    """Thin, fault-tolerant wrapper around the Ollama chat API."""

    def __init__(self, config: OllamaConfig):
        self.config = config
        self.available = False
        self._client = None
        if config.enabled and _OLLAMA_IMPORT_OK:
            self._try_connect()

    # ----- connection ------------------------------------------------------
    def _try_connect(self) -> None:
        try:
            headers = {}
            if self.config.auth_header:
                headers["Authorization"] = self.config.auth_header
            self._client = _ollama.Client(host=self.config.base_url, headers=headers or None)
            # Probe: list models. If the daemon is down this raises.
            self._client.list()
            self.available = True
        except Exception:
            self.available = False
            self._client = None

    # ----- core chat -------------------------------------------------------
    def _chat(self, system: str, user: str, max_tokens: int = 80) -> Optional[str]:
        if not self.available or self._client is None:
            return None
        try:
            resp = self._client.chat(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options={"num_predict": max_tokens, "temperature": 0.8},
            )
            return (resp.get("message", {}) or {}).get("content", "").strip()
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
            max_tokens=8,
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
