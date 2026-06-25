"""Tests for llm/glm_client.py: disabled/template fallback (real) + mocked API."""

from unittest.mock import MagicMock

from core.config import LLMConfig
from core.observation import Observation
from llm.glm_client import LLMClient


def make_obs(agent="cop", opponent_visible=True, opponent_pos=(4, 4)):
    return Observation(
        agent=agent, self_pos=(0, 0), opponent_visible=opponent_visible,
        opponent_pos=opponent_pos, visible_barriers=[], vision_radius=2,
        grid_size=(5, 5), move_number=1, max_moves=25, barriers_left=2,
    )


def fake_response(content, prompt_tokens=10, completion_tokens=5):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    resp.usage = MagicMock(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return resp


def test_disabled_llm_is_unavailable_and_uses_template():
    client = LLMClient(LLMConfig(enabled=False))
    assert client.available is False
    obs = make_obs(agent="cop", opponent_visible=True)
    msg = client.generate_message(obs, "hi")
    assert msg == "I have eyes on you and I'm closing the distance fast."


def test_enabled_without_api_key_falls_back_to_template(monkeypatch):
    monkeypatch.delenv("GLM_API_KEY", raising=False)
    client = LLMClient(LLMConfig(enabled=True, api_key=""))
    assert client.available is False
    obs = make_obs(agent="thief", opponent_visible=False, opponent_pos=None)
    msg = client.generate_message(obs, "")
    assert msg == ("All quiet here. I'm staying light on my feet and keeping "
                    "my options open.")


def test_infer_direction_falls_back_to_keyword_scan_when_unavailable():
    client = LLMClient(LLMConfig(enabled=False))
    obs = make_obs()
    direction = client.infer_direction(obs, "I'm heading north fast")
    assert direction == "north"
    assert client.infer_direction(obs, "") == "unknown"


def test_chat_with_mocked_openai_client_accumulates_usage():
    client = LLMClient(LLMConfig(enabled=False))
    client.available = True
    client._client = MagicMock()
    client._client.chat.completions.create.return_value = fake_response("Closing in fast.")

    obs = make_obs()
    msg = client.generate_message(obs, "where are you")
    assert "Closing in fast" in msg
    assert client.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    client._client.chat.completions.create.return_value = fake_response("Still here.")
    client.generate_message(obs, "still there?")
    assert client.usage == {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}


def test_infer_direction_with_mocked_client_returns_valid_word():
    client = LLMClient(LLMConfig(enabled=False))
    client.available = True
    client._client = MagicMock()
    client._client.chat.completions.create.return_value = fake_response("northeast")

    obs = make_obs()
    direction = client.infer_direction(obs, "I'm near the corner")
    assert direction == "northeast"


def test_infer_direction_with_invalid_word_falls_back_to_keyword_scan():
    client = LLMClient(LLMConfig(enabled=False))
    client.available = True
    client._client = MagicMock()
    client._client.chat.completions.create.return_value = fake_response("gibberish")

    obs = make_obs()
    direction = client.infer_direction(obs, "I am going south")
    assert direction == "south"


def test_chat_exception_path_marks_unavailable():
    client = LLMClient(LLMConfig(enabled=False))
    client.available = True
    client._client = MagicMock()
    client._client.chat.completions.create.side_effect = RuntimeError("boom")

    obs = make_obs()
    msg = client.generate_message(obs, "hi")
    assert client.available is False
    # Falls back to template since _chat returned None.
    assert msg == "I have eyes on you and I'm closing the distance fast."


def test_resolve_key_prefers_config_over_env(monkeypatch):
    monkeypatch.setenv("GLM_API_KEY", "env-key")
    client = LLMClient(LLMConfig(enabled=False, api_key="config-key"))
    assert client._resolve_key() == "config-key"


def test_resolve_key_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("GLM_API_KEY", "env-key")
    client = LLMClient(LLMConfig(enabled=False, api_key=""))
    assert client._resolve_key() == "env-key"


def test_try_connect_with_key_sets_available(monkeypatch):
    monkeypatch.setenv("GLM_API_KEY", "some-key")
    client = LLMClient(LLMConfig(enabled=True, api_key=""))
    assert client.available is True
    assert client._client is not None
