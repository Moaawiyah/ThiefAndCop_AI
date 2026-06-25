"""Tests for llm/prompts.py: prompt templates, no real coordinates leaked."""

from core.observation import Observation
from llm import prompts


def make_obs(agent="cop", self_pos=(0, 0), opponent_visible=True,
             opponent_pos=(4, 4), visible_barriers=None):
    return Observation(
        agent=agent, self_pos=self_pos, opponent_visible=opponent_visible,
        opponent_pos=opponent_pos, visible_barriers=visible_barriers or [],
        vision_radius=2, grid_size=(5, 5), move_number=3, max_moves=25,
        barriers_left=2,
    )


def test_system_prompt_selects_by_agent():
    assert prompts.system_prompt("cop") == prompts.COP_SYSTEM
    assert prompts.system_prompt("thief") == prompts.THIEF_SYSTEM


def test_message_prompt_includes_incoming_message_and_role():
    obs = make_obs(agent="thief")
    out = prompts.message_prompt(obs, "I see you")
    assert "THIEF" in out
    assert "I see you" in out
    assert "coordinates" in out


def test_message_prompt_handles_empty_incoming_message():
    obs = make_obs()
    out = prompts.message_prompt(obs, "")
    assert "(no message yet)" in out


def test_belief_prompt_format():
    obs = make_obs(agent="cop")
    out = prompts.belief_prompt(obs, "heading south fast")
    assert "COP" in out
    assert "heading south fast" in out
    assert "unknown" in out


def test_describe_observation_no_opponent_visible():
    obs = make_obs(opponent_visible=False, opponent_pos=None)
    desc = prompts._describe_observation(obs)
    assert "out of sight" in desc


def test_describe_observation_with_barriers_and_region():
    obs = make_obs(self_pos=(0, 0), opponent_pos=(4, 4), visible_barriers=[(1, 1)])
    desc = prompts._describe_observation(obs)
    assert "barriers" in desc
    assert "move 3 of 25" in desc


def test_describe_observation_middle_region():
    obs = make_obs(self_pos=(2, 2), opponent_visible=False, opponent_pos=None)
    desc = prompts._describe_observation(obs)
    assert "the middle" in desc
