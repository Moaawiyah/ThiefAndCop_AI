"""Tests for core/config.py loading of the qlearning epsilon schedules."""

import yaml

from core.config import load_config


BASE_YAML = {
    "grid_size": [4, 4],
    "max_moves": 8,
    "num_games": 1,
    "max_barriers": 2,
    "vision_radius": 1,
    "qlearning": {},
}


def _write_config(tmp_path, ql_overrides):
    data = dict(BASE_YAML)
    data["qlearning"] = ql_overrides
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def test_thief_epsilon_decay_defaults_to_epsilon_decay_when_unset(tmp_path):
    path = _write_config(tmp_path, {"epsilon_decay": 0.987})
    cfg = load_config(path)
    assert cfg.qlearning.epsilon_decay == 0.987
    assert cfg.qlearning.thief_epsilon_decay == 0.987


def test_thief_epsilon_decay_overrides_independently(tmp_path):
    path = _write_config(tmp_path, {"epsilon_decay": 0.99, "thief_epsilon_decay": 0.999})
    cfg = load_config(path)
    assert cfg.qlearning.epsilon_decay == 0.99
    assert cfg.qlearning.thief_epsilon_decay == 0.999
