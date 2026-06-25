"""Tests for the plain mcp_servers/tools.py functions against a real GameSession."""

import pytest

from core.config import load_config
from core.engine import GameEngine
from mcp_servers import tools as T
from mcp_servers.session import AuthError, GameSession


def make_session():
    cfg = load_config()
    eng = GameEngine(cfg)
    session = GameSession(config=cfg, agent="cop", engine=eng)
    session.start_sub_game()
    return session


TOKEN = load_config().mcp.auth_token


def test_send_and_read_message_round_trip():
    s = make_session()
    res = T.send_message(s, TOKEN, sender="cop", text="closing in")
    assert res == {"ok": True, "sender": "cop", "text": "closing in"}
    out = T.read_message(s, TOKEN, reader="thief")
    assert out["message"] == "closing in"


def test_auth_rejected_with_wrong_token():
    s = make_session()
    with pytest.raises(AuthError):
        T.send_message(s, "bad-token", sender="cop", text="hi")


def test_get_observation_returns_dict_with_expected_keys():
    s = make_session()
    obs = T.get_observation(s, TOKEN, agent="cop")
    assert obs["agent"] == "cop"
    assert "self_pos" in obs and "opponent_visible" in obs


def test_verify_position_confirms_and_denies():
    s = make_session()
    actual = s.engine.state.cop
    ok = T.verify_position(s, TOKEN, agent="cop", claimed_pos=actual)
    assert ok["confirmed"] is True
    wrong = (actual[0], actual[1])
    bad_claim = ((actual[0] + 1) % s.config.rows, actual[1])
    bad = T.verify_position(s, TOKEN, agent="cop", claimed_pos=bad_claim)
    assert bad["confirmed"] is False
    assert wrong == actual  # sanity: didn't mutate state


def test_submit_move_detects_capture():
    s = make_session()
    s.engine.state.cop = (2, 2)
    s.engine.state.thief = (2, 3)
    res = T.submit_move(s, TOKEN, agent="thief", action="W")
    assert res["captured"] is True
    assert s.done is True
    assert s.winner == "cop"


def test_submit_move_without_capture():
    s = make_session()
    s.engine.state.cop = (0, 0)
    s.engine.state.thief = (4, 4)
    res = T.submit_move(s, TOKEN, agent="cop", action="E")
    assert res["captured"] is False
    assert s.done is False


def test_place_barrier_only_for_cop():
    s = make_session()
    res = T.place_barrier(s, TOKEN, agent="thief")
    assert res == {"ok": False, "error": "only the cop may place barriers"}


def test_place_barrier_limit_and_placed_flag():
    s = make_session()
    s.config.max_barriers = 1
    cop_pos = s.engine.state.cop
    res1 = T.place_barrier(s, TOKEN, agent="cop")
    assert res1["placed"] is True
    assert res1["barriers_left"] == 0
    # Second attempt: limit reached (barriers_placed already 1).
    s.engine.state.cop = cop_pos
    res2 = T.place_barrier(s, TOKEN, agent="cop")
    assert res2["barriers_placed"] == 1


def test_game_status_fields():
    s = make_session()
    status = T.game_status(s, TOKEN)
    for key in ("ok", "sub_game_index", "move_number", "max_moves", "done",
                "winner", "cop_total", "thief_total", "cop_pos", "thief_pos"):
        assert key in status
    assert status["done"] is False


def test_start_sub_game_with_explicit_positions():
    s = make_session()
    res = T.start_sub_game(s, TOKEN, cop_row=1, cop_col=1, thief_row=3, thief_col=3)
    assert res["cop_pos"] == [1, 1]
    assert res["thief_pos"] == [3, 3]
    assert s.engine.state.cop == (1, 1)
    assert s.engine.state.thief == (3, 3)


def test_start_sub_game_without_explicit_positions_samples():
    s = make_session()
    res = T.start_sub_game(s, TOKEN)
    assert res["sub_game_index"] == 2
    assert isinstance(res["cop_pos"], list)


def test_advance_move_counter_timeout_is_thief_win():
    s = make_session()
    cfg = s.config
    for _ in range(cfg.max_moves):
        out = T.advance_move_counter(s, TOKEN)
    assert out["done"] is True
    assert out["winner"] == "thief"
    assert s.winner == "thief"


def test_advance_move_counter_no_timeout_yet():
    s = make_session()
    out = T.advance_move_counter(s, TOKEN)
    assert out["move_number"] == 1
    assert out["done"] is False
    assert out["winner"] is None
