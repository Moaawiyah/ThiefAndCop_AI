"""Prompt templates for the NL layer (assignment §5.1).

The agents exchange *free natural language* — never raw numeric coordinates. The
LLM is used for two jobs each turn:

1. ``message_prompt`` — generate a short NL message describing the agent's
   intentions and local observations (bluffing/deception is allowed for the
   thief, §5.1).
2. ``belief_prompt`` — read the opponent's NL message and *guess* a rough
   direction/region of the opponent, which the orchestrator folds into the
   Q-policy belief.

Both prompts are deliberately small (short dialogues -> low token use, §7.1).
The orchestrator owns these prompts and the LLM call — the MCP servers never run
the LLM (critical architecture rule, §5.2).
"""

from __future__ import annotations

from core.observation import Observation

COP_SYSTEM = (
    "You are COP, an autonomous pursuit agent on a grid. Your goal is to corner "
    "and capture the THIEF. You communicate ONLY in short, free natural language "
    "(one or two sentences). Never state exact numeric coordinates. Describe your "
    "intentions and what you sense nearby. Be tactical and confident."
)

THIEF_SYSTEM = (
    "You are THIEF, an autonomous evasion agent on a grid. Your goal is to survive "
    "without being caught by the COP. You communicate ONLY in short, free natural "
    "language (one or two sentences). Never state exact numeric coordinates. You "
    "may bluff or mislead about where you are heading. Be evasive and cunning."
)


def _describe_observation(obs: Observation) -> str:
    """Turn a partial observation into NL hints (no raw coordinates leaked)."""
    rows, cols = obs.grid_size
    r, c = obs.self_pos
    # Qualitative region instead of exact numbers.
    vert = "north" if r < rows / 3 else ("south" if r > 2 * rows / 3 else "center")
    horiz = "west" if c < cols / 3 else ("east" if c > 2 * cols / 3 else "center")
    region = f"{vert}-{horiz}".replace("center-center", "the middle")

    parts = [f"I am somewhere around the {region} of the board."]
    if obs.opponent_visible and obs.opponent_pos is not None:
        orr, occ = obs.opponent_pos
        dv = "north" if orr < r else ("south" if orr > r else "level")
        dh = "west" if occ < c else ("east" if occ > c else "level")
        parts.append(f"I can see my target to my {dv}/{dh}.")
    else:
        parts.append("My target is out of sight right now.")
    if obs.visible_barriers:
        parts.append("There are barriers near me.")
    parts.append(f"This is move {obs.move_number} of {obs.max_moves}.")
    return " ".join(parts)


def message_prompt(obs: Observation, opponent_message: str) -> str:
    """Build the user prompt asking the agent to produce an NL message."""
    role = obs.agent.upper()
    incoming = opponent_message.strip() or "(no message yet)"
    return (
        f"Situation for {role}: {_describe_observation(obs)}\n"
        f"The other agent just said: \"{incoming}\"\n"
        "Reply with ONE short natural-language message (max 2 sentences) for the "
        "other agent. Do not include coordinates or numbers."
    )


def belief_prompt(obs: Observation, opponent_message: str) -> str:
    """Ask the LLM to infer a coarse direction of the opponent from their text."""
    return (
        f"You are {obs.agent.upper()}. The opponent said: "
        f"\"{opponent_message.strip()}\"\n"
        "Based only on this message, guess the opponent's rough direction relative "
        "to the board. Answer with EXACTLY one word from: north, south, east, "
        "west, northeast, northwest, southeast, southwest, center, unknown."
    )


def system_prompt(agent: str) -> str:
    return COP_SYSTEM if agent == "cop" else THIEF_SYSTEM
