# Product Requirements Document (PRD)

**Project:** Dual AI Agent — Cop & Thief Pursuit via MCP Servers
**Course:** Orchestration of AI Agents — Dr. Yoram Segal, University of Haifa
**Assignment:** HW6

---

## 1. Overview / Problem Statement

This project implements a **Decentralized Partially Observable Markov Decision Process
(Dec-POMDP)** pursuit game between two autonomous agents — a **Cop** and a **Thief** —
operating on a configurable grid world. Each agent is served by its own **FastMCP server**
and driven by a shared **MCP-client orchestrator** that mediates the turn loop.

The pedagogical value of the assignment is **not** the optimality of the pursuit strategy.
It is the **orchestration architecture** and the **free-form natural-language communication**
between agents: two independent MCP services, an LLM-mediated dialogue with no rigid position
protocol, a tabular reinforcement-learning decision core, and the surrounding production
concerns (configuration, testing, visualization, reporting, deployment, and security).

The formal model is the Dec-POMDP tuple
⟨ n, S, {Aᵢ}, P, R, {Ωᵢ}, O, γ ⟩, documented in full in `README.md`.

---

## 2. Goals & Non-Goals

### Goals
- Demonstrate a clean **multi-server MCP** topology (one server per agent) with a
  single orchestrating MCP client.
- Drive agent coordination through **natural-language, free-form dialogue** rather than a
  fixed coordinate-exchange protocol.
- Provide a **config-driven** grid engine (no hard-coded game parameters).
- Learn agent behavior with **tabular Q-learning** (Bellman update, epsilon-greedy).
- Deliver supporting infrastructure: GUI visualization, automated JSON email reporting,
  cloud deployment, and staged sanity checks.
- Meet all rubric quality gates (line limits, coverage, lint, dependency management, secrets).

### Non-Goals
- Achieving a provably optimal or competitive pursuit policy.
- Real-time / high-performance simulation at scale.
- A polished end-user product UI beyond what is needed to evidence behavior.
- The inter-group bonus competition (optional stretch; see §6).

---

## 3. Functional Requirements

Each assignment-required component and its implementing module(s). All are present and
verified working.

| # | Requirement | Implementing Module(s) | Status |
|---|-------------|------------------------|--------|
| 1 | Grid game engine: configurable 5×5, 8-directional movement (incl. diagonals), barriers (max 5, cop only), capture detection | `core/grid.py`, `core/engine.py`, `core/state.py`, `core/observation.py`, `core/policies.py` | ✅ Done |
| 2 | Two separate FastMCP servers (cop + thief) exposing tools | `mcp_servers/cop_server.py`, `thief_server.py`, `server_factory.py`, `tools.py`, `session.py` | ✅ Done |
| 3 | Natural-language free-form agent dialogue via LLM (no rigid position protocol) | `llm/glm_client.py`, `llm/prompts.py` (GLM cloud API; deterministic NL-template fallback without a key) | ✅ Done |
| 4 | MCP client / orchestrator driving the dialogue loop | `orchestrator.py`, `mcp_client/orchestrator_runner.py`, `bus.py`, `run_logging.py` | ✅ Done |
| 5 | Tabular Q-learning with Bellman update + epsilon-greedy | `agents/qlearning.py`, `train.py`, `train_utils.py`, `policy.py` | ✅ Done |
| 6 | Central config file, no hard-coding | `config.yaml`, `core/config.py`, `core/config_schema.py` | ✅ Done |
| 7 | Scoring: cop_win 20, thief_win 10, losses 5 per sub-game; 6 sub-games per game | `config.yaml` + `core/engine.py` | ✅ Done |
| 8 | GUI visualization (headless screenshots in artifacts) | `gui/visualizer.py`, `gui/render.py` | ✅ Done |
| 9 | Automated email report, JSON-only body, Gmail API; Internal + Inter-Group Bonus JSON schemas | `reporting/email_report.py`, `reporting/report_schema.py` | ✅ Done |
| 10 | Cloud deployment (Prefect Cloud) + security (token auth, tunneling) | `deploy/prefect_flow.py`, `deploy/ngrok.yaml` | ✅ Done |
| 11 | Staged sanity checks 2×2 → 3×3 → 4×4 → 5×5 | `scripts/sanity_check.py` | ✅ Done |
| 12 | README with Dec-POMDP formal model, architecture, evidence | `README.md` | ✅ Done |
| 13 | Proof artifacts (learning curve, GUI screenshots, full-game + NL dialogue logs, Q-tables, orchestrator logs) | `artifacts/` | ✅ Done |

---

## 4. Non-Functional Requirements

- **File size:** every `.py` file ≤ 150 code-lines.
- **Test coverage:** ≥ 85% total line coverage.
- **Lint:** `ruff` reports no errors.
- **Dependencies:** managed exclusively via **uv** (`uv run …`); no ad-hoc installs.
- **Secrets:** no real credentials or tokens committed to the repository.
- **Configuration:** all tunable game/runtime parameters live in `config.yaml`; no hard-coded
  values in source.

---

## 5. Acceptance Criteria (Rubric Gates) — Measured

| Gate | Requirement | Current Measured Value | Status |
|------|-------------|------------------------|--------|
| Tests pass | All tests green | **147 passing** | ✅ |
| Coverage | ≥ 85% total | **98%** | ✅ |
| Line limit | ≤ 150 code-lines per file | All ≤ 150 (largest 147; tight files trimmed 150 → 140) | ✅ |
| Lint | `ruff` clean | All checks pass | ✅ |
| Dependency mgmt | uv-only | `uv run pytest --cov` | ✅ |
| Secrets | none committed | Dev placeholder token only | ✅ |
| Config-driven | no hard-coding | Audited: every game parameter sourced from `config.yaml` | ✅ |

> Note: the two previously-tight files (`mcp_servers/tools.py`,
> `mcp_client/orchestrator_runner.py`) were trimmed from 150 to **140** code-lines for margin.
> Targeted tests lifted the weakest modules to `reporting/email_report.py` 99%,
> `core/state.py` 100%, and `gui/visualizer.py` 90%; every source module now clears the 85%
> bar individually (lowest: `core/config.py`, 89%). A hard-coding audit confirmed all game
> parameters flow from `config.yaml` (the only literals are config-layer defaults, UI pixel
> margins, and an always-overridden observation fallback).

---

## 6. Out of Scope / Future Work

- **Inter-group bonus competition (10 pts, optional):** a head-to-head series against a
  second team's agents, exchanged over the Inter-Group Bonus Game JSON schema. The schema
  builder (`reporting/report_schema.py::build_bonus_game_report`) is implemented and unit
  tested, but **no live cross-team series is run** — single-team self-play only. Activating it
  requires a partner team and their two public MCP URLs. Tracked in `TODO.md`.
- Replacing the deterministic NL-template fallback with full LLM dialogue in all runs.
