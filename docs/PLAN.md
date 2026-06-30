# Implementation / Architecture Plan

**Project:** Dual AI Agent — Cop & Thief Pursuit via MCP Servers
**Course:** Orchestration of AI Agents — Dr. Yoram Segal, University of Haifa

---

## 1. Engineering Priority Order (8 Phases)

The assignment prescribes a build order that establishes a working core before layering on
intelligence and infrastructure. All phases are complete.

| Phase | Goal | Realizing Modules | Status |
|-------|------|-------------------|--------|
| 1 | **Game logic** — grid, movement, barriers, capture | `core/grid.py`, `core/engine.py`, `core/state.py`, `core/observation.py`, `core/policies.py` | ✅ Complete |
| 2 | **Basic MCP transport** — two servers exposing tools | `mcp_servers/cop_server.py`, `thief_server.py`, `server_factory.py`, `tools.py`, `session.py` | ✅ Complete |
| 3 | **Local pipeline** — orchestrator drives end-to-end turn loop | `orchestrator.py`, `mcp_client/orchestrator_runner.py`, `bus.py`, `run_logging.py` | ✅ Complete |
| 4 | **Decision mechanism** — tabular Q-learning | `agents/qlearning.py`, `train.py`, `train_utils.py`, `policy.py` | ✅ Complete |
| 5 | **NL integration** — free-form LLM dialogue | `llm/glm_client.py`, `llm/prompts.py` | ✅ Complete |
| 6 | **GUI** — visualization + headless screenshots | `gui/visualizer.py`, `gui/render.py` | ✅ Complete |
| 7 | **Cloud deploy** — Prefect Cloud + tunneling/security | `deploy/prefect_flow.py`, `deploy/ngrok.yaml` | ✅ Complete |
| 8 | **Gmail reporting** — JSON-only automated email | `reporting/email_report.py`, `reporting/report_schema.py` | ✅ Complete |

---

## 2. Architecture (Text Diagram)

The LLM lives **client-side** within the orchestrator. The two MCP servers are "thin": they
expose tools that mutate/query game state and never embed reasoning.

```
                       ┌──────────────────────────────────────────┐
                       │            ORCHESTRATOR (MCP Client)        │
                       │                                            │
                       │   ┌──────────────┐    ┌────────────────┐   │
                       │   │   LLM (GLM)  │    │  Q-learning    │   │
                       │   │  NL dialogue │    │  policy/Q-tab  │   │
                       │   └──────────────┘    └────────────────┘   │
                       │            turn loop / bus / logging        │
                       └───────┬───────────────────────────┬────────┘
                               │ HTTP (MCP)                 │ HTTP (MCP)
                               │ token auth                 │ token auth
                       ┌───────▼─────────┐         ┌────────▼────────┐
                       │  COP MCP SERVER │         │ THIEF MCP SERVER │
                       │  (FastMCP)      │         │  (FastMCP)       │
                       │  tools: move,   │         │  tools: move,    │
                       │  observe, talk, │         │  observe, talk   │
                       │  place_barrier  │         │                  │
                       └───────┬─────────┘         └────────┬─────────┘
                               │                            │
                               └────────────┬───────────────┘
                                            ▼
                                  ┌───────────────────┐
                                  │   GAME ENGINE     │
                                  │ grid / state /    │
                                  │ scoring / capture │
                                  └───────────────────┘
```

- **One server per agent** (cop, thief) — true decentralized topology.
- **Reasoning is centralized in the client** (LLM + Q-policy); servers are tool surfaces only.
- Servers communicate over **HTTP** with **token authentication**; cloud exposure is
  tunneled (see §5).

---

## 3. Data Flow for One Turn

1. **Observation** — orchestrator requests the acting agent's local observation from its MCP
   server (`core/observation.py`): partial view of the grid per the Dec-POMDP {Ωᵢ}, O.
2. **NL message** — the observation is rendered into a prompt (`llm/prompts.py`); the LLM
   (`llm/glm_client.py`) emits a **free-form natural-language** message describing intent.
   Without an API key, a deterministic NL template produces an equivalent message.
3. **Decision → tool call** — the Q-policy (`agents/policy.py`) selects an action
   (epsilon-greedy over the Q-table); the orchestrator issues the corresponding **MCP tool
   call** (e.g. `move`, `place_barrier`) to the agent's server.
4. **State update** — the server applies the action through the **engine** (`core/engine.py`,
   `core/state.py`): movement (8-directional), barrier placement (cop only, ≤ 5), and
   **capture detection**.
5. **Score** — on a sub-game terminal state, the engine applies scoring (cop_win 20,
   thief_win 10, losses 5); results and the NL dialogue are appended to run logs
   (`mcp_client/run_logging.py`).

Six sub-games constitute one game; the Q-tables are updated via the Bellman rule across
episodes during training (`agents/train.py`).

---

## 4. Testing & Quality Strategy

- **Runner:** `uv run pytest --cov` — single source of truth for tests and coverage.
- **Coverage gate (≥ 85%):** enforced in CI/local; current total **98%** across **147 tests**,
  with every source module individually above the bar (lowest `core/config.py`, 89%).
- **Lint gate:** `uv run ruff check` must report no errors.
- **Line-limit gate (≤ 150 code-lines/file):** verified per module; the two previously-tight
  files were trimmed from 150 to 140 code-lines, leaving `core/engine.py` (147) as the
  current ceiling.
- **Config-driven testing:** sanity ladder (`scripts/sanity_check.py`) exercises the engine
  at increasing scale **2×2 → 3×3 → 4×4 → 5×5** to catch boundary/scaling regressions before
  full 5×5 runs.
- **Determinism:** the NL-template fallback keeps the full pipeline runnable and testable
  without external LLM access, so CI does not depend on a live API key.

---

## 5. Deployment Plan

1. **Local (two-port) run** — cop and thief MCP servers launched on separate localhost ports;
   orchestrator connects to both. This is the development and demo baseline.
2. **Cloud (Prefect Cloud)** — the run is wrapped as a Prefect flow (`deploy/prefect_flow.py`)
   and deployed to Prefect Cloud for scheduled/managed execution (verified real deployment).
3. **Tunneling & security** — server exposure is gated by **token authentication**
   (`mcp.auth_token` in `config.yaml`) and routed through an **ngrok** tunnel
   (`deploy/ngrok.yaml`). The committed token is a development placeholder and must be rotated
   for any real deployment; the public tunnel URL is opened only on demand.
