# HW6 — Dual AI Agent Cop & Thief Pursuit via MCP Servers

> Course: *Orchestration of AI Agents* (Dr. Yoram Segal), University of Haifa.
> Two autonomous agents — a **Cop** (שוטר) and a **Thief** (גנב) — play a
> turn-based pursuit game on a configurable 2D grid under **partial
> observability**, communicating in **free natural language** routed through two
> **separate MCP servers**, with decisions driven by **tabular Q-Learning**.

The graded value of this project is the **orchestration and natural-language
communication between remote agents under uncertainty** — not the game-winning
strategy itself.

> **Project docs:** see [`docs/PRD.md`](docs/PRD.md) (requirements ↔ modules map),
> [`docs/PLAN.md`](docs/PLAN.md) (architecture & build phases), and
> [`docs/TODO.md`](docs/TODO.md) (open items before submission).

---

## 1. Quick start

```bash
# 1. install deps (uv resolves pyproject.toml + uv.lock into a local .venv)
uv sync

# 2. run the staged sanity checks (2x2 -> 5x5), headless
uv run python3 scripts/sanity_check.py

# 3. play a full 6-sub-game series OVER THE LIVE LLM (z.ai GLM)
#    orchestrator.py auto-loads .env, so a bare run already goes over the LLM:
uv run python3 orchestrator.py                   # networked series, verbose
uv run python3 orchestrator.py --inprocess       # same, no servers needed
#    confirm at the end:  "LLM (GLM) active: True | tokens used: {...non-zero...}"

# 4. train the Q-Learning agents (writes Q-tables + learning curve to artifacts/)
uv run python3 agents/train.py                  # full run (config.qlearning.episodes)
uv run python3 agents/train.py --episodes 2000  # fast smoke test

# 5. launch the GUI (live window, or headless screenshots for the report)
uv run python3 gui/visualizer.py                 # needs a display
uv run python3 gui/visualizer.py --headless      # renders frames to artifacts/

# 6. produce the JSON report (dry-run prints the schema-valid Internal Game JSON)
uv run python3 reporting/email_report.py --dry-run

# 7. run the test suite
uv run pytest tests/ -q
```

Package management is **uv-only**: dependencies live in `pyproject.toml`, pinned
in `uv.lock`. There is no `requirements.txt`.

**The game runs over a real LLM.** The configured backend is the **z.ai GLM
cloud API** (`glm-4.7-flashx`, OpenAI-compatible) in `config.yaml`. The API key
is read from `GLM_API_KEY` — copy `.env.example` to `.env` and fill it in (the
`.env` file is gitignored and must never be committed). Because `config.yaml`
leaves `llm.api_key` empty, the client falls back to that env var.
**`orchestrator.py` auto-loads `.env`** (see `_load_dotenv`), so a bare
`uv run python3 orchestrator.py` already runs over the LLM — no manual `export`
or wrapper needed. A successful run ends with `LLM (GLM) active: True`
and a non-zero token count — that is the proof the dialogue went over the LLM.
If `.env` is missing or `GLM_API_KEY` is unreachable, the client degrades
gracefully to deterministic NL templates (`active: False`, 0 tokens) so the
pipeline still completes on the Q-policy alone.

The same `llm/glm_client.py` speaks to **any** OpenAI-compatible endpoint, so a
local [Ollama](https://ollama.com) server (e.g. `ollama serve` + `qwen2.5:7b` at
`http://localhost:11434/v1`) is a drop-in offline alternative — just point
`llm.base_url`/`llm.model` at it and set `llm.api_key: "ollama"`.

### Running the networked MCP pipeline

The two FastMCP servers can also be run as real network services (separate
ports) with the orchestrator connecting over HTTP:

```bash
# terminal 1
uv run python3 -m mcp_servers.cop_server      # http://127.0.0.1:8101/mcp
# terminal 2
uv run python3 -m mcp_servers.thief_server    # http://127.0.0.1:8102/mcp
# terminal 3
uv run python3 -c "import orchestrator; print(orchestrator.run(networked=True, verbose=False)['totals'])"
```

The networked path was verified end-to-end: each live server handled ~350 real
MCP `POST /mcp` tool calls per series and the orchestrator completed a full
6-sub-game series with correct scores.

---

## 2. Project structure

```
hw6/
  config.yaml              # ALL parameters (no hard-coding)
  pyproject.toml / uv.lock  # dependencies (uv-only; no requirements.txt)
  README.md                # this scientific report
  core/
    config.py              # load config.yaml -> typed dataclasses
    config_schema.py       # validation of every config field (no hard-coding)
    grid.py                # 2D board, 8-dir moves, barriers, bounds/impassable
    state.py               # GameState: positions, barriers, move counter, scores
    engine.py              # turn loop, capture/timeout, sub-game & series, scoring
    observation.py         # partial-observation function O(state, agent)
    policies.py            # heuristic baseline policies (distance/random)
  agents/
    qlearning.py           # numpy Q-table, state encoding, eps-greedy, Bellman
    train.py               # self-play training -> q_*.npy + learning curve CSV/PNG
    train_utils.py         # reward shaping, eval, curve/CSV/PNG writers
    policy.py              # Q-policy with last-known-opponent belief tracking
  llm/
    glm_client.py          # fault-tolerant OpenAI-compatible LLM client (z.ai GLM by default; graceful degradation)
    prompts.py             # NL system/turn prompts (generate + parse)
  mcp_servers/
    tools.py               # shared tool impls (the game logic behind the tools)
    session.py             # per-server game session state + auth gate
    server_factory.py      # builds a FastMCP server exposing those tools
    cop_server.py          # cop MCP server (port 8101)
    thief_server.py        # thief MCP server (port 8102)
  mcp_client/
    bus.py                 # InProcessBus / NetworkedBus transports
    orchestrator_runner.py # the turn loop driving both agents over a bus
    run_logging.py         # structured per-run orchestrator logs -> artifacts/logs
  orchestrator.py          # MCP CLIENT entrypoint: owns LLM + dialogue + game loop (auto-loads .env)
  gui/
    visualizer.py          # Pygame real-time grid (headless-safe)
    render.py              # pure frame-drawing helpers (testable, no I/O)
  reporting/
    report_schema.py       # Internal Game JSON + Bonus JSON
    email_report.py        # Gmail API (OAuth) sender; dry-run by default
  deploy/
    ngrok.yaml             # optional legacy scaffold: secure local-LLM tunnel
    prefect_flow.py        # Prefect Cloud deployment scaffold (Phase 7, optional)
  docs/                    # PRD.md, PLAN.md, TODO.md (requirements, plan, open items)
  tests/                   # pytest suite (147 tests, 98% coverage)
  scripts/sanity_check.py  # staged 2x2 -> 5x5 runs
  artifacts/               # trained Q-tables, learning curves, GUI screenshots, logs
```

---

## 3. Formal model — Dec-POMDP

The pursuit is modelled as a **Decentralized Partially Observable Markov Decision
Process**, defined by the tuple

> ⟨ *n*, *S*, {*Aᵢ*}, *P*, *R*, {*Ωᵢ*}, *O*, *γ* ⟩

mapped to this game as follows:

| Symbol | Meaning | This game |
|---|---|---|
| **n** | number of agents | `2` — cop and thief |
| **S** | global state space | (cop cell, thief cell, set of barrier cells, move counter). For an *R×C* grid with *k* barriers: positions ∈ `RC × RC`, barriers ∈ `2^{RC}`. Implemented in `core/engine.py::GameState` + `core/grid.py`. |
| **{Aᵢ}** | per-agent actions | Thief: 8 moves + `stay` (9). Cop: 8 moves + `stay` + `barrier` (10). See `core/grid.py::DIRECTIONS` and `agents/qlearning.py::action_set`. |
| **P** | transition function | Deterministic move resolution with bounds/barrier blocking (off-board or into-barrier ⇒ no-op), and barrier placement mutating S. `core/grid.py::apply_move`, `core/engine.py::apply_*_action`. |
| **R** | reward / scoring | Terminal scoring table: capture ⇒ cop +20 / thief +5; survival ⇒ cop +5 / thief +10 (all from `config.yaml`). For learning, `agents/train.py` adds small Chebyshev-distance shaping so the sparse terminal signal is learnable on small grids. |
| **{Ωᵢ}** | per-agent observation space | Everything within Chebyshev `vision_radius` of the agent's own cell: own position, opponent position **iff** in range, nearby barriers. `core/observation.py::Observation`. |
| **O** | observation function | `O(state, agent)` reveals the opponent only when `chebyshev(self, opp) ≤ vision_radius`; otherwise `opponent_visible = False`. `core/observation.py::observe`. |
| **γ** | discount factor | `config.qlearning.discount_factor = 0.9`. |

**Decentralization & partial observability.** Each agent acts on its *local*
observation only; the opponent's exact cell is hidden outside the vision radius.
The agents therefore maintain a *belief* of the opponent: the Q-policy encodes the
**last-known** opponent cell (or a dedicated "unknown" slot), and the NL channel
lets each agent volunteer (or bluff about) its intentions — the orchestration
problem the assignment targets.

---

## 4. Architecture — MCP client vs. servers

The single most important rule: **the LLM is NOT inside the MCP server.**

```
                +------------------- orchestrator.py (MCP CLIENT) -------------------+
                |  owns: dialogue logic, belief tracking, Q/heuristic policies,      |
                |        and the LLM (z.ai GLM via llm/glm_client.py)                |
                +---------------------------+---------------------------------------+
                          | tool calls (HTTP / in-process)        | tool calls
                          v                                        v
        +------- cop_server.py (MCP) -------+    +------- thief_server.py (MCP) ------+
        | exposes TOOLS only:               |    | exposes TOOLS only:                |
        |  send_message / read_message      |    |  send_message / read_message       |
        |  get_observation / verify_position|    |  get_observation / verify_position |
        |  submit_move / place_barrier      |    |  submit_move / place_barrier       |
        |  game_status / advance_move_counter|   |  game_status / advance_move_counter|
        |  start_sub_game                   |    |  start_sub_game                    |
        +-----------------------------------+    +------------------------------------+
```

* **Two separate servers, separate ports** (cop `:8101`, thief `:8102`) — exactly
  the "one MCP URL per agent" requirement.
* **Free natural language, never raw coordinates.** Each turn the orchestrator
  reads the opponent's message (`read_message`), asks the LLM to generate this
  agent's NL message (`llm/prompts.py`), posts it (`send_message`), infers the
  opponent's coarse direction from their text, and only then selects a grid
  action via the Q-policy. See the sample dialogue below.
* **Token-based auth.** Every tool requires the `config.mcp.auth_token`;
  calls with a wrong token raise `AuthError`. Rotating the token revokes access.
* **Two transports, one loop.** `orchestrator.py` runs the identical turn loop
  over either an `InProcessBus` (direct tool-function calls — used by tests/CI)
  or a `NetworkedBus` (real `fastmcp.Client` HTTP calls to the live servers).

### Orchestration challenges & how they are handled

* **Linguistic ambiguity.** Free NL has no fixed schema, so the cop cannot simply
  parse coordinates out of the thief's chatter. We *constrain the channel's role*
  rather than the channel's form: the LLM's NL output is used as a soft belief
  signal (coarse direction word), while the authoritative action is taken by the
  Q-policy. The game therefore remains correct even when the dialogue is vague,
  poetic, or deceptive.
* **Deception / bluffing.** The thief's system prompt explicitly permits
  misleading messages. Because action selection does not *trust* the message
  content blindly, a bluff degrades belief quality but cannot corrupt the game
  state — robustness by design.
* **Mutual understanding & verification.** `verify_position` provides a mutual
  location-confirmation tool: an agent can assert where it believes it is and the
  authoritative server confirms/denies, decoupling bookkeeping errors from the NL
  layer.
* **Graceful degradation.** If the LLM is disabled/unreachable, the client falls
  back to deterministic NL *templates* (still free text, never coordinates) and a
  keyword direction parser, so the full pipeline runs with zero LLM dependency.

---

## 5. Q-Learning decision mechanism

Tabular Q-Learning with the Bellman update (`agents/qlearning.py`):

> Q(s,a) ← Q(s,a) + α [ r + γ · maxₐ′ Q(s′,a′) − Q(s,a) ]

* **State encoding** (partial-observation friendly):
  `index = self_cell · (num_cells + 1) + opp_slot`, where `opp_slot` is the
  *last-known* opponent cell, or a dedicated "unknown" slot when the opponent is
  out of vision. This lets one table generalise across turns of (in)visibility.
* **Actions:** cop = 8 moves + `stay` + `barrier`; thief = 8 moves + `stay`.
* **Exploration:** ε-greedy with `epsilon_start=1.0`, `epsilon_decay=0.9995`,
  `epsilon_min=0.05` (all from `config.yaml`).
* **Training:** `agents/train.py` runs cop/thief **self-play**, writes
  `artifacts/q_cop.npy`, `artifacts/q_thief.npy`, `artifacts/learning_curve.csv`
  and `artifacts/learning_curve.png`.

**Result (2000-episode smoke run, 5×5):** the trained cop captured the heuristic
thief in **99%** of evaluation sub-games versus **14%** for a random cop — a clear
win over the baseline.

---

## 6. Visualizations & evidence

All artifacts are written to `artifacts/`.

### 6.1 Q-Table learning curve
`artifacts/learning_curve.png` — the cop's moving-average episode reward (blue)
rises and the capture rate (green) converges toward ~1.0 as ε (red) decays,
demonstrating successful learning.

### 6.2 GUI screenshots
`artifacts/gui_*.png` — real-time Pygame rendering of the grid showing the cop
(blue **C**), the thief (red **T**), barriers, the move counter, live scores, and
the latest NL messages from each agent. Generated headlessly with
`python3 gui/visualizer.py --headless`.

### 6.3 CLI logs — real natural-language dialogue
`artifacts/nl_dialogue_log.txt` — a full series run with the **live LLM (z.ai GLM `glm-4.7-flashx`)**
enabled. Excerpt:

```
[ 1] THIEF says: "You think you're tracking me well, but I'm staying one step ahead..." -> {'type': 'move', 'action': 'N'}
     COP   says: "Evasive as ever, I sense your presence nearby... every move you make is leading closer to a corner..." -> {'type': 'move', 'action': 'E'}
...
 >> capture! cop landed on thief at (1, 2)
```

The agents exchange genuine free-text tactical messages (including bluffing) while
the Q-policy drives the actual grid moves — exactly the orchestration the
assignment asks for.

---

## 7. Cloud deployment & security (Approach 2 — optional, Phase 7)

Not exercised by the local pipeline, but scaffolded:

* `deploy/ngrok.yaml` — OPTIONAL legacy scaffold: exposes a **local** LLM
  (`127.0.0.1:11434`) over HTTPS behind an ngrok **Traffic Policy** that enforces
  **Basic Auth**, so only requests with the right `Authorization` header reach the
  model. The default backend is the **z.ai GLM cloud API** (Approach 1), which
  needs no tunnel; the ngrok scaffold only matters if you instead expose a
  *local* model (e.g. Ollama) to a remote orchestrator. Set `llm.base_url` /
  `llm.api_key` / `llm.model` in `config.yaml` to point at any other
  OpenAI-compatible endpoint.
* `deploy/prefect_flow.py` — a Prefect flow wrapping each MCP server as a managed
  process with a public URL; access stays **token-gated** (rotate
  `config.mcp.auth_token` to revoke).

Security note: the orchestrator only makes **outbound** calls, so no
inbound ports need opening on the development machine.

---

## 8. Reporting

After the 6th sub-game the **cop** sends a single email whose body is **JSON
only** to `config.report.email_target` via the **Gmail API with OAuth** (a token,
not a password). To keep the project testable without credentials, sending is
**dry-run by default**:

```bash
uv run python3 reporting/email_report.py --dry-run   # prints schema-valid Internal Game JSON
uv run python3 reporting/email_report.py --play      # play a real series, then report
uv run python3 reporting/email_report.py --send      # really send (needs OAuth creds)
```

For `--send`, place an OAuth client secret at `reporting/credentials.json`
(Gmail API enabled in Google Cloud Console); a user `token.json` is cached on
first run. `reporting/report_schema.py` also builds the **Inter-Group Bonus JSON**
.

### 8.1 Inter-group bonus series — implemented but not exercised

The optional inter-group competition is supported **at the schema level
only**. `reporting/report_schema.py::build_bonus_game_report` emits a fully
schema-valid **Inter-Group Bonus JSON** — two groups, four MCP URLs, the
role-swap pairing (3 sub-games of group-A cop vs group-B thief, then 3 reversed),
`totals_by_group`, `bonus_claim`, and `mutual_agreement` — and it is unit
tested (`tests/test_report_schema.py`). **No live cross-team series is ever run,
however:** this submission performs single-team self-play only, so the bonus
builder is never invoked by the pipeline and the two-pair game structure exists as
ready-to-use scaffolding rather than executed evidence. Activating it requires a
partner team and their two public MCP URLs.

---

## 9. Configuration (no hard-coding)

Every game parameter lives in `config.yaml`: `grid_size`, `max_moves`,
`num_games`, `max_barriers`, the full `scoring` table, `vision_radius`,
`allow_diagonal`, start rules, LLM backend (z.ai GLM), MCP host/port/token, Q-Learning
hyper-parameters, and report metadata. `core/config.py` validates these into typed
dataclasses; the rest of the code never hard-codes a game constant.

**Team metadata** in `config.yaml` (`report.group_name`, `report.students`,
`report.github_repo`, `report.cop_mcp_url`, `report.thief_mcp_url`) are
placeholders — fill them in before submission.

---

## 10. Verification summary

| Check | Command | Result |
|---|---|---|
| Unit tests | `uv run pytest -q` | **147 passed** |
| Coverage | `uv run pytest --cov` | **98%** total (gate ≥85%) |
| Lint | `uv run ruff check .` | all checks passed |
| File size | ≤150 code-lines/file | largest ≤150 (gate ≤150) |
| Staged sanity | `scripts/sanity_check.py` | full 6-sub-game series at 2×2→5×5, sensible scores |
| Local series (in-process) | `orchestrator.py --inprocess` | completes 6 sub-games autonomously with NL logs |
| Local series (networked) | live servers + `orchestrator.run(networked=True)` | completes via real MCP HTTP tool calls |
| Q-Learning | `agents/train.py --episodes 2000` | trained cop 99% capture vs 14% random; curve + CSV emitted |
| GUI | `gui/visualizer.py --headless` | renders + saves screenshots headlessly |
| Reporting | `reporting/email_report.py --dry-run` | prints schema-valid Internal Game JSON |
```
