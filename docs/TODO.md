# TODO

**Project:** Dual AI Agent — Cop & Thief Pursuit via MCP Servers
**Course:** Orchestration of AI Agents — Dr. Yoram Segal, University of Haifa

---

## Completed (foundational work)

- [x] Grid game engine: configurable 5×5, 8-directional movement, barriers (cop only, ≤ 5), capture detection
- [x] Two separate FastMCP servers (cop + thief) exposing tools
- [x] MCP-client orchestrator driving the turn/dialogue loop
- [x] Free-form natural-language agent dialogue via LLM (GLM cloud + deterministic fallback)
- [x] Tabular Q-learning (Bellman update, epsilon-greedy)
- [x] Central `config.yaml` — no hard-coded game parameters
- [x] Scoring system (cop_win 20, thief_win 10, losses 5) over 6 sub-games per game
- [x] GUI visualization + headless screenshots
- [x] Automated JSON-only email report (Gmail API) with Internal + Inter-Group JSON schemas
- [x] Cloud deployment (Prefect Cloud) + token auth + ngrok tunneling
- [x] Staged sanity checks 2×2 → 3×3 → 4×4 → 5×5
- [x] README with Dec-POMDP formal model, architecture, and evidence
- [x] Proof artifacts in `artifacts/` (learning curve, screenshots, logs, Q-tables)
- [x] Quality gates met: **189 tests passing, strong coverage, ruff clean, all files ≤ 150 lines**
- [x] Split/trimmed oversized modules so source files stay comfortably under the line limit
- [x] Raised weak modules: `email_report.py` 77→99%, `core/state.py` 84→100%, `gui/visualizer.py` 84→90%
- [x] Hard-coding audit: confirmed all game parameters come from `config.yaml` (no functional hard-codes)

---

## Before submission

- [ ] **Confirm email target.** `config.yaml` → `report.email_target` is
      `moaawiyah.haj@icloud.com`, but the stated grader/submission address is
      `rmisegal+uoh26b@gmail.com`. Verify the correct target before final submission.
- [ ] **Fill submission metadata in `config.yaml`:** `group_name` (currently `"Team-Name"`),
      `students` (currently `[]`), and `github_repo` (currently `""`).
- [ ] **Secure the MCP auth token.** `config.yaml` → `mcp.auth_token` is the dev placeholder
      `"change-me-dev-token"`. Rotate to a real secret for any deployment and never commit it.

---

## Quality polish

- [x] **Trim/split oversized files** — source modules now stay under the
      project line-limit gate with some margin.
- [x] **Raise low-coverage modules** above the 85% gate individually:
  - [x] `reporting/email_report.py` — 77% → 99%
  - [x] `core/state.py` — 84% → 100%
  - [x] `gui/visualizer.py` — 84% → 90%
  - [x] `core/config.py` — 89% (already above the gate; left as-is)
- [x] Trimmed/split `core/engine.py` for extra line-limit margin.

---

## Stretch / bonus

- [ ] **Inter-group bonus competition (10 pts, optional).** The Bonus Game JSON builder is
      implemented and unit tested but never invoked — only single-team self-play is run. To
      claim the bonus, drive a live head-to-head series against a second team's agents using
      their two public MCP URLs. Requires a partner team; not on the critical path.
