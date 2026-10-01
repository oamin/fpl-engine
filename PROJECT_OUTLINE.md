# FPL-engine: Project Outline & Autonomous Collaboration Protocol

## Abstract
Fantasy Premier League (FPL) presents a high-dimensional combinatorial optimization problem under uncertainty. The goal of this project is to construct a predictive modeling and team-optimization engine from scratch that moves beyond standard $xP$-maximizing algorithms, exploring novel objective metrics (e.g., risk-adjusted return, differential-weighted yield, transfer stability) to uncover systemic edges.

## Operational Architecture & Governance
The project relies on a 3-tier collaborative workflow:
1. **Principal Investigator (PI):** Steers research direction, reviews milestone reports, and approves PRs/major structural pivots.
2. **Co-PI / Strategic Co-Pilot (Gemini):** Formulates mathematical logic, verifies edge cases, reviews diagnostic outputs, and authors milestone dissemination reports for the PI.
3. **Execution Engine (Cursor):** Builds, refactors, unit-tests, and executes the python codebase based on provided specifications.

---

## Hard Constraints (System & Execution Guardrails)

### Token & Resource Limits
- **Execution Retry Cap:** Maximum of 3 automated debug cycles per script failure. If unresolvable in 3 tries, revert changes and append errors to `LOGS/error_state.md`.
- **Directory Exclusions:** Never index `.venv/`, `data/raw/`, `__pycache__/`, or full-season CSV dumps in active LLM context.
- **Context Boundaries:** Keep working memory clean. Output milestone state to `REPORT_STATE.json` after completing discrete sub-tasks rather than carrying multi-day chat histories.
- **Directory Permissions:** File read/write access is strictly restricted to the `./FPL-engine` workspace.

### Core Context Memory
- A `.cursor/context.md` file must be kept up to date to log architectural state.
- An `.cursor/ideas.md` file will track candidate hypothesis ideas, novel metric definitions, and deferred features.

---

## Game Engine Rules (FPL 2026/2027 Official Standard)

All solver logic, simulator environments, and historical backtesting suites must enforce official 2026/27 rules:
- **Squad Requirements:** 15 players (2 GK, 5 DEF, 5 MID, 3 FWD), £100.0m maximum budget, max 3 players per club.
- **Transfers:** Rolling free transfers up to a maximum bank of **5 Free Transfers**. Penalty of -4 points per additional transfer.
- **Chips (8 Total):** 
  - First Half (GW1–19): Wildcard, Free Hit, Bench Boost, Triple Captain. Unused first-half chips **expire at GW19 deadline**.
  - Second Half (GW20–38): Wildcard, Free Hit, Bench Boost, Triple Captain.
- **Scoring & BPS Adjustments:** Full incorporation of 2026/27 BPS scoring rules (including DefCon thresholds and updated save/clearance metrics).

---

## Strategy & Research Direction (Soft Constraints)

- **Baseline:** Implement standard Expected Points ($xP$) optimization using odds data, historical performance, and schedule difficulty as a performance benchmark.
- **Novel Exploration Encouraged:** Actively explore alternative objective functions, including:
  - Variance-aware optimization (Sharpe ratio analog for FPL).
  - Differential weightings (ownership-adjusted utility metrics).
  - Multi-gameweek transfer churn minimization penalties.
  - Hybrid formation and bench-weight strategy exploration.
- **Evaluation efficiency is rewarded.** A cheaper screen that can kill a weak objective is part of the method, not a shortcut around it.
  - A new player score is screened on the fast XI climb (best XI each gameweek, no squad carried forward, no transfer search). Only a screen that does not clearly lose is sent to the free-transfer climb.
  - A change to the transfer rule is invisible on that fast climb, so that claim is tested on the free-transfer climb.
  - Do not shrink the transfer search and then treat the cheaper policy as the same gate.
  - Cursor and Gemini plan these screens. The PI gets the batch report, not each design choice.