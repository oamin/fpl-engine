"""Path-V diagnostic: generative FT+captain vs mean-layer FT+captain.

Protocol: ``cascade_pathv_gen_vs_mean_milp``. Same horizon/seeds/N; toggles
``generative_attack``. Diagnostic only — not a gated four-season promote.

  PYTHONPATH=. python3 -m src.eval.cascade_pathv_gen_vs_mean
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORT_MD = ROOT / "reports" / "cascade_pathv_gen_vs_mean.md"
OUT_GEN = ROOT / "data/predictions/2026-27/cascade_pathv_gen"
OUT_MEAN = ROOT / "data/predictions/2026-27/cascade_pathv_mean"
STATE_JSON = ROOT / "data/processed/cascade_pathv" / "state.json"

N = 20
SEED = 7
DECISION_GW = 6
HORIZON_END = 19
PRICED_GW = 8
SNAPSHOT_EVERY = 2


def _now() -> float:
    return time.time()


def _save(state: dict[str, Any]) -> None:
    STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATE_JSON.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _run_arm(*, generative: bool, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "src.models.multistage_mc_milp",
        "--decision-gw",
        str(DECISION_GW),
        "--horizon-end",
        str(HORIZON_END),
        "--n",
        str(N),
        "--seed",
        str(SEED),
        "--snapshot-every",
        str(SNAPSHOT_EVERY),
        "--objective",
        "xi_plus_captain_ft",
        "--no-minutes-overlay",
        "--priced-gw",
        str(PRICED_GW),
        "--out",
        str(out),
    ]
    if generative:
        cmd.append("--generative-attack")
    else:
        cmd.append("--no-generative-attack")
    label = "gen" if generative else "mean"
    print(f"[pathv] running {label} N={N}…", flush=True)
    t0 = _now()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    elapsed = _now() - t0
    # find trial json
    cands = sorted(out.glob("*trial.json"))
    trial: dict[str, Any] = {}
    if cands:
        trial = json.loads(cands[0].read_text())
    freqs = trial.get("frequencies") or []
    top = [
        {
            "name": r.get("name"),
            "freq": r.get("freq"),
            "freq_xi": r.get("freq_xi"),
            "freq_captain": r.get("freq_captain"),
            "team": r.get("team_norm"),
        }
        for r in freqs[:15]
    ]
    cap1 = float(freqs[0].get("freq_captain") or 0.0) if freqs else None
    # top captain by freq_captain
    if freqs:
        by_cap = sorted(freqs, key=lambda r: float(r.get("freq_captain") or 0), reverse=True)
        cap_top = {
            "name": by_cap[0].get("name"),
            "freq_captain": by_cap[0].get("freq_captain"),
        }
    else:
        cap_top = {}
    sources = {
        s.get("source")
        for s in (trial.get("sample_strand_scorelines") or [])
        if isinstance(s, dict)
    }
    return {
        "generative_attack": generative,
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "n_squads": trial.get("n_squads_solved"),
        "strands_with_squads": trial.get("strands_with_squads"),
        "mean_transfers": trial.get("mean_transfers"),
        "mean_hits": trial.get("mean_hits"),
        "scoring": trial.get("scoring"),
        "sources": sorted(sources),
        "top15": top,
        "top_captain": cap_top,
        "trial_path": str(cands[0]) if cands else None,
        "stdout_tail": (proc.stdout or "")[-800:],
        "stderr_tail": (proc.stderr or "")[-800:],
    }


def _jaccard_top(a: list[dict[str, Any]], b: list[dict[str, Any]], k: int = 15) -> float:
    sa = {r.get("name") for r in a[:k] if r.get("name")}
    sb = {r.get("name") for r in b[:k] if r.get("name")}
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def run() -> dict[str, Any]:
    t0 = _now()
    state: dict[str, Any] = {
        "task": "cascade_pathv_gen_vs_mean_milp",
        "started_unix": t0,
        "n": N,
        "seed": SEED,
        "decision_gw": DECISION_GW,
        "horizon_end": HORIZON_END,
    }
    _save(state)
    gen = _run_arm(generative=True, out=OUT_GEN)
    state["gen"] = gen
    _save(state)
    mean = _run_arm(generative=False, out=OUT_MEAN)
    state["mean"] = mean
    overlap = _jaccard_top(gen.get("top15") or [], mean.get("top15") or [], 15)
    state["top15_jaccard"] = overlap
    state["elapsed_s"] = _now() - t0
    state["passed_smoke"] = (
        gen.get("returncode") == 0
        and mean.get("returncode") == 0
        and int(gen.get("n_squads") or 0) > 0
        and int(mean.get("n_squads") or 0) > 0
    )
    state["finished_unix"] = _now()
    _save(state)

    lines = [
        "# Path-V diagnostic — generative vs mean-layer FT+captain",
        "",
        f"Protocol: `cascade_pathv_gen_vs_mean_milp` · N={N} · seed={SEED} · "
        f"decision GW{DECISION_GW} → horizon {HORIZON_END} · priced GW{PRICED_GW} · "
        f"objective `xi_plus_captain_ft`",
        "",
        "**Diagnostic only** — not a gated four-season path-V promote.",
        "",
        f"- Smoke passed: **{state['passed_smoke']}**",
        f"- Top-15 freq Jaccard (gen ∩ mean): **{overlap:.3f}**",
        f"- Wall: {state['elapsed_s']:.1f}s",
        "",
        "## Generative strands",
        "",
        f"- Squads: {gen.get('n_squads')} · strands_with_squads={gen.get('strands_with_squads')}",
        f"- Mean transfers / hits: {gen.get('mean_transfers')} / {gen.get('mean_hits')}",
        f"- Sources: {gen.get('sources')}",
        f"- Scoring: {gen.get('scoring')}",
        f"- Top captain: {gen.get('top_captain')}",
        f"- Elapsed: {gen.get('elapsed_s'):.1f}s",
        "",
        "Top freq:",
        "",
    ]
    for r in gen.get("top15") or []:
        lines.append(
            f"- {r.get('name')} ({r.get('team')}): freq={r.get('freq')} "
            f"xi={r.get('freq_xi')} cap={r.get('freq_captain')}"
        )
    lines += [
        "",
        "## Mean-layer strands",
        "",
        f"- Squads: {mean.get('n_squads')} · strands_with_squads={mean.get('strands_with_squads')}",
        f"- Mean transfers / hits: {mean.get('mean_transfers')} / {mean.get('mean_hits')}",
        f"- Sources: {mean.get('sources')}",
        f"- Scoring: {mean.get('scoring')}",
        f"- Top captain: {mean.get('top_captain')}",
        f"- Elapsed: {mean.get('elapsed_s'):.1f}s",
        "",
        "Top freq:",
        "",
    ]
    for r in mean.get("top15") or []:
        lines.append(
            f"- {r.get('name')} ({r.get('team')}): freq={r.get('freq')} "
            f"xi={r.get('freq_xi')} cap={r.get('freq_captain')}"
        )
    lines += [
        "",
        "Artifacts:",
        f"- `{OUT_GEN}`",
        f"- `{OUT_MEAN}`",
        "",
        "Reproduce:",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.cascade_pathv_gen_vs_mean",
        "```",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    state["report"] = str(REPORT_MD)
    _save(state)
    print(
        f"[pathv] done passed={state['passed_smoke']} jaccard={overlap:.3f} "
        f"({state['elapsed_s']:.1f}s)",
        flush=True,
    )
    return state


def main() -> None:
    run()


if __name__ == "__main__":
    main()
