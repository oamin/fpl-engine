"""Layer 5 Steps 1–3 audit — legality, TC/BB scoring, WC/FH state.

  python -m src.strategy.layer5_audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.strategy.layer5_week import audit_legality_cases, summarise_audit

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def write_report(doc: dict, path: Path) -> None:
    lines = [
        "# Layer 5 audit — Steps 1–3",
        "",
        "Helpers wrap `ChipWallet` / `hit_cost` / `advance_ft` / `captain_multiplier`. "
        "Step 2 = TC/BB scoring identity. Step 3 = WC keeps rebuild / FH reverts / FT "
        "preserved. Joint chip+FT foresight is Step 4. No play thresholds.",
        "",
        f"Cases passed: **{doc['n_pass']}/{doc['n_checks']}**",
        "",
        "| check | pass | detail |",
        "| --- | --- | --- |",
    ]
    for c in doc["checks"]:
        lines.append(
            f"| `{c['name']}` | {'PASS' if c['pass'] else 'FAIL'} | {c.get('detail', '')} |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- Step 1–3 gate physics only. Step 4 = strategy (joint outer beam).",
        "- Inner boards: priced μ_post, γ=1, unconstrained hits.",
        "",
        "JSON: `data/processed/strategy_layer5_audit.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    checks = audit_legality_cases()
    doc = summarise_audit(checks)
    out_json = PROCESSED / "strategy_layer5_audit.json"
    out_md = REPORTS / "strategy_layer5_audit.md"
    out_json.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    write_report(doc, out_md)
    print(json.dumps({"n_pass": doc["n_pass"], "n_checks": doc["n_checks"], "passed": doc["passed"]}, indent=2))
    print(f"wrote {out_md}")
    return 0 if doc["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
