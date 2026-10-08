"""Append-only weekly freeze for the live multi-week bar.

Gemini 2026-10-08 (bc-90156d5d): schema locked before GW6 points; first row
written at T−1h only. ``HOLDOUT_FREEZE.json`` is untouched. Forecast blocks
are immutable after append; only ``realised`` may be filled later.

Bar (locked to ``src.eval.decision_spec``): N = MIN_LIVE_WEEKS from GW6,
first formal read at LIVE_REVIEW_GW. Primary squad metric is cumulative
realised decision XI − 1FT XI (hits charged). Forecast encompassing of
``score_xp`` vs ``ep_next`` follows the existing live review constants.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.eval.decision_spec import (
    LIVE_CONTINUE_TO_GW,
    LIVE_PRIMARY_FROM_GW,
    LIVE_REVIEW_GW,
    MIN_LIVE_WEEKS,
)

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "data" / "predictions" / "2026-27" / "week_freeze.jsonl"
BAR_START_GW = int(LIVE_PRIMARY_FROM_GW)
BAR_END_GW = BAR_START_GW + int(MIN_LIVE_WEEKS) - 1  # GW6..GW25 inclusive

REQUIRED_TOP = (
    "gw",
    "deadline_utc",
    "frozen_at_utc",
    "provenance",
    "decision",
    "counterfactuals",
    "forecast_expected_points",
    "realised",
)
REQUIRED_PROVENANCE = (
    "official_capture_file",
    "official_sha256",
    "minutes_sha256",
    "odds_source",
    "odds_manifest_sha256",
    "pool_forecast_sha256",
)
REQUIRED_DECISION = (
    "chip_played",
    "squad_15",
    "starting_11",
    "captain",
    "vice_captain",
    "bench_order",
)
REQUIRED_COUNTERFACTUALS = (
    "hold_starting_11",
    "hold_captain",
    "ft1_starting_11",
    "ft1_captain",
    "ft1_move",
)
REQUIRED_FORECAST = (
    "decision_xi_c_ep_next",
    "decision_xi_c_score_xp",
    "hold_xi_c_ep_next",
    "ft1_xi_c_ep_next",
)
REQUIRED_REALISED = (
    "evaluated_at_utc",
    "actual_decision_xi_pts",
    "actual_hold_xi_pts",
    "actual_ft1_xi_pts",
    "net_gain_vs_hold",
    "net_gain_vs_ft1",
    "pool_spearman_rho",
    "clean_sheet_brier",
)


class FreezeError(RuntimeError):
    """Schema or ledger rule broken."""


class ProvenanceViolationError(FreezeError):
    """A write tried to change an immutable freeze field."""


def sha256_file(path: Path) -> str:
    """Hex digest of a file on disk."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(obj: object) -> str:
    """Stable hash of a JSON-serialisable object."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(text.encode("utf-8"))


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_keys(block: Mapping[str, Any], keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in block]
    if missing:
        raise FreezeError(f"{label} missing keys: {', '.join(missing)}")


def validate_freeze_row(row: Mapping[str, Any], *, require_null_realised: bool = False) -> None:
    """Raise FreezeError when the row fails the locked schema."""
    _require_keys(row, REQUIRED_TOP, "row")
    if not isinstance(row["gw"], int) or int(row["gw"]) < 1:
        raise FreezeError("gw must be a positive int")
    for key in ("deadline_utc", "frozen_at_utc"):
        if not isinstance(row[key], str) or not row[key]:
            raise FreezeError(f"{key} must be a non-empty string")
    prov = row["provenance"]
    if not isinstance(prov, Mapping):
        raise FreezeError("provenance must be an object")
    _require_keys(prov, REQUIRED_PROVENANCE, "provenance")
    for hash_key in (
        "official_sha256",
        "minutes_sha256",
        "odds_manifest_sha256",
        "pool_forecast_sha256",
    ):
        if not _is_sha256(prov[hash_key]):
            raise FreezeError(f"provenance.{hash_key} must be a 64-hex sha256")
    if str(prov["odds_source"]) not in {"betfair", "odds_api", "none"}:
        raise FreezeError("provenance.odds_source must be betfair|odds_api|none")
    decision = row["decision"]
    if not isinstance(decision, Mapping):
        raise FreezeError("decision must be an object")
    _require_keys(decision, REQUIRED_DECISION, "decision")
    squad = list(decision["squad_15"])
    xi = list(decision["starting_11"])
    bench = list(decision["bench_order"])
    if len(squad) != 15:
        raise FreezeError("decision.squad_15 must have 15 ids")
    if len(xi) != 11:
        raise FreezeError("decision.starting_11 must have 11 ids")
    if len(bench) != 4:
        raise FreezeError("decision.bench_order must have 4 ids")
    if set(xi) | set(bench) != set(squad):
        raise FreezeError("starting_11 and bench_order must partition squad_15")
    if decision["captain"] not in xi:
        raise FreezeError("captain must be in starting_11")
    if decision["vice_captain"] not in squad:
        raise FreezeError("vice_captain must be in squad_15")
    counter = row["counterfactuals"]
    if not isinstance(counter, Mapping):
        raise FreezeError("counterfactuals must be an object")
    _require_keys(counter, REQUIRED_COUNTERFACTUALS, "counterfactuals")
    if len(list(counter["hold_starting_11"])) != 11:
        raise FreezeError("hold_starting_11 must have 11 ids")
    if len(list(counter["ft1_starting_11"])) != 11:
        raise FreezeError("ft1_starting_11 must have 11 ids")
    forecast = row["forecast_expected_points"]
    if not isinstance(forecast, Mapping):
        raise FreezeError("forecast_expected_points must be an object")
    _require_keys(forecast, REQUIRED_FORECAST, "forecast_expected_points")
    for key in REQUIRED_FORECAST:
        value = forecast[key]
        if value is not None and not isinstance(value, (int, float)):
            raise FreezeError(f"forecast_expected_points.{key} must be numeric or null")
    realised = row["realised"]
    if not isinstance(realised, Mapping):
        raise FreezeError("realised must be an object")
    _require_keys(realised, REQUIRED_REALISED, "realised")
    if require_null_realised:
        for key in REQUIRED_REALISED:
            if realised[key] is not None:
                raise FreezeError(f"realised.{key} must be null at freeze time")


def empty_realised() -> dict[str, None]:
    return {key: None for key in REQUIRED_REALISED}


def build_freeze_row(
    *,
    gw: int,
    deadline_utc: str,
    frozen_at_utc: str,
    provenance: Mapping[str, Any],
    decision: Mapping[str, Any],
    counterfactuals: Mapping[str, Any],
    forecast_expected_points: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble one freeze row with null realised fields."""
    row = {
        "gw": int(gw),
        "deadline_utc": str(deadline_utc),
        "frozen_at_utc": str(frozen_at_utc),
        "provenance": dict(provenance),
        "decision": dict(decision),
        "counterfactuals": dict(counterfactuals),
        "forecast_expected_points": dict(forecast_expected_points),
        "realised": empty_realised(),
    }
    validate_freeze_row(row, require_null_realised=True)
    return row


def read_ledger(path: Path | None = None) -> list[dict[str, Any]]:
    """Load every freeze row. Missing file → empty list."""
    ledger = Path(path) if path is not None else LEDGER
    if not ledger.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise FreezeError(f"{ledger}:{line_no} is not JSON") from exc
        validate_freeze_row(row)
        rows.append(row)
    return rows


def write_deadline_freeze(row: Mapping[str, Any], path: Path | None = None) -> Path:
    """Append one freeze row. Refuses a second row for the same gameweek."""
    validate_freeze_row(row, require_null_realised=True)
    ledger = Path(path) if path is not None else LEDGER
    ledger.parent.mkdir(parents=True, exist_ok=True)
    existing = read_ledger(ledger)
    gw = int(row["gw"])
    if any(int(item["gw"]) == gw for item in existing):
        raise ProvenanceViolationError(f"GW{gw} already has a freeze row")
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return ledger


def _immutable_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    for key in REQUIRED_TOP:
        if key == "realised":
            continue
        if json.dumps(left[key], sort_keys=True) != json.dumps(right[key], sort_keys=True):
            return False
    return True


def attach_realised_outcomes(
    gw: int,
    realised: Mapping[str, Any],
    path: Path | None = None,
) -> dict[str, Any]:
    """Fill the realised block for one gameweek. Forecast fields stay fixed."""
    ledger = Path(path) if path is not None else LEDGER
    rows = read_ledger(ledger)
    index = next((i for i, row in enumerate(rows) if int(row["gw"]) == int(gw)), None)
    if index is None:
        raise FreezeError(f"GW{gw} has no freeze row")
    current = rows[index]
    if current["realised"].get("evaluated_at_utc") is not None:
        raise ProvenanceViolationError(f"GW{gw} realised outcomes already attached")
    merged = dict(current)
    block = dict(current["realised"])
    for key in REQUIRED_REALISED:
        if key not in realised:
            raise FreezeError(f"realised missing {key}")
        block[key] = realised[key]
    merged["realised"] = block
    validate_freeze_row(merged)
    if not _immutable_equal(current, merged):
        raise ProvenanceViolationError(f"GW{gw} immutable freeze fields changed")
    rows[index] = merged
    tmp = ledger.with_suffix(ledger.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(ledger)
    return merged


def bar_definition() -> dict[str, Any]:
    """The multi-week bar locked before GW6 points."""
    return {
        "start_gw": BAR_START_GW,
        "end_gw": BAR_END_GW,
        "n_weeks": int(MIN_LIVE_WEEKS),
        "first_read_gw": int(LIVE_REVIEW_GW),
        "continue_to_gw": int(LIVE_CONTINUE_TO_GW),
        "primary_squad_metric": "sum(actual_decision_xi_pts - actual_ft1_xi_pts) over freeze rows",
        "squad_pass": "cumulative net vs 1FT > 0 with 95% block-bootstrap interval excluding 0",
        "forecast_metric": "encompassing regression points ~ ep_next + score_xp on frozen pool",
        "forecast_pass": "beta(score_xp) > 0 at p < 0.05; else undetermined through continue_to_gw",
        "ledger": str(LEDGER.relative_to(ROOT)),
        "first_row_at": "T-1h decision window only; no pre-deadline stub rows",
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``--bar`` prints the locked bar; ``--from-json`` appends one freeze row."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bar", action="store_true", help="print the locked multi-week bar")
    parser.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="path to a freeze row JSON (realised must be null / omitted)",
    )
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.bar:
        print(json.dumps(bar_definition(), indent=2))
        return 0
    if args.from_json is None:
        parser.error("pass --bar or --from-json")
    payload = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
    if "realised" not in payload:
        payload["realised"] = empty_realised()
    path = write_deadline_freeze(payload, args.ledger)
    print(f"appended GW{int(payload['gw'])} to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
