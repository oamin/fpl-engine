"""Betfair Exchange client for live market-implied probabilities.

Secrets load from the environment or ``.env``. The app key, password, and
session token are never returned by helpers that format reports. Raw books
stay under a gitignored path; only derived mid-probabilities are reported.

Gemini 2026-10-08 (bc-e75c8209): unweighted back/lay mid, simplex
normalisation, liquidity gates. Outright expected rank is a diagnostic prior
for unpriced weeks, not an active decision input. Assists DROP. BTTS and
team clean sheet are diagnostics only.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

import httpx

ROOT = Path(__file__).resolve().parents[2]
IDENTITY_URL = "https://identitysso.betfair.com/api/login"
KEEPALIVE_URL = "https://identitysso.betfair.com/api/keepAlive"
BETTING_URL = "https://api.betfair.com/exchange/betting/rest/v1.0"
# Premier League competition id on the UK exchange.
EPL_COMPETITION_ID = "10932509"
EVENT_TYPE_SOCCER = "1"
# Relative spread gate on (lay - back) / back for a single runner mid.
MAX_REL_SPREAD = 0.25
# Match-odds totalMatched floors (GBP) for the tiered live pot.
MIN_MATCHED_TIER1 = 25_000.0
MIN_MATCHED_TIER2 = 5_000.0
MAX_SPREAD_TIER1 = 0.10
MAX_SPREAD_TIER2 = 0.20
# Prior used when shrinking a thin match-odds book (Gemini 2026-10-08).
NEUTRAL_1X2 = (0.40, 0.27, 0.33)
# Back-compat alias used by the diagnostic pull report.
MIN_MATCHED_MATCH_ODDS = MIN_MATCHED_TIER1


def load_secret(name: str) -> str:
    """Read ``name`` from the process environment or ``.env``. Empty if unset."""
    found = os.environ.get(name, "").strip()
    if found:
        return found
    path = ROOT / ".env"
    if not path.is_file():
        return ""
    prefix = f"{name}="
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def redact(text: str, *secrets: str) -> str:
    """Strip known secrets from a log or error string."""
    out = text or ""
    for secret in secrets:
        if secret:
            out = out.replace(secret, "[redacted]")
    out = re.sub(r"(X-Authentication:\s*)\S+", r"\1[redacted]", out, flags=re.I)
    out = re.sub(r"(X-Application:\s*)\S+", r"\1[redacted]", out, flags=re.I)
    out = re.sub(r'("token"\s*:\s*")[^"]+"', r'\1[redacted]"', out)
    return out


def mid_implied(
    back: float | None,
    lay: float | None,
    *,
    max_rel_spread: float = MAX_REL_SPREAD,
) -> float | None:
    """Unweighted mid of best back and best lay implied probabilities.

    Returns None when either side is missing, crossed, or wider than
    ``max_rel_spread``.
    """
    if back is None or lay is None:
        return None
    if back <= 1.0 or lay <= 1.0:
        return None
    if lay <= back:
        return None
    if (lay - back) / back > max_rel_spread:
        return None
    return 0.5 * (1.0 / back + 1.0 / lay)


def max_rel_spread(book: Mapping[str, Any], catalogue: Mapping[str, Any]) -> float | None:
    """Largest (lay-back)/back among two-sided runners. None if none qualify."""
    names_ok = {int(r["selectionId"]) for r in catalogue.get("runners") or []}
    widest: float | None = None
    for runner in book.get("runners") or []:
        if int(runner["selectionId"]) not in names_ok:
            continue
        back, lay = best_prices(runner)
        if back is None or lay is None or back <= 1.0 or lay <= back:
            continue
        rel = (lay - back) / back
        if widest is None or rel > widest:
            widest = rel
    return widest


def fair_decimal(probability: float) -> float:
    """Decimal odds for a simplex probability. Clipped away from 0 and 1."""
    p = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return 1.0 / p


def shrink_1x2(
    p_home: float,
    p_draw: float,
    p_away: float,
    *,
    matched: float,
) -> tuple[float, float, float, str]:
    """Tiered liquidity protocol. Returns probs and tier label."""
    prior_h, prior_d, prior_a = NEUTRAL_1X2
    if matched >= MIN_MATCHED_TIER1:
        return p_home, p_draw, p_away, "tier1"
    if matched >= MIN_MATCHED_TIER2:
        weight = matched / MIN_MATCHED_TIER1
        return (
            weight * p_home + (1.0 - weight) * prior_h,
            weight * p_draw + (1.0 - weight) * prior_d,
            weight * p_away + (1.0 - weight) * prior_a,
            "tier2",
        )
    return prior_h, prior_d, prior_a, "tier3_neutral"


def simplex(raw: Mapping[str, float | None], mass: float = 1.0) -> dict[str, float]:
    """Renormalise finite mids so they sum to ``mass``. Drop None entries."""
    valid = {k: float(v) for k, v in raw.items() if v is not None and v > 0.0}
    total = sum(valid.values())
    if total <= 0.0 or mass <= 0.0:
        return {}
    return {k: mass * (v / total) for k, v in valid.items()}


def expected_rank(
    p_win: float,
    p_top6: float,
    p_rel: float,
) -> float:
    """Expected finishing position from three outright probabilities.

    ``E[R] = 12 - 3 π_win - 8 π_top6 + 7 π_rel``. Clips inputs to [0, 1] and
    enforces π_top6 ≥ π_win and π_top6 + π_rel ≤ 1.
    """
    win = min(max(float(p_win), 0.0), 1.0)
    top6 = min(max(float(p_top6), 0.0), 1.0)
    rel = min(max(float(p_rel), 0.0), 1.0)
    if top6 < win:
        top6 = win
    if top6 + rel > 1.0:
        rel = max(0.0, 1.0 - top6)
    return 12.0 - 3.0 * win - 8.0 * top6 + 7.0 * rel


def strength_from_rank(rank: float) -> float:
    """Map expected rank in [1, 20] to strength in about [-1, 1]."""
    return (10.5 - float(rank)) / 9.5


def best_prices(runner: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """Best available back and lay prices from a ``listMarketBook`` runner.

    Betfair puts the ladder under ``ex`` when ``EX_BEST_OFFERS`` is requested.
    Older shapes keep the arrays on the runner root; both are accepted.
    """
    exchange = runner.get("ex") if isinstance(runner.get("ex"), Mapping) else {}
    back_levels = exchange.get("availableToBack") or runner.get("availableToBack") or []
    lay_levels = exchange.get("availableToLay") or runner.get("availableToLay") or []
    back = None
    lay = None
    for level in back_levels:
        price = level.get("price")
        if price is None:
            continue
        price = float(price)
        if back is None or price > back:
            back = price
    for level in lay_levels:
        price = level.get("price")
        if price is None:
            continue
        price = float(price)
        if lay is None or price < lay:
            lay = price
    return back, lay


class BetfairClient:
    """Minimal Exchange betting client (delayed app key is enough for reads)."""

    def __init__(
        self,
        app_key: str | None = None,
        session: str | None = None,
        *,
        timeout: float = 30.0,
    ) -> None:
        self.app_key = (app_key if app_key is not None else load_secret("BETFAIR_APP_KEY")).strip()
        self.session = (
            session if session is not None else load_secret("BETFAIR_SESSION_TOKEN")
        ).strip()
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> BetfairClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def login(
        self,
        username: str | None = None,
        password: str | None = None,
    ) -> str:
        """Interactive SSO login. Returns the session token; stores it on self."""
        user = (username if username is not None else load_secret("BETFAIR_USERNAME")).strip()
        pwd = (password if password is not None else load_secret("BETFAIR_PASSWORD")).strip()
        if not self.app_key:
            raise RuntimeError("BETFAIR_APP_KEY is not set")
        if not user or not pwd:
            raise RuntimeError(
                "Betfair needs BETFAIR_USERNAME and BETFAIR_PASSWORD "
                "(or BETFAIR_SESSION_TOKEN) in the environment"
            )
        response = self._client.post(
            IDENTITY_URL,
            headers={
                "X-Application": self.app_key,
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            data={"username": user, "password": pwd},
        )
        body = response.text
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(redact(f"login not JSON: {body[:200]}", self.app_key, pwd)) from exc
        status = str(payload.get("status") or "")
        if status != "SUCCESS" or not payload.get("token"):
            err = payload.get("error") or status or body[:200]
            raise RuntimeError(redact(f"Betfair login failed: {err}", self.app_key, user, pwd))
        self.session = str(payload["token"])
        return self.session

    def ensure_session(self) -> str:
        """Reuse ``BETFAIR_SESSION_TOKEN`` or log in with username/password."""
        if self.session:
            return self.session
        return self.login()

    def _headers(self) -> dict[str, str]:
        token = self.ensure_session()
        if not self.app_key:
            raise RuntimeError("BETFAIR_APP_KEY is not set")
        return {
            "X-Application": self.app_key,
            "X-Authentication": token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _post(self, method: str, body: Mapping[str, Any]) -> Any:
        url = f"{BETTING_URL}/{method}/"
        response = self._client.post(url, headers=self._headers(), json=dict(body))
        if response.status_code != 200:
            raise RuntimeError(
                redact(
                    f"{method} HTTP {response.status_code}: {response.text[:300]}",
                    self.app_key,
                    self.session,
                )
            )
        return response.json()

    def list_market_catalogue(
        self,
        market_filter: Mapping[str, Any],
        *,
        max_results: int = 200,
        market_projection: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        projection = market_projection or [
            "COMPETITION",
            "EVENT",
            "EVENT_TYPE",
            "MARKET_START_TIME",
            "MARKET_DESCRIPTION",
            "RUNNER_DESCRIPTION",
        ]
        result = self._post(
            "listMarketCatalogue",
            {
                "filter": dict(market_filter),
                "maxResults": max_results,
                "marketProjection": projection,
            },
        )
        return list(result or [])

    def list_market_book(
        self,
        market_ids: list[str],
        *,
        price_data: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not market_ids:
            return []
        result = self._post(
            "listMarketBook",
            {
                "marketIds": list(market_ids),
                "priceProjection": {
                    "priceData": price_data or ["EX_BEST_OFFERS"],
                    "virtualise": False,
                },
            },
        )
        return list(result or [])


def runner_mids(book: Mapping[str, Any], catalogue: Mapping[str, Any]) -> dict[str, float | None]:
    """Map runner name → mid implied probability for one market book."""
    names = {
        int(r["selectionId"]): str(r.get("runnerName") or r["selectionId"])
        for r in catalogue.get("runners") or []
    }
    out: dict[str, float | None] = {}
    for runner in book.get("runners") or []:
        sid = int(runner["selectionId"])
        name = names.get(sid, str(sid))
        back, lay = best_prices(runner)
        out[name] = mid_implied(back, lay)
    return out


def classify_outright(market_name: str) -> str | None:
    """Map a catalogue market name to winner / top6 / relegation."""
    name = market_name.lower()
    if "relegat" in name:
        return "relegation"
    if "top 6" in name or "top6" in name or "to finish in top 6" in name:
        return "top6"
    if "winner" in name or name.strip() in {"outright", "outright winner"}:
        return "winner"
    if "premier league" in name and "outright" in name:
        return "winner"
    return None


def _market_type(catalogue: Mapping[str, Any]) -> str:
    desc = catalogue.get("description") or {}
    return str(desc.get("marketType") or catalogue.get("marketName") or "").upper()


def match_probs_from_book(
    book: Mapping[str, Any],
    catalogue: Mapping[str, Any],
    *,
    home_name: str,
    away_name: str,
) -> tuple[dict[str, float], float, float | None, str] | None:
    """Simplex 1X2 probs, matched volume, max spread, tier label.

    Returns None when fewer than two runners have a two-sided mid.
    """
    from src.teams import norm_team

    mids = runner_mids(book, catalogue)
    draw = mids.get("The Draw")
    home = away = None
    home_key = norm_team(home_name)
    away_key = norm_team(away_name)
    for name, mid in mids.items():
        if name == "The Draw":
            continue
        key = norm_team(name)
        if key == home_key:
            home = mid
        elif key == away_key:
            away = mid
    # Fallback: event "Home v Away" ordering of non-draw runners.
    if home is None or away is None:
        others = [(n, v) for n, v in mids.items() if n != "The Draw"]
        if len(others) >= 2:
            home = home if home is not None else others[0][1]
            away = away if away is not None else others[1][1]
    raw = {"home": home, "draw": draw, "away": away}
    if sum(1 for v in raw.values() if v is not None) < 2:
        return None
    # Fill a missing side from residual before simplex when exactly one missing.
    present = {k: float(v) for k, v in raw.items() if v is not None}
    if len(present) == 2:
        missing = next(k for k in raw if k not in present)
        residual = max(0.0, 1.0 - sum(present.values()))
        present[missing] = residual if residual > 0 else 1e-6
    probs = simplex(present, mass=1.0)
    if len(probs) < 3:
        return None
    matched = float(book.get("totalMatched") or 0.0)
    spread = max_rel_spread(book, catalogue)
    if spread is not None and spread > MAX_SPREAD_TIER2:
        p_h, p_d, p_a, tier = NEUTRAL_1X2[0], NEUTRAL_1X2[1], NEUTRAL_1X2[2], "tier3_wide"
    else:
        p_h, p_d, p_a, tier = shrink_1x2(
            probs["home"], probs["draw"], probs["away"], matched=matched
        )
        if spread is not None and spread > MAX_SPREAD_TIER1 and tier == "tier1":
            # Demote a wide tier-1 book to shrinkage.
            weight = matched / MIN_MATCHED_TIER1
            prior_h, prior_d, prior_a = NEUTRAL_1X2
            p_h = weight * p_h + (1.0 - weight) * prior_h
            p_d = weight * p_d + (1.0 - weight) * prior_d
            p_a = weight * p_a + (1.0 - weight) * prior_a
            tier = "tier2_wide"
    # Renormalise after shrinkage.
    total = p_h + p_d + p_a
    p_h, p_d, p_a = p_h / total, p_d / total, p_a / total
    return {"home": p_h, "draw": p_d, "away": p_a}, matched, spread, tier


def ou25_decimals(book: Mapping[str, Any], catalogue: Mapping[str, Any]) -> tuple[float, float] | None:
    """Fair decimal Over/Under 2.5 from a Betfair OVER_UNDER_25 book."""
    mids = runner_mids(book, catalogue)
    over = under = None
    for name, mid in mids.items():
        label = name.lower()
        if "over" in label:
            over = mid
        elif "under" in label:
            under = mid
    if over is None or under is None:
        return None
    probs = simplex({"over": over, "under": under}, mass=1.0)
    if "over" not in probs or "under" not in probs:
        return None
    return fair_decimal(probs["over"]), fair_decimal(probs["under"])


def fetch_epl_line_quotes(
    client: BetfairClient,
    *,
    raw_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """MATCH_ODDS + OVER_UNDER_25 quotes keyed for ``lines.assemble``.

    Each quote carries fair decimal AvgH/D/A (1/p after simplex and liquidity
    tiers) so ``side_pot``'s Shin path reduces to the identity. Odds API and
    ESPN are not consulted.
    """
    catalogue = client.list_market_catalogue(
        {
            "eventTypeIds": [EVENT_TYPE_SOCCER],
            "competitionIds": [EPL_COMPETITION_ID],
            "marketTypeCodes": ["MATCH_ODDS", "OVER_UNDER_25"],
        },
        max_results=200,
    )
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "catalogue.json").write_text(
            json.dumps(catalogue, indent=2), encoding="utf-8"
        )
    market_ids = [str(m["marketId"]) for m in catalogue if m.get("marketId")]
    books = client.list_market_book(market_ids)
    if raw_dir is not None:
        (raw_dir / "books.json").write_text(json.dumps(books, indent=2), encoding="utf-8")
    book_by_id = {str(b["marketId"]): b for b in books}

    # Group by event id.
    by_event: dict[str, dict[str, Any]] = {}
    for market in catalogue:
        event = market.get("event") or {}
        eid = str(event.get("id") or "")
        if not eid:
            continue
        slot = by_event.setdefault(
            eid,
            {
                "event": event,
                "match_odds": None,
                "ou25": None,
                "start": market.get("marketStartTime") or event.get("openDate"),
            },
        )
        mtype = _market_type(market)
        name = str(market.get("marketName") or "").upper()
        if mtype == "MATCH_ODDS" or name == "MATCH ODDS":
            slot["match_odds"] = market
        elif mtype == "OVER_UNDER_25" or "OVER/UNDER 2.5" in name or "O/U 2.5" in name:
            slot["ou25"] = market

    from src.teams import norm_team as _norm

    def _key_team(name: str) -> str:
        return _norm(str(name).replace("&", " and "))

    quotes: list[dict[str, Any]] = []
    tiers: dict[str, int] = {}
    for eid, slot in by_event.items():
        market = slot.get("match_odds")
        if market is None:
            continue
        book = book_by_id.get(str(market["marketId"]))
        if book is None:
            continue
        event = slot["event"]
        event_name = str(event.get("name") or "")
        if " v " not in event_name:
            continue
        home_name, away_name = event_name.split(" v ", 1)
        home_name, away_name = home_name.strip(), away_name.strip()
        parsed = match_probs_from_book(
            book, market, home_name=home_name, away_name=away_name
        )
        if parsed is None:
            # Illiquid / one-sided → neutral pot, still emit a row so the week prices.
            p_h, p_d, p_a = NEUTRAL_1X2
            matched = float(book.get("totalMatched") or 0.0)
            tier = "tier3_unquoted"
            spread = None
        else:
            probs, matched, spread, tier = parsed
            p_h, p_d, p_a = probs["home"], probs["draw"], probs["away"]
        tiers[tier] = tiers.get(tier, 0) + 1
        over_dec = under_dec = None
        ou_market = slot.get("ou25")
        if ou_market is not None:
            ou_book = book_by_id.get(str(ou_market["marketId"]))
            if ou_book is not None:
                ou = ou25_decimals(ou_book, ou_market)
                if ou is not None:
                    over_dec, under_dec = ou
        start = str(slot.get("start") or "")
        day = start[:10] if len(start) >= 10 else ""
        if not day:
            continue
        quotes.append(
            {
                "key": (day, _key_team(home_name), _key_team(away_name)),
                "day": day,
                "home": home_name,
                "away": away_name,
                "avg_h": round(fair_decimal(p_h), 4),
                "avg_d": round(fair_decimal(p_d), 4),
                "avg_a": round(fair_decimal(p_a), 4),
                "over": None if over_dec is None else round(over_dec, 4),
                "under": None if under_dec is None else round(under_dec, 4),
                "source": "betfair",
                "books": 1,
                "tier": tier,
                "matched": matched,
                "spread": spread,
                "event_id": eid,
                "market_id": market["marketId"],
            }
        )
    meta = {
        "sent": True,
        "ok": True,
        "reason": "betfair",
        "detail": "",
        "source": "betfair",
        "n_events": len(quotes),
        "n_markets": len(catalogue),
        "tiers": tiers,
        "remaining": "",
        "last": "",
        "used": "",
    }
    return quotes, meta
