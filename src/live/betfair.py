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
# Relative spread gate on (lay - back) / back.
MAX_REL_SPREAD = 0.25
# Match-odds totalMatched floor (GBP) before a fixture mid is trusted.
MIN_MATCHED_MATCH_ODDS = 25_000.0


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


def mid_implied(back: float | None, lay: float | None) -> float | None:
    """Unweighted mid of best back and best lay implied probabilities.

    Returns None when either side is missing, crossed, or wider than
    ``MAX_REL_SPREAD``.
    """
    if back is None or lay is None:
        return None
    if back <= 1.0 or lay <= 1.0:
        return None
    if lay <= back:
        return None
    if (lay - back) / back > MAX_REL_SPREAD:
        return None
    return 0.5 * (1.0 / back + 1.0 / lay)


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
    """Best available back and lay prices from a ``listMarketBook`` runner."""
    back = None
    lay = None
    for level in runner.get("availableToBack") or []:
        price = level.get("price")
        if price is None:
            continue
        price = float(price)
        if back is None or price > back:
            back = price
    for level in runner.get("availableToLay") or []:
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
