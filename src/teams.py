"""Shared team-name normalisation."""

from __future__ import annotations

TEAM_ALIASES = {
    "man united": "man utd",
    "manchester united": "man utd",
    "man utd": "man utd",
    "man city": "man city",
    "manchester city": "man city",
    "tottenham": "spurs",
    "spurs": "spurs",
    "nottingham forest": "nottm forest",
    "nott'm forest": "nottm forest",
    "nottm forest": "nottm forest",
    "wolverhampton wanderers": "wolves",
    "wolves": "wolves",
    "brighton and hove albion": "brighton",
    "brighton": "brighton",
    "newcastle united": "newcastle",
    "newcastle": "newcastle",
    "west ham united": "west ham",
    "west ham": "west ham",
    "leeds united": "leeds",
    "leeds": "leeds",
    "leicester city": "leicester",
    "leicester": "leicester",
    "ipswich town": "ipswich",
    "ipswich": "ipswich",
    "sheffield united": "sheffield utd",
    "sheffield utd": "sheffield utd",
    "afc bournemouth": "bournemouth",
    "bournemouth": "bournemouth",
    "crystal palace": "crystal palace",
    "aston villa": "aston villa",
    "arsenal": "arsenal",
    "chelsea": "chelsea",
    "everton": "everton",
    "fulham": "fulham",
    "liverpool": "liverpool",
    "brentford": "brentford",
    "burnley": "burnley",
    "sunderland": "sunderland",
}


def norm_team(name: str) -> str:
    cleaned = " ".join(str(name).lower().replace(".", "").split())
    return TEAM_ALIASES.get(cleaned, cleaned)
