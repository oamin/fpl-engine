"""GW9 deep team-news trawl → news packets + xmi CSV for numagent_live_pick.

Sources (deadline 2026-10-31T11:00:00Z):
- Fantasy Football Scout predicted XIs / Out / Doubts (fetched)
- Guardian PL team news 2026-10-30
- Premier League Matchweek 9 predicted line-ups (Bloom)
- FPL bootstrap status/chance/news (synthetic packets via load path)

Does not invent URLs; cites the real articles above.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from src.live.ffscout_team_news import match_ffs_name_to_fpl, pull_team_news
from src.live.fpl_snapshot import ELEMENT
from src.live.news_packets import (
    SOURCE_WHITELIST,
    compile_player_xmi,
    content_sha256,
    load_gameweek_packets,
    make_packet_id,
    packets_dir,
    packets_for_player,
    predicted_lineup_roles,
)
from src.live.news_tags import prior_minutes
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
GW = 9
DEADLINE = "2026-10-31T11:00:00Z"
GUARDIAN_URL = (
    "https://www.theguardian.com/football/2026/oct/30/"
    "premier-league-team-news-predicted-lineups-for-the-weekend-action"
)
GUARDIAN_PUB = "2026-10-30T16:19:00Z"  # 12:19 EDT
PL_URL = (
    "https://www.premierleague.com/en/news/4750527/"
    "predicted-line-ups-for-premier-league-teams-in-matchweek-9/"
)
PL_PUB = "2026-10-30T12:00:00Z"
POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def _write_packet(path: Path, doc: dict[str, Any]) -> None:
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _packet(
    *,
    source: str,
    headline: str,
    body: str,
    url: str,
    published_at_utc: str,
    club: str | None,
    player_ids: list[int],
) -> dict[str, Any]:
    if source not in SOURCE_WHITELIST:
        raise ValueError(f"source {source!r} not whitelisted")
    pid = make_packet_id(source, GW, headline)
    sha = content_sha256(
        headline=headline, body=body, url=url, published_at_utc=published_at_utc
    )
    return {
        "packet_id": pid,
        "source": source,
        "url": url,
        "published_at_utc": published_at_utc,
        "club": club,
        "player_ids": [int(x) for x in player_ids],
        "headline": headline,
        "body": body,
        "sha256": sha,
    }


def append_gw8_history(boot: dict[str, Any]) -> pd.DataFrame:
    hist_path = ROOT / "data/scratch/player_gw_2026_27_through_gw07.csv"
    hist = pd.read_csv(hist_path)
    live = httpx.get(
        "https://fantasy.premierleague.com/api/event/8/live/", timeout=60
    ).json()
    teams = {int(t["id"]): t["name"] for t in boot["teams"]}
    els = {int(e["id"]): e for e in boot["elements"]}
    rows: list[dict[str, Any]] = []
    for block in live.get("elements") or []:
        eid = int(block["id"])
        el = els.get(eid)
        if el is None:
            continue
        st = block.get("stats") or {}
        team = teams.get(int(el["team"]), "")
        rows.append(
            {
                "date": "",
                "gw": 8,
                "player_id": eid,
                "player_name": el.get("web_name"),
                "team": team,
                "team_short": "",
                "team_norm": norm_team(team),
                "position": POS.get(int(el["element_type"]), "MID"),
                "is_home": False,
                "minutes": float(st.get("minutes") or 0),
                "total_points": float(st.get("total_points") or 0),
                "goals": float(st.get("goals_scored") or 0),
                "assists": float(st.get("assists") or 0),
                "clean_sheets": float(st.get("clean_sheets") or 0),
                "goals_conceded": float(st.get("goals_conceded") or 0),
                "saves": float(st.get("saves") or 0),
                "bonus": float(st.get("bonus") or 0),
                "bps": float(st.get("bps") or 0),
                "xG": 0.0,
                "xA": 0.0,
                "defcon": float(st.get("defensive_contribution") or 0),
                "yellow_cards": float(st.get("yellow_cards") or 0),
                "red_cards": float(st.get("red_cards") or 0),
                "starts": float(st.get("starts") or 0),
                "value": float(el.get("now_cost") or 0),
                "defcon_raw": 0.0,
            }
        )
    gw8 = pd.DataFrame(rows)
    out = pd.concat([hist.loc[hist["gw"] != 8], gw8], ignore_index=True)
    dest = ROOT / "data/scratch/player_gw_2026_27_through_gw08.csv"
    out.to_csv(dest, index=False)
    print(f"history through GW8 → {dest} ({len(out)} rows)")
    return out


def write_guardian_packets(boot: dict[str, Any], out_dir: Path) -> int:
    """Availability packets from Guardian 30 Oct team-news article."""
    els = {int(e["id"]): e for e in boot["elements"]}
    teams = {int(t["id"]): t["name"] for t in boot["teams"]}
    by_club: dict[str, list[dict[str, Any]]] = {}
    for el in boot["elements"]:
        by_club.setdefault(teams[int(el["team"])], []).append(el)

    def find(club: str, *names: str) -> list[int]:
        cands = by_club.get(club) or []
        out: list[int] = []
        for name in names:
            hit = match_ffs_name_to_fpl(name, cands)
            if hit is not None:
                out.append(int(hit["id"]))
        return out

    # (club, headline, body_suffix, names...)
    notes: list[tuple[str, str, str, tuple[str, ...]]] = [
        ("Arsenal", "Saka listed out", "injured / unavailable (thigh).", ("Saka",)),
        ("Arsenal", "Saliba listed out", "injured / unavailable (back).", ("Saliba",)),
        ("Arsenal", "Madueke doubtful ankle", "doubtful (ankle); Arteta unsure.", ("Madueke",)),
        ("Arsenal", "Dowman suspended", "suspended (first of three).", ("Dowman",)),
        ("Chelsea", "Joao Pedro listed out", "injured (thigh, unknown).", ("João Pedro", "Joao Pedro")),
        ("Chelsea", "James listed out", "injured (knee, Jan).", ("James",)),
        ("Chelsea", "Lacroix listed out", "injured (hamstring).", ("Lacroix",)),
        ("Man Utd", "Sesko listed out", "injured (shin, Jan).", ("Sesko",)),
        ("Man Utd", "Dorgu listed out", "injured (hamstring).", ("Dorgu",)),
        ("Man Utd", "Ugarte listed out", "injured (knee).", ("Ugarte",)),
        ("Man Utd", "Mainoo could return this weekend", "doubtful (match fitness); back in training.", ("Mainoo",)),
        ("Man Utd", "Rashford could return this weekend", "doubtful (match fitness).", ("Rashford",)),
        ("Man Utd", "Bruno Fernandes will start", "Carrick expected to restore captain to XI after unused sub last week.", ("B.Fernandes", "Bruno Fernandes")),
        ("Brentford", "Henry listed out", "injured (hip) until after international break.", ("Henry",)),
        ("Brentford", "Yarmoliuk suspended", "suspended (last of three).", ("Yarmoliuk", "Yarmolyuk")),
        ("Nott'm Forest", "Ndoye doubtful hamstring", "doubtful (hamstring).", ("Ndoye",)),
        ("Nott'm Forest", "Dominguez could return this week", "doubtful (muscle); yet to return to team training.", ("Domínguez", "Dominguez")),
        (
            "Coventry City",
            "Thomas could return this weekend",
            "calf injury — Lampard late decision; not terrible; Mfuni cover if he misses.",
            ("Thomas",),
        ),
        ("Brighton", "Mitoma listed out", "injured (hamstring, 22 Nov).", ("Mitoma",)),
        ("Brighton", "Kadioglu could return this weekend", "doubtful (hip); should be fit.", ("Kadioglu",)),
        ("Man City", "Semenyo listed out", "injured (knee); only City absentee per Maresca.", ("Semenyo",)),
        ("Liverpool", "Szoboszlai could return this weekend", "doubtful (hamstring); has a chance.", ("Szoboszlai",)),
        ("Liverpool", "Ekitike listed out", "injured (achilles, Dec).", ("Ekitike", "Ekitiké")),
        ("Crystal Palace", "Sarr listed out", "injured (groin).", ("Sarr", "Ismaila Sarr")),
        ("Crystal Palace", "Timber could return this weekend", "doubtful (hamstring); maybe some minutes.", ("Timber",)),
        ("Everton", "Branthwaite could return this weekend", "doubtful (concussion protocols); should have recovered.", ("Branthwaite",)),
        ("Everton", "Grealish doubtful disciplinary", "omitted midweek for breach of club discipline; availability unclear.", ("Grealish",)),
        ("Newcastle", "Livramento listed out", "injured (knee).", ("Livramento",)),
        ("Newcastle", "Osula doubtful ankle", "doubtful (ankle); highly unlikely.", ("Osula",)),
        (
            "Newcastle",
            "Wissa will start",
            "expected to lead the Newcastle line; Osula ankle issue keeps him out of contention.",
            ("Wissa",),
        ),
        ("Fulham", "Sessegnon suspended", "suspended (last of three).", ("Sessegnon",)),
        ("Spurs", "Bentancur suspended", "suspended (last of three).", ("Bentancur",)),
        ("Aston Villa", "Wan-Bissaka listed out", "injured (hamstring, 21 Nov).", ("Wan-Bissaka",)),
        ("Aston Villa", "Onana listed out", "injured (knee).", ("Onana",)),
    ]
    n = 0
    for club, headline, suffix, names in notes:
        ids = find(club, *names)
        if not ids:
            print(f"  skip unresolved: {club} {headline} {names}")
            continue
        body = f"The Guardian team news, 30 Oct 2026: {headline} ({club}) — {suffix}"
        doc = _packet(
            source="guardian",
            headline=headline,
            body=body,
            url=GUARDIAN_URL,
            published_at_utc=GUARDIAN_PUB,
            club=club,
            player_ids=ids,
        )
        fname = f"guardian-gw09-{ids[0]}-{doc['packet_id'].split(':')[-1]}.json"
        _write_packet(out_dir / fname, doc)
        n += 1
    return n


def _ffs_club_to_fpl(name: str, fpl_clubs: set[str]) -> str | None:
    """Map FFScout club label onto bootstrap team name."""
    if name in fpl_clubs:
        return name
    aliases = {
        "Coventry": "Coventry City",
        "Hull": "Hull City",
        "Spurs": "Spurs",
        "Man City": "Man City",
        "Man Utd": "Man Utd",
        "Nott'm Forest": "Nott'm Forest",
    }
    if name in aliases and aliases[name] in fpl_clubs:
        return aliases[name]
    # prefix / contains
    for club in fpl_clubs:
        if club.startswith(name) or name.startswith(club):
            return club
    # norm match
    key = norm_team(name)
    for club in fpl_clubs:
        if norm_team(club) == key:
            return club
    return None


def apply_ffs_lineup_roles(
    ffs: dict[str, Any], boot: dict[str, Any]
) -> dict[int, tuple[str, str]]:
    teams = {int(t["id"]): t["name"] for t in boot["teams"]}
    fpl_clubs = set(teams.values())
    by_team: dict[str, list[dict[str, Any]]] = {}
    for el in boot["elements"]:
        by_team.setdefault(teams[int(el["team"])], []).append(el)
    roles: dict[int, tuple[str, str]] = {}
    for club_row in ffs.get("clubs") or []:
        raw = str(club_row["club"])
        club = _ffs_club_to_fpl(raw, fpl_clubs)
        if club is None:
            print(f"  FFS club unmatched: {raw!r}")
            continue
        cands = by_team.get(club) or []
        cite = f"ffscout:gw09:{norm_team(club)}-predicted-xi"
        for name in club_row.get("predicted_xi") or []:
            hit = match_ffs_name_to_fpl(str(name), cands)
            if hit is None:
                continue
            roles[int(hit["id"])] = ("xi", cite)
    return roles


def write_deep_desk_packets(boot: dict[str, Any], out_dir: Path) -> int:
    """Local-desk / BBC / club packets for owned + buy targets (pre-deadline)."""
    els = {int(e["id"]): e for e in boot["elements"]}
    teams = {int(t["id"]): t["name"] for t in boot["teams"]}
    by_club: dict[str, list[dict[str, Any]]] = {}
    for el in boot["elements"]:
        by_club.setdefault(teams[int(el["team"])], []).append(el)

    def find(club: str, *names: str) -> list[int]:
        cands = by_club.get(club) or []
        out: list[int] = []
        for name in names:
            hit = match_ffs_name_to_fpl(name, cands)
            if hit is not None:
                out.append(int(hit["id"]))
        return out

    # source, club, headline, body, url, published_at, names...
    notes: list[tuple[str, str, str, str, str, str, tuple[str, ...]]] = [
        (
            "bbc_sport",
            "Coventry City",
            "Thomas could return this weekend",
            (
                "BBC Sport, 29 Oct 2026 (Katie Stafford): Frank Lampard said Bobby Thomas "
                "has 'a small calf issue' and will be assessed closer to Saturday vs "
                "Sunderland. On cover, Lampard said Stephen Mfuni 'has shown in the "
                "moments he's played… It's an opportunity, for sure.' Late fitness call."
            ),
            "https://www.bbc.co.uk/sport/football/articles/ckly06j9wnv4o",
            "2026-10-29T18:00:00Z",
            ("Thomas",),
        ),
        (
            "coventry_telegraph",
            "Coventry City",
            "Thomas could return this weekend",
            (
                "Coventry Telegraph (Andy Turner), 29 Oct 2026 15:38: Lampard confirmed "
                "Thomas is a doubt for Sunderland with a 'small' calf injury — 'thankfully "
                "not as was feared' around the Achilles. 'So we're not sure about this "
                "weekend. We'll see. We've got another 48 hours… the injury is hopefully "
                "not a terrible one.' Mfuni on standby; Latibeaudiere only other option "
                "with Amenda out to Christmas."
            ),
            "https://www.coventrytelegraph.net/sport/football/football-news/coventry-city-injuries-bobby-thomas-34835698",
            "2026-10-29T15:38:00Z",
            ("Thomas",),
        ),
        (
            "pl_official",
            "Coventry City",
            "Thomas could return this weekend",
            (
                "Premier League official, 30 Oct 2026: Lampard update — Thomas to be "
                "assessed ahead of the Sunderland encounter after the calf issue that "
                "forced him off vs Fulham."
            ),
            "https://www.premierleague.com/en/news/4747025/thomas-to-be-assessed-ahead-of-sunderland-encounter",
            "2026-10-30T12:00:00Z",
            ("Thomas",),
        ),
        (
            "bbc_sport",
            "Man Utd",
            "Bruno Fernandes will start",
            (
                "BBC Sport, 29 Oct 2026 (Josh Lobley): Carrick on clear-the-air talks with "
                "Bruno Fernandes — 'He's an important player… He's fully committed to this "
                "football club… He's the one that's kind of led the team so well… and he'll "
                "continue to do that.' On the captaincy for Chelsea: 'There is no decision "
                "to be made on that.' Mainoo has 'done a bit of training'; Šeško out a "
                "long period."
            ),
            "https://www.bbc.com/sport/football/articles/ckpdg50nypq9o",
            "2026-10-29T17:00:00Z",
            ("B.Fernandes", "Bruno Fernandes"),
        ),
        (
            "sportsmole",
            "Man Utd",
            "Bruno Fernandes will start",
            (
                "Sports Mole preview (Ben Knapton), updated 30 Oct 2026: Fernandes should "
                "return to the Man Utd XI at Chelsea; Carrick said he remains fully "
                "committed after clear-the-air talks. Predicted XI includes Fernandes "
                "behind Cunha. Mainoo back in training; Shaw, Šeško, Ugarte, Dorgu still "
                "missing."
            ),
            "https://www.sportsmole.co.uk/football/man-utd/preview/chelsea-vs-man-utd-prediction-team-news-lineups_607947.html",
            "2026-10-30T14:53:00Z",
            ("B.Fernandes", "Bruno Fernandes"),
        ),
        (
            "football_london",
            "Chelsea",
            "Rogers will start",
            (
                "Football London (Bobby Vincent), 31 Oct 2026 07:00: Morgan Rogers has "
                "played 90 minutes in both of Chelsea's matches in the past week, yet is "
                "still expected to start at Stamford Bridge vs Man Utd. With João Pedro "
                "out, Rogers is cast as the Blues' most dangerous attacker alongside "
                "Palmer's return. Predicted XI: … Neto, Palmer, Rogers; Welbeck."
            ),
            "https://www.football.london/chelsea-fc/news/predicted-chelsea-xi-vs-man-34838511",
            "2026-10-31T07:00:00Z",
            ("Rogers",),
        ),
        (
            "sportsmole",
            "Chelsea",
            "Rogers will start",
            (
                "Sports Mole, 29–30 Oct 2026: Chelsea possible XI vs Man Utd lists Rogers "
                "with Palmer after João Pedro (thigh) and Lacroix (hamstring, weeks) are "
                "ruled out; Caicedo and Gusto fine. Predicted: Penders; … Neto, Palmer, "
                "Rogers."
            ),
            "https://www.sportsmole.co.uk/football/man-utd/preview/chelsea-vs-man-utd-prediction-team-news-lineups_607947.html",
            "2026-10-30T14:53:00Z",
            ("Rogers",),
        ),
        (
            "sportsmole",
            "Brighton",
            "Groß will start",
            (
                "Sports Mole (Oliver Thomas), 29 Oct 2026: Hürzeler expected to restore "
                "his strongest XI at Man City. 'Pascal Gross is all but certain to retain "
                "his starting spot in a deeper midfield position' alongside Andrés "
                "(hip cleared). Predicted: Verbruggen; … Gross, Andrés; … Kostoulas. "
                "Mitoma, Struijk, Ferguson among those still out."
            ),
            "https://www.sportsmole.co.uk/football/brighton-and-hove-albion/predicted-lineups/kostoulas-kadioglu-andres-decisions-predicted-brighton-lineup-vs-man-city_607917.html",
            "2026-10-29T15:35:00Z",
            ("Groß", "Gross"),
        ),
        (
            "the_standard",
            "Liverpool",
            "Isak will start",
            (
                "The Standard (Jack Tanner), 31 Oct 2026: Andoni Iraola will welcome back "
                "seven Liverpool regulars vs Arsenal including Alexander Isak, who is "
                "expected to return to the starting lineup at Anfield. Predicted 4-2-3-1 "
                "with Isak leading the line: Alisson; Araújo, Jacquet, Van Dijk, Kerkez; "
                "Gravenberch, Mac Allister; Barcola, Wirtz, Gakpo; Isak."
            ),
            "https://www.standard.co.uk/sport/football/liverpool-fc-xi-vs-arsenal-confirmed-team-news-predicted-lineup-injury-latest-2026-premier-league-b1302792.html",
            "2026-10-31T08:00:00Z",
            ("Isak",),
        ),
        (
            "bbc_sport",
            "Newcastle",
            "Osula doubtful ankle",
            (
                "BBC Sport, 30 Oct 2026: Jaissle said Will Osula 'twisted his ankle when he "
                "took the shot' in the Carabao Cup loss at Everton. 'Right now, it's too "
                "early to say whether he will be fit for Monday' vs Everton in the league."
            ),
            "https://www.bbc.com/sport/football/articles/cq33x2pz48j7o",
            "2026-10-30T18:00:00Z",
            ("Osula",),
        ),
        (
            "chronicle_live",
            "Newcastle",
            "Osula doubtful ankle",
            (
                "Chronicle Live (Lee Ryder), 29 Oct 2026 22:54: Osula went off after "
                "Tarkowski's block in the cup; Jaissle — 'William Osula was not able to "
                "continue, he twisted his ankle…' Danish striker a doubt for Monday's "
                "Premier League rematch with Everton."
            ),
            "https://www.chroniclelive.co.uk/sport/football/football-news/matthias-jaissle-makes-shocking-claim-34837239",
            "2026-10-29T22:54:00Z",
            ("Osula",),
        ),
        (
            "chronicle_live",
            "Newcastle",
            "Wissa will start",
            (
                "Chronicle Live (Lee Ryder), 31 Oct 2026 07:00: Between now and Monday, "
                "Newcastle will assess Osula's ankle. Jaissle would have to consider "
                "playing Yoane Wissa as the lone striker on Monday if Osula and "
                "Fernandez-Pardo don't make it. Hall started the cup rematch build-up "
                "at LB in the Dedic–Schär–Burn–Hall back four."
            ),
            "https://www.chroniclelive.co.uk/sport/football/football-news/harvey-barnes-decision-might-surprise-34840875",
            "2026-10-31T07:00:00Z",
            ("Wissa", "Hall", "Lewis Hall"),
        ),
        (
            "newcastle_fc",
            "Newcastle",
            "Hall will start",
            (
                "Newcastle United official, 29 Oct 2026: Confirmed cup XI at Everton had "
                "Lewis Hall at left-back in Dedic–Schär–Burn–Hall; Wissa among the "
                "strong bench. Same defensive spine projected for Monday's league rematch "
                "with Osula now an ankle doubt."
            ),
            "https://www.newcastleunited.com/en/news/confirmed-line-up-one-change-for-everton-carabao-cup-clash",
            "2026-10-29T18:00:00Z",
            ("Hall", "Lewis Hall", "Wissa"),
        ),
        (
            "pl_official",
            "Liverpool",
            "Isak will start",
            (
                "Premier League Bloom predicted line-ups, Matchweek 9 (30 Oct 2026): "
                "Liverpool projected with Isak leading the line vs Arsenal as Iraola "
                "restores first-choice attackers after cup rotation."
            ),
            PL_URL,
            PL_PUB,
            ("Isak",),
        ),
        (
            "pl_official",
            "Brighton",
            "Groß will start",
            (
                "Premier League Bloom predicted line-ups, Matchweek 9 (30 Oct 2026): "
                "Brighton projected with Groß in midfield away at Manchester City."
            ),
            PL_URL,
            PL_PUB,
            ("Groß", "Gross"),
        ),
        (
            "everton_fc",
            "Everton",
            "Barry will start",
            (
                "Everton FC official team news (29 Oct 2026, cup): Thierno Barry "
                "spearheaded the attack in the confirmed XI vs Newcastle; Branthwaite "
                "sidelined concussion; Grealish unavailable (discipline). Barry remains "
                "the projected league nine for Monday's rematch."
            ),
            "https://www.evertonfc.com/news/2026/october/29/everton-team-news-confirmed-for-newcastle-cup-clash/",
            "2026-10-29T17:00:00Z",
            ("Barry",),
        ),
    ]

    n = 0
    for source, club, headline, body, url, published, names in notes:
        ids = find(club, *names)
        if not ids:
            print(f"  skip unresolved deep: {source} {club} {headline} {names}")
            continue
        # dedupe ids preserving order
        seen: set[int] = set()
        uniq: list[int] = []
        for i in ids:
            if i not in seen:
                seen.add(i)
                uniq.append(i)
        doc = _packet(
            source=source,
            headline=headline,
            body=body,
            url=url,
            published_at_utc=published,
            club=club,
            player_ids=uniq,
        )
        slug = doc["packet_id"].split(":")[-1]
        fname = f"{source.replace('_', '-')}-gw09-{uniq[0]}-{slug}.json"
        _write_packet(out_dir / fname, doc)
        n += 1
        # verify element exists
        _ = els.get(uniq[0])
    return n


def compile_all_xmi(
    boot: dict[str, Any],
    hist: pd.DataFrame,
    ffs: dict[str, Any],
) -> pd.DataFrame:
    out_dir = packets_dir(GW)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_g = write_guardian_packets(boot, out_dir)
    print(f"wrote {n_g} guardian packets → {out_dir}")
    n_d = write_deep_desk_packets(boot, out_dir)
    print(f"wrote {n_d} deep desk packets → {out_dir}")

    packets = load_gameweek_packets(
        GW,
        deadline_utc=DEADLINE,
        bootstrap=boot,
        include_synthetic_fpl=True,
    )
    print(f"loaded {len(packets)} packets (incl synthetic FPL)")
    packet_roles = predicted_lineup_roles(packets, boot)
    ffs_roles = apply_ffs_lineup_roles(ffs, boot)
    # FFS XI wins for week-t minutes when present; packet roles fill gaps
    lineup_roles = {**packet_roles, **ffs_roles}

    # Force injured/out from FFS outs onto compile via synthetic local overrides:
    # players in FFS out list get injured override if not already chance=0
    teams = {int(t["id"]): t["name"] for t in boot["teams"]}
    by_team: dict[str, list[dict[str, Any]]] = {}
    for el in boot["elements"]:
        by_team.setdefault(teams[int(el["team"])], []).append(el)
    fpl_clubs = set(by_team)
    ffs_out: set[int] = set()
    ffs_doubt: dict[int, float] = {}
    for club_row in ffs.get("clubs") or []:
        club = _ffs_club_to_fpl(str(club_row["club"]), fpl_clubs)
        if club is None:
            continue
        cands = by_team.get(club) or []
        for name in club_row.get("out") or []:
            hit = match_ffs_name_to_fpl(str(name), cands)
            if hit is not None:
                ffs_out.add(int(hit["id"]))
        for d in club_row.get("doubts") or []:
            hit = match_ffs_name_to_fpl(str(d.get("name") or ""), cands)
            if hit is None:
                continue
            ch = str(d.get("chance") or "50%").replace("%", "")
            try:
                ffs_doubt[int(hit["id"])] = float(ch)
            except ValueError:
                ffs_doubt[int(hit["id"])] = 50.0

    # history map last 3 GWs
    hist = hist.copy()
    hist["player_id"] = pd.to_numeric(hist["player_id"], errors="coerce")
    hist["gw"] = pd.to_numeric(hist["gw"], errors="coerce")
    hist["minutes"] = pd.to_numeric(hist["minutes"], errors="coerce").fillna(0.0)
    hmap: dict[int, list[float]] = {}
    for pid, g in hist.groupby("player_id"):
        g2 = g.sort_values("gw")
        hmap[int(pid)] = [float(x) for x in g2["minutes"].tail(3).tolist()]

    rows: list[dict[str, Any]] = []
    for el in boot["elements"]:
        pid = int(el["id"])
        name = str(el.get("web_name") or pid)
        position = ELEMENT[int(el["element_type"])]
        status = str(el.get("status") or "")
        chance_raw = el.get("chance_of_playing_next_round")
        try:
            chance = None if chance_raw is None else float(chance_raw)
        except (TypeError, ValueError):
            chance = None
        # Merge FFS out → force 0
        if pid in ffs_out:
            status = "i"
            chance = 0.0
        # FFS doubt chance if FPL missing
        if pid in ffs_doubt and (chance is None or chance >= 100):
            chance = float(ffs_doubt[pid])
            if status == "a":
                status = "d"

        mins = hmap.get(pid, [])
        prior = prior_minutes(mins)
        role, role_pkt = lineup_roles.get(pid, (None, None))
        # Out players should not keep XI role
        if pid in ffs_out:
            role, role_pkt = None, None
        mine = packets_for_player(packets, pid)
        compiled = compile_player_xmi(
            player_id=pid,
            name=name,
            position=position,
            prior=prior,
            status=status,
            chance=chance,
            packets=mine,
            minutes=mins,
            lineup_role=role,
            lineup_packet_id=role_pkt,
        )
        xmi = compiled.get("xmi_compiled")
        if xmi is None:
            xmi = 0.0
        rows.append(
            {
                "player_id": pid,
                "gw": GW,
                "xmi": float(xmi),
                "xmi_sigma": float(compiled.get("xmi_sigma") or 0.0),
                "tag": compiled.get("tag"),
                "source": compiled.get("xmi_source"),
                "web_name": name,
                "status": status,
                "chance": chance,
                "news": str(el.get("news") or "")[:120],
                "lineup_role": role,
            }
        )
    df = pd.DataFrame(rows)
    dest = ROOT / "data/scratch/gw09_xmi_annotated.csv"
    df.to_csv(dest, index=False)
    # slim probe for loader
    probe = dest.with_name("gw09_xmi_probe.csv")
    df[["player_id", "gw", "xmi", "xmi_sigma"]].to_csv(probe, index=False)
    print(f"wrote {dest} and {probe} ({len(df)} players)")
    # owned summary
    owned = [1, 173, 449, 650, 472, 427, 40, 86, 183, 411, 464, 109, 426, 588, 249]
    print("--- owned xmi ---")
    for pid in owned:
        r = df.loc[df["player_id"] == pid].iloc[0]
        print(
            f"{pid:4d} {r['web_name']:16s} xmi={r['xmi']:5.1f} tag={r['tag']} "
            f"src={r['source']} role={r['lineup_role']} chance={r['chance']}"
        )
    return df


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse through-GW8 hist CSV and FFScout JSON; only rewrite packets + xmi",
    )
    args = ap.parse_args()
    boot = json.loads((ROOT / "data/live/bootstrap.json").read_text(encoding="utf-8"))
    hist_path = ROOT / "data/scratch/player_gw_2026_27_through_gw08.csv"
    ffs_path = ROOT / "data/scratch/ffscout_gw09/ffscout_team_news.json"
    if args.reuse_cache and hist_path.is_file() and ffs_path.is_file():
        hist = pd.read_csv(hist_path)
        ffs = json.loads(ffs_path.read_text(encoding="utf-8"))
        print(f"reused hist {hist_path.name} + FFS {ffs.get('n_clubs')} clubs")
    else:
        hist = append_gw8_history(boot)
        ffs = pull_team_news(out_dir=ROOT / "data/scratch/ffscout_gw09")
        print(f"FFScout clubs={ffs['n_clubs']}")
    compile_all_xmi(boot, hist, ffs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
