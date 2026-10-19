"""Audited news packets: whitelist, leakage gate, synthetic FPL fallback."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.live import news_packets as np


def _packet(**overrides: object) -> dict:
    base = {
        "packet_id": "bbc_sport:gw06:test-hamstring",
        "source": "bbc_sport",
        "url": "https://www.bbc.com/sport/football/articles/van-ewijk-hamstring",
        "published_at_utc": "2026-10-09T12:00:00Z",
        "club": "Coventry City",
        "player_ids": [175],
        "headline": "van Ewijk hamstring update",
        "body": "van Ewijk is doubtful with a hamstring injury.",
    }
    base["sha256"] = np.content_sha256(
        headline=str(base["headline"]),
        body=str(base["body"]),
        url=str(base["url"]),
        published_at_utc=str(base["published_at_utc"]),
    )
    base.update(overrides)
    if "sha256" not in overrides and any(
        key in overrides for key in ("headline", "body", "url", "published_at_utc")
    ):
        base["sha256"] = np.content_sha256(
            headline=str(base["headline"]),
            body=str(base["body"]),
            url=str(base["url"]),
            published_at_utc=str(base["published_at_utc"]),
        )
    return base


class NewsPackets(unittest.TestCase):
    def test_validate_ok(self) -> None:
        packet = np.validate_packet(_packet(), deadline_utc="2026-10-10T10:00:00Z")
        self.assertEqual(packet.source, "bbc_sport")
        doc = packet.to_packet_doc()
        self.assertIn("hamstring", doc.text.lower())

    def test_rejects_late_packet(self) -> None:
        raw = _packet(published_at_utc="2026-10-10T11:00:00Z")
        with self.assertRaises(np.LeakageError):
            np.validate_packet(raw, deadline_utc="2026-10-10T10:00:00Z")

    def test_rejects_placeholder_url(self) -> None:
        raw = _packet(url="https://www.mancity.com/news/haaland-presser-example")
        with self.assertRaises(np.PacketError) as caught:
            np.validate_packet(raw, deadline_utc="2026-10-10T10:00:00Z")
        self.assertIn("placeholder", str(caught.exception))

    def test_every_club_has_a_local_desk(self) -> None:
        self.assertEqual(len(np.CLUB_PRESS), 20)
        for club, sources in np.CLUB_PRESS.items():
            self.assertTrue(sources, club)
            for source in sources:
                self.assertIn(source, np.SOURCE_WHITELIST, club)

    def test_ask_fills_last_three_games(self) -> None:
        row = np.compile_player_xmi(
            player_id=1,
            name="Lammens",
            position="GKP",
            prior=90.0,
            status="a",
            chance=None,
            packets=[],
            minutes=[90.0, 0.0, 90.0, 90.0, 45.0],
        )
        self.assertEqual(row["tag"], "rolling avg")
        self.assertEqual(row["xmi_source"], "rolling")
        self.assertAlmostEqual(float(row["xmi_compiled"]), 75.0)
        # Last three: 90, 90, 45 → sample SD
        self.assertAlmostEqual(float(row["xmi_sigma"]), np.rolling_game_minutes_sigma([90.0, 90.0, 45.0]))

    def test_starter_press_keeps_rolling_default(self) -> None:
        """Affirmative starter press is not an override tag — rolling stays."""
        deadline = "2026-10-10T10:00:00Z"
        packet = np.validate_packet(
            _packet(
                source="bbc_sport",
                packet_id="bbc_sport:gw06:starter-note",
                url="https://www.bbc.com/sport/football/articles/will-start",
                headline="Player will start",
                body="He will start and is first choice this weekend.",
                published_at_utc="2026-10-08T12:00:00Z",
                player_ids=[1],
                club="Arsenal",
            ),
            deadline_utc=deadline,
        )
        tag, _ = np.classify_packets_deterministic([packet], player_name="Player")
        self.assertEqual(tag, "none")
        row = np.compile_player_xmi(
            player_id=1,
            name="Player",
            position="MID",
            prior=90.0,
            status="a",
            chance=None,
            packets=[packet],
            minutes=[60.0, 70.0, 80.0],
        )
        self.assertEqual(row["context_tag"], "none")
        self.assertEqual(row["tag"], "rolling avg")
        self.assertEqual(row["xmi_source"], "rolling")
        self.assertAlmostEqual(float(row["xmi_compiled"]), 70.0)

    def test_hard_flag_forces_zero(self) -> None:
        row = np.compile_player_xmi(
            player_id=1,
            name="Out",
            position="DEF",
            prior=90.0,
            status="s",
            chance=None,
            packets=[],
            minutes=[90.0, 90.0, 90.0],
        )
        self.assertEqual(float(row["xmi_compiled"]), 0.0)
        self.assertEqual(row["xmi_source"], "override:hard_flag")

    def test_rolling_sigma_helpers(self) -> None:
        self.assertAlmostEqual(np.rolling_game_minutes_sigma([90.0, 90.0, 45.0]), (1350.0 / 2) ** 0.5, places=6)
        self.assertEqual(np.rolling_game_minutes_sigma([90.0]), 0.0)
        # firm starter uses ≥60′ apps in the window when enough exist
        sig = np.sigma_m_for_regime(
            tag="firm_starter",
            xmi=90.0,
            minutes=[90.0, 20.0, 90.0, 85.0],
            prior=90.0,
            window=3,
        )
        # window = 20, 90, 85 → only 90 and 85 are starts → SD of those two
        self.assertAlmostEqual(sig, (12.5) ** 0.5, places=6)
        mix = np.sigma_m_for_regime(tag="50/50", xmi=45.0, minutes=[90.0, 90.0, 90.0], prior=90.0)
        self.assertAlmostEqual(mix, 45.0)
        self.assertEqual(np.sigma_m_for_regime(tag="injured", xmi=0.0, minutes=[90.0, 80.0, 70.0]), 0.0)

    def test_could_return_is_not_ruled_out(self) -> None:
        deadline = "2026-10-10T10:00:00Z"
        packet = np.validate_packet(
            _packet(
                source="the_standard",
                packet_id="the_standard:gw06:dowman-could-return",
                url="https://www.standard.co.uk/sport/football/premier-league-injury-update-latest-news-return-dates-all-clubs-b1300072.html",
                headline="Dowman could return this weekend",
                body="The Standard says Max Dowman could return this weekend (knock).",
                published_at_utc="2026-10-08T17:34:13Z",
                player_ids=[1],
                club="Arsenal",
            ),
            deadline_utc=deadline,
        )
        tag, cited = np.classify_packets_deterministic([packet], player_name="Dowman")
        self.assertEqual(tag, "50/50")
        self.assertEqual(cited, ["the_standard:gw06:dowman-could-return"])
        row = np.compile_player_xmi(
            player_id=1,
            name="Dowman",
            position="MID",
            prior=90.0,
            status="a",
            chance=None,
            packets=[packet],
        )
        self.assertEqual(row["tag"], "50/50")
        self.assertAlmostEqual(float(row["xmi_compiled"]), 45.0)
        self.assertAlmostEqual(float(row["xmi_sigma"]), 45.0)

    def test_surname_match_breaks_miley_and_folds_norgaard(self) -> None:
        elements = [
            {"id": 65, "team": 3, "web_name": "J.Araujo", "first_name": "Julián", "second_name": "Araujo Zúñiga"},
            {"id": 78, "team": 3, "web_name": "Kroupi.Jr", "first_name": "Junior", "second_name": "Kroupi"},
            {"id": 21, "team": 9, "web_name": "Nørgaard", "first_name": "Christian", "second_name": "Nørgaard"},
            {"id": 405, "team": 17, "web_name": "N.Gonzalez", "first_name": "Nico", "second_name": "González Iglesias"},
            {"id": 459, "team": 17, "web_name": "L.Miley", "first_name": "Lewis", "second_name": "Miley"},
            {"id": 659, "team": 17, "web_name": "Miley", "first_name": "Mason", "second_name": "Miley"},
        ]
        teams = {3: "Bournemouth", 9: "Everton", 17: "Newcastle"}
        cases = [
            ("Julian Araujo", "Bournemouth", 65),
            ("Eli Junior Kroupi", "Bournemouth", 78),
            ("Christian Norgaard", "Everton", 21),
            ("Nico Gonzalez", "Newcastle", 405),
            ("Lewis Miley", "Newcastle", 459),
        ]
        for name, club, pid in cases:
            self.assertEqual(np.match_element_id(name, club, elements, teams), pid, name)
        self.assertIsNone(np.match_element_id("Miley", "Newcastle", elements, teams))

    def test_local_outlets_are_whitelisted(self) -> None:
        raw = _packet(source="sportsmole", packet_id="sportsmole:gw06:haaland")
        packet = np.validate_packet(raw, deadline_utc="2026-10-10T10:00:00Z")
        self.assertEqual(packet.source, "sportsmole")

    def test_latest_available_note_supersedes_injury_doubt(self) -> None:
        deadline = "2026-10-10T10:00:00Z"
        early = np.validate_packet(
            _packet(
                packet_id="bbc_sport:gw06:haaland-doubt",
                headline="Haaland injury doubt",
                body="Erling Haaland is an injury doubt after he limped off.",
                published_at_utc="2026-10-05T14:54:39Z",
                player_ids=[411],
            ),
            deadline_utc=deadline,
        )
        late = np.validate_packet(
            _packet(
                source="sportsmole",
                packet_id="sportsmole:gw06:haaland-fatigue",
                url="https://www.sportsmole.co.uk/football/man-city/haaland-fatigue",
                headline="Haaland minor doubt for Liverpool",
                body=(
                    "Status: minor doubt. Type of issue: fatigue. "
                    "Haaland is not nursing an injury and should be available for selection."
                ),
                published_at_utc="2026-10-08T07:10:00Z",
                player_ids=[411],
            ),
            deadline_utc=deadline,
        )
        tag, cited = np.classify_packets_deterministic([early, late], player_name="Haaland")
        self.assertEqual(tag, "none")
        self.assertEqual(cited, ["sportsmole:gw06:haaland-fatigue"])

    def test_rejects_unwhitelisted_source(self) -> None:
        raw = _packet(source="random_blog", packet_id="random_blog:gw06:x")
        with self.assertRaises(np.WhitelistError):
            np.validate_packet(raw, deadline_utc="2026-10-10T10:00:00Z")

    def test_rejects_bad_hash(self) -> None:
        raw = _packet(sha256="0" * 64)
        with self.assertRaises(np.PacketError):
            np.validate_packet(raw, deadline_utc="2026-10-10T10:00:00Z")

    def test_load_files_and_empty_dir(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            deadline = "2026-10-10T10:00:00Z"
            self.assertEqual(np.load_packet_files(root / "missing", deadline_utc=deadline), [])
            dest = np.packets_dir(6, root)
            written = np.write_packet(
                np.validate_packet(_packet(), deadline_utc=deadline),
                dest,
            )
            self.assertTrue(written.is_file())
            loaded = np.load_gameweek_packets(
                6, deadline_utc=deadline, root=root, include_synthetic_fpl=False
            )
            self.assertEqual(len(loaded), 1)

    def test_compile_high_profile_packets(self) -> None:
        deadline = "2026-10-10T10:00:00Z"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dest = np.packets_dir(6, root)
            raw = _packet()
            np.write_packet(np.validate_packet(raw, deadline_utc=deadline), dest)
            bootstrap = {
                "teams": [{"id": 7, "name": "Coventry City"}],
                "elements": [
                    {
                        "id": 175,
                        "web_name": "van Ewijk",
                        "element_type": 2,
                        "team": 7,
                        "status": "d",
                        "chance_of_playing_next_round": 75,
                        "news": "Hamstring injury - 75% chance of playing",
                        "news_added": "2026-10-08T10:00:00Z",
                    }
                ],
            }
            result = np.compile_high_profile_test(
                gw=6,
                deadline_utc=deadline,
                bootstrap=bootstrap,
                player_ids=[175],
                history={175: [90.0, 90.0, 90.0]},
                root=root,
            )
            row = result["players"][0]
            self.assertEqual(row["tag"], "injured")
            self.assertAlmostEqual(float(row["xmi_compiled"]), 67.5)
            self.assertIn("van Ewijk", result["context_markdown"])

    def test_synthetic_fpl_from_bootstrap(self) -> None:
        bootstrap = {
            "teams": [{"id": 7, "name": "Coventry City"}],
            "elements": [
                {
                    "id": 175,
                    "web_name": "van Ewijk",
                    "element_type": 2,
                    "team": 7,
                    "status": "d",
                    "chance_of_playing_next_round": 75,
                    "news": "Hamstring injury - 75% chance of playing",
                    "news_added": "2026-10-08T10:00:00Z",
                }
            ],
        }
        packets = np.synthetic_fpl_packets(
            bootstrap, gw=6, deadline_utc="2026-10-10T10:00:00Z"
        )
        self.assertEqual(len(packets), 1)
        self.assertEqual(packets[0].source, "fpl_bootstrap")
        cases = np.build_player_cases(
            packets, bootstrap, gw=6, deadline_utc="2026-10-10T10:00:00Z"
        )
        self.assertEqual(cases[0]["player_id"], 175)
        self.assertEqual(len(cases[0]["docs"]), 1)

    def test_compound_surname_matches_last_token(self) -> None:
        elements = [
            {
                "id": 390,
                "team": 15,
                "web_name": "Rúben",
                "first_name": "Rúben",
                "second_name": "dos Santos Gato Alves Dias",
            }
        ]
        teams = {15: "Man City"}
        self.assertEqual(np.match_element_id("Dias", "Man City", elements, teams), 390)

    def test_predicted_xi_rotates_out_earlier_starter(self) -> None:
        """MEN XI includes Gvardiol; later Standard XI does not → rotated."""
        deadline = "2026-10-17T10:00:00Z"
        bootstrap = {
            "teams": [{"id": 15, "name": "Man City", "short_name": "MCI"}],
            "elements": [
                {"id": 384, "team": 15, "web_name": "Donnarumma", "first_name": "Gianluigi", "second_name": "Donnarumma", "element_type": 1, "status": "a"},
                {"id": 389, "team": 15, "web_name": "Nunes", "first_name": "Matheus", "second_name": "Nunes", "element_type": 2, "status": "a"},
                {"id": 393, "team": 15, "web_name": "Khusanov", "first_name": "Abdukodir", "second_name": "Khusanov", "element_type": 2, "status": "a"},
                {"id": 388, "team": 15, "web_name": "Guéhi", "first_name": "Marc", "second_name": "Guéhi", "element_type": 2, "status": "a"},
                {"id": 391, "team": 15, "web_name": "Gvardiol", "first_name": "Joško", "second_name": "Gvardiol", "element_type": 2, "status": "a"},
                {"id": 390, "team": 15, "web_name": "Rúben", "first_name": "Rúben", "second_name": "dos Santos Gato Alves Dias", "element_type": 2, "status": "a"},
                {"id": 387, "team": 15, "web_name": "O'Reilly", "first_name": "Nico", "second_name": "O'Reilly", "element_type": 2, "status": "a"},
                {"id": 155, "team": 15, "web_name": "Enzo", "first_name": "Enzo", "second_name": "Fernández", "element_type": 3, "status": "a"},
                {"id": 481, "team": 15, "web_name": "Anderson", "first_name": "Elliot", "second_name": "Anderson", "element_type": 3, "status": "a"},
                {"id": 616, "team": 15, "web_name": "Bouaddi", "first_name": "Ayyoub", "second_name": "Bouaddi", "element_type": 3, "status": "a"},
                {"id": 237, "team": 15, "web_name": "Ndiaye", "first_name": "Iliman", "second_name": "Ndiaye", "element_type": 3, "status": "a"},
                {"id": 399, "team": 15, "web_name": "Cherki", "first_name": "Rayan", "second_name": "Cherki", "element_type": 3, "status": "a"},
                {"id": 400, "team": 15, "web_name": "Doku", "first_name": "Jérémy", "second_name": "Doku", "element_type": 3, "status": "a"},
                {"id": 397, "team": 15, "web_name": "Semenyo", "first_name": "Antoine", "second_name": "Semenyo", "element_type": 3, "status": "a"},
                {"id": 398, "team": 15, "web_name": "Foden", "first_name": "Phil", "second_name": "Foden", "element_type": 3, "status": "a"},
                {"id": 411, "team": 15, "web_name": "Haaland", "first_name": "Erling", "second_name": "Haaland", "element_type": 4, "status": "a"},
            ],
        }
        men = np.validate_packet(
            _packet(
                source="manchestereveningnews",
                packet_id="manchestereveningnews:gw07:399-cherki-will-start",
                url="https://www.manchestereveningnews.co.uk/sport/football/football-news/man-city-line-up-vs-34776100",
                published_at_utc="2026-10-17T03:30:00Z",
                club="Man City",
                player_ids=[399],
                headline="Cherki will start vs Ipswich in predicted XI",
                body=(
                    "City predicted XI: Donnarumma; Nunes, Khusanov, Guehi, Gvardiol; "
                    "Fernandez, Anderson; Ndiaye, Cherki, Doku; Haaland."
                ),
            ),
            deadline_utc=deadline,
        )
        standard = np.validate_packet(
            _packet(
                source="the_standard",
                packet_id="the_standard:gw07:399-cherki-left-out",
                url="https://www.standard.co.uk/sport/football/man-city-xi-vs-ipswich-confirmed-team-news-predicted-lineup-injury-latest-premier-league-b1301072.html",
                published_at_utc="2026-10-17T06:40:00Z",
                club="Man City",
                player_ids=[399],
                headline="Cherki left out of predicted starting XI",
                body=(
                    "Cherki left out of the predicted starting XI (Foden preferred). "
                    "Predicted XI: Donnarumma; Nunes, Dias, Guehi, O'Reilly; "
                    "Bouaddi, Fernandez; Semenyo, Foden, Ndiaye; Haaland."
                ),
            ),
            deadline_utc=deadline,
        )
        roles = np.predicted_lineup_roles([men, standard], bootstrap)
        self.assertEqual(roles[391][0], "rotated")  # Gvardiol
        self.assertEqual(roles[388][0], "xi")  # Guéhi
        self.assertEqual(roles[387][0], "xi")  # O'Reilly
        self.assertEqual(roles[393][0], "rotated")  # Khusanov only in earlier XI

        gvard = np.compile_player_xmi(
            player_id=391,
            name="Gvardiol",
            position="DEF",
            prior=68.0,
            status="a",
            chance=None,
            packets=[],
            minutes=[90.0, 90.0, 24.0],
            lineup_role=roles[391][0],
            lineup_packet_id=roles[391][1],
        )
        self.assertEqual(gvard["tag"], "benched")
        self.assertEqual(gvard["xmi_source"], "lineup:rotated")
        self.assertAlmostEqual(float(gvard["xmi_compiled"]), 15.0)

        # Cherki: SUPPORT "left out" on player_ids packet beats any lineup role.
        cherki = np.compile_player_xmi(
            player_id=399,
            name="Cherki",
            position="MID",
            prior=45.0,
            status="a",
            chance=None,
            packets=[standard],
            minutes=[60.0, 70.0, 45.0],
            lineup_role="rotated",
            lineup_packet_id=roles.get(399, (None, None))[1],
        )
        self.assertEqual(cherki["tag"], "benched")
        self.assertEqual(cherki["xmi_source"], "override:benched")


if __name__ == "__main__":
    unittest.main()
