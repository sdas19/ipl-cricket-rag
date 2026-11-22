"""Tests for Lesson 3b: ingest_deliveries, resolve, stats, rewrite and the router.
No database and no API calls: fakes stand in for both.  Run:  python -m pytest -q"""

import pytest
from langchain_core.messages import AIMessage

import stats
from ingest_deliveries import number_playoffs, parse_match
from resolve import resolve_match, resolve_player, resolve_team
from rewrite import Plan, parse_plan, rewrite
from router import decide
from test_chunker import MATCH

KNOWN = ["V Kohli", "RG Sharma", "Abhishek Sharma", "JM Sharma", "JJ Bumrah", "SS Iyer", "VR Iyer",
         "Shashank Singh", "N Pooran", "MS Dhoni", "SA Yadav", "JR Hazlewood"]
MATCHES = [
    {"match_id": "m38", "season": "2025", "match_date": "2025-04-20", "match_number": 38, "stage": "League",
     "team1": "Chennai Super Kings", "team2": "Mumbai Indians"},
    {"match_id": "m03", "season": "2025", "match_date": "2025-03-23", "match_number": 3, "stage": "League",
     "team1": "Chennai Super Kings", "team2": "Mumbai Indians"},
    {"match_id": "1473511", "season": "2025", "match_date": "2025-06-03", "match_number": 74, "stage": "Final",
     "team1": "Royal Challengers Bengaluru", "team2": "Punjab Kings"},
]


# ---- ingest_deliveries ---------------------------------------------------------

def test_parse_match_balls():
    match, balls, players = parse_match(MATCH, "1473511")
    assert match["stage"] == "Final" and match["result"] == "Royal Challengers Bengaluru won by 6 runs"
    assert len(balls) == 13                                   # 7 + 6; the super over is skipped
    wide = balls[1]
    assert wide["is_wide"] and not wide["is_legal"] and wide["ball_label"] == "0.2"
    legbye = balls[6]
    assert legbye["total_runs"] == 2 and legbye["bowler_runs"] == 0   # leg byes aren't the bowler's
    out = balls[3]
    assert out["is_wicket"] and out["bowler_wicket"] and out["fielder"] == "SS Iyer"
    assert balls[7]["over_number"] == 18 and balls[7]["phase"] == "death"
    assert balls[7]["batting_team"] == "Punjab Kings" and balls[7]["bowling_team"] == "Royal Challengers Bengaluru"


def test_playoffs_numbered_after_league():
    ms = [{"match_number": 70, "match_date": "2025-05-27"},
          {"match_number": None, "match_date": "2025-06-03"},
          {"match_number": None, "match_date": "2025-05-29"}]
    number_playoffs(ms)
    assert [m["match_number"] for m in ms] == [70, 72, 71]


# ---- resolve ---------------------------------------------------------------

def test_teams():
    assert resolve_team("RCB") == resolve_team("bangalore") == "Royal Challengers Bengaluru"
    assert resolve_team("kxip") == "Punjab Kings" and resolve_team("Atlantis") is None


@pytest.mark.parametrize("typed,expected", [
    ("Virat Kohli", "V Kohli"), ("kohli", "V Kohli"), ("King Kohli", "V Kohli"),
    ("Rohit Sharma", "RG Sharma"), ("Jasprit Bumrah", "JJ Bumrah"), ("Shashank", "Shashank Singh"),
    ("SKY", "SA Yadav"), ("Shreyas Iyer", "SS Iyer"),
])
def test_player_names(typed, expected):
    assert resolve_player(typed, KNOWN)[0] == expected


def test_unknown_and_ambiguous_players():
    assert resolve_player("Ashok Dinda", KNOWN) == (None, [])                 # not in our data
    name, cands = resolve_player("Iyer", KNOWN)
    assert name is None and cands == ["SS Iyer", "VR Iyer"]                    # ask which one


def test_match_resolution():
    assert resolve_match(MATCHES, "2025", stage="Final")[0]["match_id"] == "1473511"
    assert resolve_match(MATCHES, "2025", match_number=3)[0]["match_id"] == "m03"
    m, note = resolve_match(MATCHES, "2025", teams=["Mumbai Indians", "Chennai Super Kings"])
    assert m["match_id"] == "m38" and "using the latest" in note            # two fit: latest, and say so
    assert resolve_match(MATCHES, "2025", teams=["Mumbai Indians"]) == (None, None)  # one team = a team question
    assert resolve_match(MATCHES, "2025") == (None, None)                    # season-wide
    assert resolve_match(MATCHES, "2025", stage="Eliminator")[0] is None      # nothing fits


# ---- stats -----------------------------------------------------------------

def test_stats_templates_and_safety():
    title, sql, params = stats.build_query("most_sixes", season="2025", phase="death", team="Mumbai Indians")
    assert "is_six" in sql and "batting_team = %(team)s" in sql and params["phase"] == "death"
    _, sql, params = stats.build_query("best_economy", season="2025")
    assert "having" in sql and params["min_balls"] == 60
    _, sql, params = stats.build_query("player_batting", season="2025", player="O'Brien; drop table x")
    assert "O'Brien" not in sql and params["player"] == "O'Brien; drop table x"   # values never pasted into SQL
    with pytest.raises(ValueError):
        stats.build_query("drop_tables", season="2025")
    with pytest.raises(ValueError):
        stats.build_query("player_bowling", season="2025")                       # needs a player


def test_stats_source_says_it_is_complete():
    text = stats.as_source("Most sixes", ["player", "value"], [("N Pooran", 40), ("SS Iyer", 39)],
                           {"season": "2025", "phase": None})
    assert text.splitlines() == ["Computed from all ball-by-ball data (season=2025): Most sixes",
                                 "1. N Pooran: 40", "2. SS Iyer: 39"]


# ---- rewrite ---------------------------------------------------------------

class FakeModel:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=self.replies.pop(0),
                         usage_metadata={"input_tokens": 700, "output_tokens": 60, "total_tokens": 760})


GOOD = ('Here is the plan: {"route": "lookup", "stage": "Final", "innings": 2, "over": 20, '
        '"search_text": "Who bowled the last over of the final?", "language": "hinglish"}')


def test_parse_plan_from_prose():
    plan = parse_plan(GOOD)
    assert (plan.route, plan.stage, plan.innings, plan.over, plan.season) == ("lookup", "Final", 2, 20, "2025")


def test_rewrite_retries_once_with_the_error():
    model = FakeModel(['{"route": "lookup", "over": 99, "search_text": "x"}', GOOD])  # over 99 is invalid
    plan, meta = rewrite("final me last over?", model, "anthropic/x")
    assert plan.over == 20 and model.calls == 2 and meta["input_tokens"] == 1400


def test_rewrite_gives_up_after_two_bad_replies():
    with pytest.raises(Exception):
        rewrite("?", FakeModel(["not json", "still not json"]), "anthropic/x")


# ---- router ----------------------------------------------------------------

class FakeDB:
    def __init__(self):
        self.searches = []

    def known_players(self, season):
        return KNOWN

    def matches(self, season):
        return MATCHES

    def lookup(self, match_id, innings=None, over=None):
        if match_id == "m38" and over == "last":            # a chase that ended in over 19
            return [("m38", "over", innings or 2, 19, 19, "Over 19 ... winning runs ...")]
        if match_id == "m38" and over is None:
            return [("m38", "match", 0, 0, 0, "Result ... Innings 2: 180/4 in 19 overs")]
        if match_id == "1473511" and over == 20:
            return [("1473511", "over", innings or 2, 20, 20, "Over 20 ... JR Hazlewood ...")]
        return []

    def search(self, text, match_id=None, players=(), k=5):
        self.searches.append((match_id, tuple(players)))
        return [] if players == ["Nobody"] else [("m38", "over", 1, 5, 5, "some over")]

    def stats(self, stat, **filters):
        return "Most sixes", ["player", "value"], [("N Pooran", 40)], "select ...", filters


def plan(**kw):
    return Plan(search_text="q", **kw)


def test_unknown_player_stops_before_search():
    db = FakeDB()
    d = decide(plan(route="search", players=["Ashok Dinda"]), db)
    assert "isn't in our IPL 2025 data" in d["stop"] and db.searches == []


def test_ambiguous_player_asks():
    assert decide(plan(route="stats", stat="player_batting", players=["Iyer"]), FakeDB())["stop"].startswith(
        "Which player do you mean: SS Iyer, VR Iyer")


def test_lookup_route_fetches_the_exact_over():
    d = decide(plan(route="lookup", stage="Final", innings=2, over=20), FakeDB())
    assert d["route"] == "lookup" and d["rows"][0][:4] == ("1473511", "over", 2, 20)


def test_stats_route_builds_one_computed_source():
    d = decide(plan(route="stats", stat="most_sixes", teams=["RCB"]), FakeDB())
    label, level, *_ , content = d["rows"][0]
    assert d["route"] == "stats" and level == "stats"
    assert "team=Royal Challengers Bengaluru" in label and content.startswith("Computed from all ball-by-ball data")


def test_lookup_without_rows_falls_back_to_search():
    db = FakeDB()
    d = decide(plan(route="lookup", stage="Final", over=7), db)
    assert d["route"] == "search" and db.searches == [("1473511", ())]


def test_out_of_scope_and_unknown_stat():
    assert "only answer questions about IPL" in decide(plan(route="out_of_scope"), FakeDB())["stop"]
    d = decide(plan(route="stats", stat="most_dropped_catches"), FakeDB())
    assert d["route"] == "search" and any("unknown stat" in n for n in d["notes"])


def test_last_over_is_a_position_not_20():
    p = parse_plan('{"route": "lookup", "stage": "Final", "over": "last", "search_text": "q"}')
    assert p.over == "last" and p.innings is None
    assert parse_plan('{"route": "lookup", "over": 20, "search_text": "q"}').over == 20
    for bad in ['"first"', "25"]:
        with pytest.raises(Exception):
            parse_plan('{"route": "lookup", "over": ' + bad + ', "search_text": "q"}')


def test_last_over_resolved_from_data():
    d = decide(plan(route="lookup", match_number=38, over="last"), FakeDB())
    assert d["rows"][0][2:4] == (2, 19)                      # innings 2 ended in over 19, not 20
    assert d["rows"][1][1] == "match"                        # plus the summary that proves it ended there
    assert any("innings 2, over 19" in n for n in d["notes"])


def test_exact_over_beats_the_search_label():
    # "Qualifier 1 ka aakhri over kaisa raha?": the rewriter said search, but named a match and an over
    db = FakeDB()
    d = decide(plan(route="search", match_number=38, over="last"), db)
    assert d["route"] == "lookup" and d["rows"][0][2:4] == (2, 19) and db.searches == []
    assert any("using lookup" in n for n in d["notes"])


def test_stats_scope_names_the_match_in_words():
    d = decide(plan(route="stats", stat="most_sixes", stage="Final"), FakeDB())
    label, content = d["rows"][0][0], d["rows"][0][5]
    assert "Final (Match 74), Royal Challengers Bengaluru vs Punjab Kings, 2025-06-03" in content
    assert "1473511" not in content and "Final 74" in label


def test_named_team_picks_its_batting_innings():
    class TwoInnings(FakeDB):
        def lookup(self, match_id, innings=None, over=None):
            return [("1473511", "over", 1, 17, 17, "Innings 1: Royal Challengers Bengaluru batting · Over 17"),
                    ("1473511", "over", 2, 17, 17, "Innings 2: Punjab Kings batting · Over 17")]
    d = decide(plan(route="lookup", stage="Final", over=17, teams=["Punjab"]), TwoInnings())
    assert [r[2] for r in d["rows"]] == [2] and any("Punjab Kings batted" in n for n in d["notes"])
