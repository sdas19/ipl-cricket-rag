"""Tests for chunker.py on a tiny hand-made match in Cricsheet's format.
Run:  pytest -q"""

from chunker import chunk_match, describe_ball, phase_of


def ball(batter, bowler, bat=0, extras=None, wicket=None, non_striker="X Partner"):
    extras = extras or {}
    d = {"batter": batter, "bowler": bowler, "non_striker": non_striker,
         "runs": {"batter": bat, "extras": sum(extras.values()), "total": bat + sum(extras.values())}}
    if extras:
        d["extras"] = extras
    if wicket:
        d["wickets"] = [wicket]
    return d


MATCH = {
    "info": {
        "dates": ["2025-06-03"], "season": "2025", "match_type": "T20",
        "teams": ["Royal Challengers Bengaluru", "Punjab Kings"],
        "venue": "Narendra Modi Stadium, Ahmedabad",
        "event": {"name": "Indian Premier League", "stage": "Final"},
        "toss": {"winner": "Punjab Kings", "decision": "field"},
        "outcome": {"winner": "Royal Challengers Bengaluru", "by": {"runs": 6}},
        "player_of_match": ["KH Pandya"],
        "registry": {"people": {"V Kohli": "ba607b88", "JJ Bumrah": "462411b3"}},
    },
    "innings": [
        {"team": "Royal Challengers Bengaluru", "overs": [
            {"over": 0, "deliveries": [
                ball("V Kohli", "JJ Bumrah", 4),
                ball("V Kohli", "JJ Bumrah", 0, extras={"wides": 1}),
                ball("V Kohli", "JJ Bumrah", 6),
                ball("V Kohli", "JJ Bumrah", 0,
                     wicket={"player_out": "V Kohli", "kind": "caught", "fielders": [{"name": "SS Iyer"}]}),
                ball("RM Patidar", "JJ Bumrah", 1),
                ball("X Partner", "JJ Bumrah", 0),
                ball("X Partner", "JJ Bumrah", 0, extras={"legbyes": 2}),
            ]},
        ]},
        {"team": "Punjab Kings", "target": {"overs": 20, "runs": 14}, "overs": [
            {"over": 17, "deliveries": [ball("SS Iyer", "B Kumar", 1) for _ in range(6)]},
        ]},
        {"team": "Punjab Kings", "super_over": True, "overs": []},
    ],
}


def chunks():
    return chunk_match(MATCH, "1473511")


def test_counts_and_levels():
    cs = chunks()
    assert [c["level"] for c in cs] == ["over", "over", "match"]   # super over skipped
    assert [c["innings"] for c in cs] == [1, 2, 0]


def test_over_header_and_ball_labels():
    c = chunks()[0]
    lines = c["content"].splitlines()
    assert lines[0] == ("Indian Premier League 2025 · Final · Royal Challengers Bengaluru vs Punjab Kings"
                        " · Narendra Modi Stadium, Ahmedabad · 2025-06-03")
    assert "Innings 1: Royal Challengers Bengaluru batting · Over 1 (powerplay)" in lines[1]
    assert lines[3] == "0.1 JJ Bumrah to V Kohli: FOUR"
    assert lines[4] == "0.2 JJ Bumrah to V Kohli: wide (+1)"        # a wide doesn't use up a ball
    assert lines[5] == "0.2 JJ Bumrah to V Kohli: SIX"
    assert "OUT! V Kohli c SS Iyer b JJ Bumrah" in lines[6]
    assert lines[-1] == "End of over 1: Royal Challengers Bengaluru 14/1 (14 runs, 1 wicket in the over)"


def test_labels_for_filters_and_evals():
    c = chunks()[0]
    assert (c["match_id"], c["innings"], c["over_start"], c["phase"]) == ("1473511", 1, 1, "powerplay")
    assert "SS Iyer" in c["players"]                                  # fielder counts as involved
    assert c["player_ids"] == ["462411b3", "ba607b88"]                # only registry names get ids


def test_chase_situation_and_death_phase():
    c = chunks()[1]
    assert "Over 18 (death)" in c["content"]
    assert "chasing 14 · Score at start of over: 0/0, need 14 off 120 balls" in c["content"]


def test_match_chunk_summary():
    m = chunks()[-1]["content"]
    assert "Result: Royal Challengers Bengaluru won by 6 runs. Player of the match: KH Pandya." in m
    assert "Toss: Punjab Kings won the toss and chose to field." in m
    assert "Innings 1: Royal Challengers Bengaluru 14/1 in 1 overs." in m
    assert "V Kohli 10 (3 balls, 1x4, 1x6)" in m                    # wide not counted as a ball faced
    assert "JJ Bumrah 1/12" in m                                      # leg byes not charged to bowler


def test_helpers():
    assert phase_of(6, "T20") == "powerplay" and phase_of(7, "T20") == "middle"
    assert phase_of(16, "T20") == "death" and phase_of(16, "ODI") is None
    assert describe_ball(ball("A", "B")) == "no run"
    assert describe_ball(ball("A", "B", 4) | {"runs": {"batter": 4, "extras": 0, "total": 4, "non_boundary": True}}) == "4 runs"
