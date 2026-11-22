"""Tests for Lesson 3c: search helpers and the eval's scoring. No network, no database.
Run:  python -m pytest -q"""

import csv

from eval import facts_ok, report, score
from rewrite import Plan
from search import keyword_query, rerank, rrf


# ---- search helpers ------------------------------------------------------------

def test_keyword_query_drops_filler_and_uses_or():
    assert keyword_query("How did Virat Kohli get out in the final?") == "virat or kohli or get or out or final"
    assert keyword_query("Shreyas Iyer final me kaise out hua?") == "shreyas or iyer or final or out or hua"
    assert keyword_query("how did the") == ""


def test_keyword_query_expands_to_the_datas_words():
    assert keyword_query("How was Rohit Sharma dismissed?") == "rohit or sharma or out"
    assert keyword_query("sabse zyada chhakke") == "sabse or zyada or six"


def row(over, content="x"):
    return ("1473511", "over", 2, over, over, content)


def test_rrf_rewards_rows_both_searches_like():
    vector = [row(13), row(15), row(11)]          # meaning: over 15 second
    keyword = [row(15), row(7)]                   # words: over 15 first
    fused, scores = rrf(vector, keyword)
    assert [r[3] for r in fused][:2] == [15, 13]  # in both lists -> top
    assert abs(scores[0] - (1 / 62 + 1 / 61)) < 1e-12


def test_rerank_reorders_by_the_models_scores():
    sent = {}

    def fake_post(url, payload):
        sent.update(url=url, payload=payload)
        return {"results": [{"index": 1, "relevance_score": 0.91}, {"index": 0, "relevance_score": 0.12}],
                "usage": {"cost": 0.000002}}

    rows, scores, cost = rerank("how did Kohli get out", [row(13, "singles"), row(15, "OUT! V Kohli")],
                                2, "cohere/rerank-v3.5", "k", "https://openrouter.ai/api/v1", post=fake_post)
    assert [r[3] for r in rows] == [15, 13] and scores == [0.91, 0.12] and cost == 0.000002
    assert sent["url"].endswith("/rerank") and sent["payload"]["documents"] == ["singles", "OUT! V Kohli"]
    assert rerank("q", [], 5, "m", "k", "u", post=fake_post) == ([], [], 0.0)


# ---- eval scoring --------------------------------------------------------------

def test_facts_with_alternatives():
    assert facts_ok("Royal Challengers Bengaluru won by 6 runs", "Royal Challengers|RCB;6")
    assert facts_ok("RCB jeeta, 6 run se", "Royal Challengers|RCB;6")
    assert not facts_ok("Punjab won", "Royal Challengers|RCB;6")
    assert facts_ok("Pooran hit 40 sixes", "Pooran;40") and not facts_ok("Pooran hit 4 sixes", "Pooran;40")


def case(**kw):
    base = {"id": "t", "type": "last_over", "language": "english", "expected_route": "lookup",
            "expected_match": "1473510", "expected_over": "last", "expected_players": "", "expected_stat": "",
            "expected_keys": "1473510 | 2 | 19-19", "must_contain": "Ashwani;26", "should_refuse": "no"}
    base.update(kw)
    return base


def output(route="lookup", answer="Ashwani Kumar bowled it, 26 runs [S1]", rows=None, cited=None, idk=False,
           match="1473510", **plan):
    rows = rows if rows is not None else [("1473510", "over", 2, 19, 19, "..."), ("1473510", "match", 0, 0, 0, "...")]
    result = None if route == "stop" else {
        "said_idk": idk, "cited": cited if cited is not None else [("S1", "1473510 | 2 | 19-19")],
        "invented_citations": [], "truncated": False}
    return {"plan": Plan(route="lookup", search_text="q", over="last", **plan), "route": route, "answer": answer,
            "decision": {"rows": rows, "match_id": match, "players": []}, "result": result,
            "cost_usd": 0.0005, "total_ms": 5000.0}


def test_a_fully_correct_run():
    s = score(case(), output())
    assert s["route_ok"] and s["match_ok"] and s["over_ok"] and s["hit1"] and s["rr"] == 1.0
    assert s["facts_ok"] and s["cited_ok"] and s["answer_ok"] and not s["refused"]


def test_failures_land_in_the_right_layer():
    wrong_match = score(case(), output(match="1473508", rows=[("1473508", "over", 2, 10, 10, "...")],
                                       answer="Musheer Khan, 14 runs", cited=[("S1", "1473508 | 2 | 10-10")]))
    assert wrong_match["match_ok"] is False and wrong_match["hit5"] is False and wrong_match["answer_ok"] is False
    refused = score(case(), output(answer="I don't know based on the available match data.", idk=True, cited=[]))
    assert refused["route_ok"] and refused["hit1"] and refused["answer_ok"] is False   # retrieval fine, answer step failed


def test_refusals_score_correctly():
    stop = case(expected_route="stop", expected_match="", expected_over="", expected_keys="",
                must_contain="isn't in our", should_refuse="yes")
    assert score(stop, output(route="stop", answer="Ashok Dinda isn't in our IPL 2025 data."))["answer_ok"]
    assert not score(stop, output(route="search", answer="Dinda bowled well [S1]"))["answer_ok"]


def test_report_and_golden_file_shape():
    scores = [score(case(), output()), score(case(id="t2"), output(match="x", answer="no"))]
    text = report(scores, "hybrid_rerank")
    assert "route" in text and "hit@1" in text and "FAILURES" in text and "t2" in text
    rows = list(csv.DictReader(open("golden.csv", encoding="utf-8")))
    assert len(rows) == 36 and len({r["id"] for r in rows}) == 36
    for r in rows:
        assert r["should_refuse"] in ("yes", "no") and r["language"] in ("english", "hinglish")
        assert set(r["expected_route"].split("|")) <= {"", "lookup", "stats", "search", "stop"}
