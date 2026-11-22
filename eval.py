"""
Evaluate the whole system on the golden set, layer by layer and by question type.

  python eval.py                                   # all 36 questions, SEARCH_MODE from .env
  python eval.py --search-mode vector --out vector.json
  python eval.py --only l01,l02 --repeat 3         # a few questions, 3 runs each
  python eval.py --types dismissal,descriptive

Each layer gets its own score, so a failure points at its cause:
  routing   right route? right match / over / players / stat?
  retrieval (lookup + search) hit@1, hit@5, MRR on the expected chunk keys
  answer    expected facts present? right chunk cited? refused when (and only when) it should?
  cost / latency per question, p50 and p95
"""

import argparse
import csv
import json
import math
import time


def percentile(values, q):
    """Nearest-rank percentile (same as Build 1): q=0.95 -> the value 95% of runs were at or under."""
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return values[max(0, math.ceil(q * len(values)) - 1)]


def split(cell):
    return [x.strip() for x in (cell or "").split(";") if x.strip()]


def norm(text):
    return " ".join((text or "").lower().replace(",", "").split())


def facts_ok(answer, must_contain):
    """Every fact must appear; a fact may list alternatives with '|' ('Royal Challengers|RCB')."""
    a = norm(answer)
    return all(any(norm(alt) in a for alt in fact.split("|")) for fact in split(must_contain))


def key_of(row):
    from answer import key_of as k
    return k(row)


def score(case, out):
    """Compare one run of the system with one golden row. Values: True / False / None (not applicable)."""
    plan, d, r = out["plan"], out["decision"], out.get("result")
    route = out["route"]
    s = {"id": case["id"], "type": case["type"], "language": case["language"], "route": route}

    # --- routing layer
    s["route_ok"] = route in case["expected_route"].split("|") if case["expected_route"] else None
    s["match_ok"] = (d.get("match_id") == case["expected_match"]) if case["expected_match"] else None
    s["over_ok"] = (str(plan.over) == case["expected_over"]) if case["expected_over"] else None
    s["stat_ok"] = (plan.stat in case["expected_stat"].split("|")) if case["expected_stat"] else None
    exp_players = split(case["expected_players"])
    s["players_ok"] = set(exp_players) <= set(d.get("players") or []) if exp_players else None

    # --- retrieval layer (only where sources are chunks with known keys)
    exp_keys = set(split(case["expected_keys"]))
    if exp_keys and route in ("lookup", "search"):
        keys = [key_of(row) for row in d.get("rows", [])]
        ranks = [i for i, k in enumerate(keys, start=1) if k in exp_keys]
        s["hit1"] = bool(ranks) and ranks[0] == 1
        s["hit5"] = bool(ranks) and ranks[0] <= 5
        s["rr"] = 1 / ranks[0] if ranks else 0.0
    else:
        s["hit1"] = s["hit5"] = s["rr"] = None

    # --- answer layer
    refused = route == "stop" or bool(r and r["said_idk"])
    should_refuse = case["should_refuse"] == "yes"
    s["refused"] = refused
    s["facts_ok"] = facts_ok(out["answer"], case["must_contain"]) if case["must_contain"] else None
    if should_refuse:
        s["answer_ok"] = refused and s["facts_ok"] is not False
    else:
        s["answer_ok"] = (not refused) and s["facts_ok"] is not False
    s["cited_ok"] = (any(k in exp_keys for _, k in r["cited"]) if (r and exp_keys and route != "stats") else None)
    s["invented_citations"] = bool(r and r["invented_citations"])
    s["truncated"] = bool(r and r["truncated"])

    s["cost_usd"] = out["cost_usd"]
    s["total_ms"] = out["total_ms"]
    s["answer"] = out["answer"]
    s["plan"] = plan.model_dump(exclude_none=True)
    return s


def rate(values):
    vals = [v for v in values if v is not None]
    return (sum(bool(v) for v in vals) / len(vals), len(vals)) if vals else (None, 0)


def report(scores, mode):
    pct = lambda x: "  n/a" if x[0] is None else f"{x[0] * 100:5.1f}% ({x[1]})"
    col = lambda key: [s[key] for s in scores]
    lines = [f"=== Eval: {len(scores)} runs, search mode {mode} ===", "",
             "ROUTING", f"  route        {pct(rate(col('route_ok')))}",
             f"  match        {pct(rate(col('match_ok')))}", f"  over         {pct(rate(col('over_ok')))}",
             f"  players      {pct(rate(col('players_ok')))}", f"  stat         {pct(rate(col('stat_ok')))}",
             "RETRIEVAL (lookup + search)", f"  hit@1        {pct(rate(col('hit1')))}",
             f"  hit@5        {pct(rate(col('hit5')))}"]
    rrs = [v for v in col("rr") if v is not None]
    lines.append(f"  MRR          {sum(rrs) / len(rrs):.2f}" if rrs else "  MRR            n/a")
    lines += ["ANSWERS", f"  correct      {pct(rate(col('answer_ok')))}",
              f"  cited right  {pct(rate(col('cited_ok')))}",
              f"  invented ids {sum(col('invented_citations'))}   cut off {sum(col('truncated'))}",
              "COST AND TIME",
              f"  ${sum(col('cost_usd')):.4f} total · ${sum(col('cost_usd')) / len(scores):.6f} per question",
              f"  p50 {percentile(col('total_ms'), .5) / 1000:.1f} s · p95 {percentile(col('total_ms'), .95) / 1000:.1f} s",
              "", f"{'type':<18}{'n':>3}{'route':>8}{'hit@5':>8}{'answer':>8}"]
    for t in dict.fromkeys(s["type"] for s in scores):
        g = [s for s in scores if s["type"] == t]
        f = lambda key: rate([s[key] for s in g])[0]
        cell = lambda x: "    -" if x is None else f"{x * 100:5.0f}%"
        lines.append(f"{t:<18}{len(g):>3}{cell(f('route_ok')):>8}{cell(f('hit5')):>8}"
                     f"{cell(f('answer_ok')):>8}")
    by_lang = {l: rate([s["answer_ok"] for s in scores if s["language"] == l]) for l in ("english", "hinglish")}
    lines += ["", "answers by language: " + ", ".join(f"{l} {pct(v)}" for l, v in by_lang.items())]

    failed = [s for s in scores if s["answer_ok"] is False or s["route_ok"] is False]
    if failed:
        lines += ["", "FAILURES"]
        for s in failed:
            why = [k for k in ("route_ok", "match_ok", "over_ok", "players_ok", "stat_ok", "hit5", "facts_ok",
                               "answer_ok") if s[k] is False]
            lines.append(f"  {s['id']} [{', '.join(why)}] route={s['route']} -> {s['answer'][:110]!r}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="golden.csv")
    ap.add_argument("--search-mode", default=None)
    ap.add_argument("--only", help="comma-separated ids")
    ap.add_argument("--types", help="comma-separated question types")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", help="save every run as JSON")
    args = ap.parse_args()

    from common import SEARCH_MODE, connect, get_chat_model
    from router import answer_question

    mode = args.search_mode or SEARCH_MODE
    cases = list(csv.DictReader(open(args.golden, encoding="utf-8")))
    if args.only:
        cases = [c for c in cases if c["id"] in args.only.split(",")]
    if args.types:
        cases = [c for c in cases if c["type"] in args.types.split(",")]

    model, scores = get_chat_model(), []
    with connect() as conn:
        for rep in range(args.repeat):
            for i, case in enumerate(cases, start=1):
                t0 = time.perf_counter()
                try:
                    s = score(case, answer_question(case["question"], conn, model, mode))
                except Exception as e:                      # one failure shouldn't stop the run
                    s = {"id": case["id"], "type": case["type"], "language": case["language"], "route": "error",
                         "answer": f"{type(e).__name__}: {e}"[:300], "cost_usd": 0.0,
                         "total_ms": (time.perf_counter() - t0) * 1000, "plan": {},
                         **{k: (False if k in ("route_ok", "answer_ok") else None) for k in
                            ("route_ok", "match_ok", "over_ok", "stat_ok", "players_ok", "hit1", "hit5", "rr",
                             "facts_ok", "answer_ok", "cited_ok")},
                         "refused": False, "invented_citations": False, "truncated": False}
                scores.append(s)
                mark = "ok " if s["answer_ok"] else "FAIL"
                print(f"[{rep + 1}/{args.repeat} {i:>2}/{len(cases)}] {mark} {case['id']} "
                      f"{s['route']:<7} {s['total_ms'] / 1000:4.1f}s  {case['question'][:60]}")

    print("\n" + report(scores, mode))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"mode": mode, "runs": scores}, f, indent=1, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
