"""
The router: one question in, the right tool for it, one grounded answer out.

  1. rewrite   cheap model call -> Plan (route, filters, English search text)
  2. names     check players/teams against our data   (stop if unknown or ambiguous)
  3. match     "the final" / "match 57" / two teams -> one match_id
  4. route     lookup -> fetch exact chunks by label columns (no search)
               stats  -> fixed SQL templates over every ball (no search)
               search -> vector / keyword / hybrid (+ rerank) search inside the filters
               out_of_scope -> stop
  5. answer    the existing answer step (citations, "I don't know" rules)

  python router.py "final me last over kisne daala aur kitne run bane?"
  python router.py "Who hit the most sixes in the 2025 season?" --show-plan
"""

import argparse
import json
import time

import search as search_modes
import stats
from resolve import resolve_match, resolve_player, resolve_team

COLUMNS = "match_id, level, innings, over_start, over_end, content"


class Database:
    """Everything the router reads, behind small methods (so tests can use a fake)."""

    def __init__(self, conn, search_mode="vector"):
        self.conn = conn
        self.search_mode = search_mode
        self.last_search = {}          # what the last search did: for printing and evals

    def known_players(self, season):
        return [r[0] for r in self.conn.execute(
            "select name from players where season = %s", (season,))]

    def matches(self, season):
        cols = ["match_id", "season", "match_date", "match_number", "stage", "team1", "team2"]
        rows = self.conn.execute(f"select {', '.join(cols)} from matches where season = %s", (season,))
        return [dict(zip(cols, r)) for r in rows]

    def lookup(self, match_id, innings=None, over=None):
        """Fetch chunks by their labels: an exact address, no search involved."""
        if over == "last":
            # The highest over actually bowled: in that innings, or in the match
            # (the last innings) when no innings is given.
            sql = (f"select {COLUMNS} from chunks where match_id = %s and level = 'over' "
                   "and (%s::int is null or innings = %s) order by innings desc, over_start desc limit 1")
            return self.conn.execute(sql, (match_id, innings, innings)).fetchall()
        if over:
            sql = (f"select {COLUMNS} from chunks where match_id = %s and level = 'over' "
                   "and over_start = %s and (%s::int is null or innings = %s) order by innings")
            return self.conn.execute(sql, (match_id, over, innings, innings)).fetchall()
        return self.conn.execute(
            f"select {COLUMNS} from chunks where match_id = %s and level = 'match'", (match_id,)).fetchall()

    def _vector(self, text, match_id, players, n):
        from ask import embed_question
        from common import EMBED_MODEL, to_pgvector

        vec, _ = embed_question(text)
        sql = (f"select {COLUMNS} from chunks where embed_model = %(model)s "
               "and (%(match)s::text is null or match_id = %(match)s) "
               "and players @> %(players)s::text[] "
               "order by embedding <=> %(q)s::vector limit %(n)s")
        return self.conn.execute(sql, {"model": EMBED_MODEL, "match": match_id, "players": list(players),
                                       "q": to_pgvector(vec), "n": n}).fetchall()

    def _keyword(self, text, match_id, players, n):
        query = search_modes.keyword_query(text)
        if not query:
            return []
        sql = (f"select {COLUMNS} from chunks, websearch_to_tsquery('simple', %(kw)s) q "
               "where tsv @@ q and (%(match)s::text is null or match_id = %(match)s) "
               "and players @> %(players)s::text[] "
               "order by ts_rank(tsv, q) desc limit %(n)s")
        return self.conn.execute(sql, {"kw": query, "match": match_id, "players": list(players),
                                       "n": n}).fetchall()

    def search(self, text, match_id=None, players=(), k=5):
        """Search inside the filters (match and/or players), in self.search_mode."""
        mode, t0 = self.search_mode, time.perf_counter()
        info = {"mode": mode, "rerank_cost": 0.0}
        if mode == "vector":
            rows = self._vector(text, match_id, players, k)
        elif mode == "keyword":
            rows = self._keyword(text, match_id, players, k)
        else:
            # Two lists of candidates, merged: rows liked by BOTH searches rise to the top.
            vec = self._vector(text, match_id, players, 30)
            kw = self._keyword(text, match_id, players, 30)
            fused, _ = search_modes.rrf(vec, kw)
            info.update(vector_hits=len(vec), keyword_hits=len(kw))
            rows = fused[:k]
            if mode == "hybrid_rerank" and fused:
                from common import OPENROUTER_BASE_URL, RERANK_MODEL, setting
                rows, scores, cost = search_modes.rerank(text, fused[:30], k, RERANK_MODEL,
                                                         setting("OPENROUTER_API_KEY"), OPENROUTER_BASE_URL)
                info.update(rerank_cost=cost, rerank_scores=[round(x, 3) for x in scores])
        info["ms"] = (time.perf_counter() - t0) * 1000
        self.last_search = info
        return rows

    def stats(self, stat, **filters):
        with self.conn.cursor() as cur:
            return stats.run(cur, stat, **filters)


def decide(plan, db):
    """Steps 2-4. Returns a dict: either 'stop' (a message, no answer model needed)
    or 'rows' (the sources for the answer step), plus 'notes' explaining choices."""
    notes = []
    season = plan.season

    # 2. Names: every player must exist in our data, unambiguously.
    players = []
    if plan.players:
        known = db.known_players(season)
        for name in plan.players:
            resolved, cands = resolve_player(name, known)
            if resolved:
                players.append(resolved)
            elif cands:
                return {"stop": f"Which player do you mean: {', '.join(cands[:6])}?", "notes": notes}
            else:
                return {"stop": f"{name} isn't in our IPL {season} data, so I can't answer that.",
                        "notes": notes}
    teams = [t for t in (resolve_team(x) for x in plan.teams) if t]
    if len(teams) < len(plan.teams):
        notes.append(f"ignored unknown team names: {plan.teams}")

    if plan.route == "out_of_scope":
        return {"stop": "I can only answer questions about IPL matches in our data.", "notes": notes}

    # 3. Which match? (None = the question is about the whole season)
    match, note = resolve_match(db.matches(season), season, plan.stage, plan.match_number, teams)
    if note:
        notes.append(note)
    if note and not match:
        return {"stop": f"I couldn't find that match in our IPL {season} data.", "notes": notes}
    match_id = match["match_id"] if match else None
    if match:
        notes.append(f"match: {match['stage']} {match['match_number']}, "
                     f"{match['team1']} vs {match['team2']}, {match['match_date']}")

    # 4. Route.
    if plan.route == "stats":
        if plan.stat not in stats.ALL_STATS:
            notes.append(f"unknown stat {plan.stat!r}: falling back to search")
        else:
            filters = {"season": season, "match_id": match_id, "phase": plan.phase}
            if plan.stat in stats.PLAYER_STATS:
                if not players:
                    return {"stop": "Which player do you mean?", "notes": notes}
                filters["player"] = players[0]
            elif len(teams) == 1:
                filters["team"] = teams[0]          # e.g. RCB's top scorer (in the season, or in one match)
            title, cols, rows, sql, params = db.stats(plan.stat, **filters)
            scope = ", ".join(f"{k}={v}" for k, v in filters.items() if v and k != "match_id")
            if match:
                scope += f", {match['stage']} {match['match_number']}"
            label = f"stats: {title} ({scope})"
            content = stats.as_source(title, cols, rows, filters, match)
            return {"route": "stats", "rows": [(label, "stats", 0, 0, 0, content)], "notes": notes,
                    "sql": sql, "params": params, "match_id": match_id, "players": players}

    # Exact fields beat the route label: if the plan names a match AND a specific
    # over, fetch that over by its labels even if the rewriter said "search"
    # ("how did the last over go?" is descriptive AND exact). The answer model
    # then describes the right over.
    exact_over = match_id and plan.over is not None and plan.route in ("lookup", "search")
    if exact_over and plan.route == "search":
        notes.append("route was 'search' but the plan names a match and an over: using lookup")
    if (plan.route == "lookup" and match_id) or exact_over:
        rows = db.lookup(match_id, plan.innings, plan.over)
        if len(rows) > 1 and plan.innings is None and len(teams) == 1 and plan.over != "last":
            # "Punjab's 17th over": keep the innings where the named team batted.
            batting = [r for r in rows if f"{teams[0]} batting" in r[5]]
            if batting:
                notes.append(f"kept the innings where {teams[0]} batted")
                rows = batting
        if rows:
            if plan.over == "last":
                notes.append(f"'last over' resolved from the data: innings {rows[0][2]}, over {rows[0][3]}")
                # The router knows this over ended the innings; the over chunk alone doesn't
                # say so. The match summary (innings totals in N overs, result) gives the
                # answer model that fact, so it doesn't have to assume it.
                rows = rows + db.lookup(match_id)
            return {"route": "lookup", "rows": rows, "notes": notes, "match_id": match_id, "players": players}
        notes.append("lookup found nothing: falling back to search")
    elif plan.route == "lookup":
        notes.append("lookup without a match: falling back to search")

    rows = db.search(plan.search_text, match_id, players)
    if not rows and players:
        notes.append("no chunks with that player: searching without the player filter")
        rows = db.search(plan.search_text, match_id)
    return {"route": "search", "rows": rows, "notes": notes, "match_id": match_id, "players": players}


def answer_question(question, conn, model, search_mode="vector"):
    """The whole pipeline for one question, returning everything measured.
    Used by main() below and by eval.py."""
    from answer import run_answer
    from caching import CachePolicy
    from common import CHAT_IN_PRICE, CHAT_MODEL, CHAT_OUT_PRICE
    from rewrite import rewrite

    out = {"question": question}
    t0 = time.perf_counter()
    plan, meta = rewrite(question, model, CHAT_MODEL, CachePolicy(), CHAT_IN_PRICE, CHAT_OUT_PRICE)
    out.update(plan=plan, rewrite=meta, rewrite_ms=(time.perf_counter() - t0) * 1000)

    db = Database(conn, search_mode)
    d = decide(plan, db)
    out.update(decision=d, search=db.last_search)
    if "stop" in d:
        out.update(route="stop", answer=d["stop"], result=None,
                   cost_usd=meta["cost_usd"] + 0, total_ms=(time.perf_counter() - t0) * 1000)
        return out

    r = run_answer(question, d["rows"], "rag", model, CHAT_MODEL, CachePolicy())
    out.update(route=d["route"], answer=r["answer"], result=r,
               cost_usd=meta["cost_usd"] + r["cost_usd"] + db.last_search.get("rerank_cost", 0.0),
               total_ms=(time.perf_counter() - t0) * 1000)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--show-plan", action="store_true")
    ap.add_argument("--show-sources", action="store_true")
    ap.add_argument("--search-mode", choices=search_modes.MODES, default=None,
                    help="default: SEARCH_MODE in .env (hybrid_rerank)")
    args = ap.parse_args()

    from common import SEARCH_MODE, connect, get_chat_model

    with connect() as conn:
        o = answer_question(args.question, conn, get_chat_model(), args.search_mode or SEARCH_MODE)
    plan, meta, d = o["plan"], o["rewrite"], o["decision"]
    if args.show_plan:
        print(json.dumps(plan.model_dump(exclude_none=True), indent=1, ensure_ascii=False))
    print(f"\nQ: {args.question}")
    for n in d["notes"]:
        print(f"  note: {n}")
    if o["route"] == "stop":
        print(f"\n{o['answer']}\n\nroute: stopped before searching · no answer model call")
        print(f"rewrite: {o['rewrite_ms']:.0f} ms · ${meta['cost_usd']:.6f}")
        return
    if args.show_sources:
        for row in d["rows"]:
            print(f"\n--- source: {row[0]} | {row[1]} ---\n{row[5]}")
    if d.get("sql"):
        print(f"  sql: {d['sql']}\n  params: {d['params']}")
    if o["search"]:
        s = o["search"]
        extra = f", rerank scores {s['rerank_scores']}" if "rerank_scores" in s else ""
        print(f"  search: {s['mode']} in {s['ms']:.0f} ms{extra}")

    r = o["result"]
    print(f"\n{r['answer']}\n")
    for sid, key in r["cited"]:
        print(f"  [{sid}] {key}")
    if r["truncated"]:
        print("  WARNING: answer was cut off (hit max_tokens)")
    print(f"\nroute: {o['route']} · {len(d['rows'])} sources · plan: {plan.route}"
          f"{' / ' + plan.stat if plan.stat else ''}")
    print(f"rewrite: {o['rewrite_ms']:.0f} ms, {meta['input_tokens']} in / {meta['output_tokens']} out"
          f" (cache read {meta['cache_read']}) · answer: TTFT {r['ttft_ms']:.0f} ms, total {r['total_ms']:.0f} ms")
    print(f"total {o['total_ms']:.0f} ms · cost ${o['cost_usd']:.6f}")


if __name__ == "__main__":
    main()
