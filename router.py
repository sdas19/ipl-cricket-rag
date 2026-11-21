"""
The router: one question in, the right tool for it, one grounded answer out.

  1. rewrite   cheap model call -> Plan (route, filters, English search text)
  2. names     check players/teams against our data   (stop if unknown or ambiguous)
  3. match     "the final" / "match 57" / two teams -> one match_id
  4. route     lookup -> fetch exact chunks by label columns (no search)
               stats  -> fixed SQL templates over every ball (no search)
               search -> vector search inside the filters
               out_of_scope -> stop
  5. answer    the existing answer step (citations, "I don't know" rules)

  python router.py "final me last over kisne daala aur kitne run bane?"
  python router.py "Who hit the most sixes in the 2025 season?" --show-plan
"""

import argparse
import json
import time

import stats
from resolve import resolve_match, resolve_player, resolve_team

COLUMNS = "match_id, level, innings, over_start, over_end, content"


class Database:
    """Everything the router reads, behind small methods (so tests can use a fake)."""

    def __init__(self, conn):
        self.conn = conn

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

    def search(self, text, match_id=None, players=(), k=5):
        """Vector search inside the filters (match and/or players)."""
        from ask import embed_question
        from common import EMBED_MODEL, to_pgvector

        vec, _ = embed_question(text)
        sql = (f"select {COLUMNS} from chunks where embed_model = %(model)s "
               "and (%(match)s::text is null or match_id = %(match)s) "
               "and players @> %(players)s::text[] "
               "order by embedding <=> %(q)s::vector limit %(k)s")
        return self.conn.execute(sql, {"model": EMBED_MODEL, "match": match_id, "players": list(players),
                                       "q": to_pgvector(vec), "k": k}).fetchall()

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
            label = f"stats: {title} ({', '.join(f'{k}={v}' for k, v in filters.items() if v)})"
            content = stats.as_source(title, cols, rows, filters)
            return {"route": "stats", "rows": [(label, "stats", 0, 0, 0, content)], "notes": notes,
                    "sql": sql, "params": params}

    # Exact fields beat the route label: if the plan names a match AND a specific
    # over, fetch that over by its labels even if the rewriter said "search"
    # ("how did the last over go?" is descriptive AND exact). The answer model
    # then describes the right over.
    exact_over = match_id and plan.over is not None and plan.route in ("lookup", "search")
    if exact_over and plan.route == "search":
        notes.append("route was 'search' but the plan names a match and an over: using lookup")
    if (plan.route == "lookup" and match_id) or exact_over:
        rows = db.lookup(match_id, plan.innings, plan.over)
        if rows:
            if plan.over == "last":
                notes.append(f"'last over' resolved from the data: innings {rows[0][2]}, over {rows[0][3]}")
                # The router knows this over ended the innings; the over chunk alone doesn't
                # say so. The match summary (innings totals in N overs, result) gives the
                # answer model that fact, so it doesn't have to assume it.
                rows = rows + db.lookup(match_id)
            return {"route": "lookup", "rows": rows, "notes": notes}
        notes.append("lookup found nothing: falling back to search")
    elif plan.route == "lookup":
        notes.append("lookup without a match: falling back to search")

    rows = db.search(plan.search_text, match_id, players)
    if not rows and players:
        notes.append("no chunks with that player: searching without the player filter")
        rows = db.search(plan.search_text, match_id)
    return {"route": "search", "rows": rows, "notes": notes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--show-plan", action="store_true")
    ap.add_argument("--show-sources", action="store_true")
    args = ap.parse_args()

    from answer import run_answer
    from caching import CachePolicy
    from common import CHAT_IN_PRICE, CHAT_MODEL, CHAT_OUT_PRICE, connect, get_chat_model
    from rewrite import rewrite

    t0 = time.perf_counter()
    model = get_chat_model()
    plan, meta = rewrite(args.question, model, CHAT_MODEL, CachePolicy(), CHAT_IN_PRICE, CHAT_OUT_PRICE)
    t_rewrite = (time.perf_counter() - t0) * 1000
    if args.show_plan:
        print(json.dumps(plan.model_dump(exclude_none=True), indent=1, ensure_ascii=False))

    with connect() as conn:
        d = decide(plan, Database(conn))
    print(f"\nQ: {args.question}")
    for n in d["notes"]:
        print(f"  note: {n}")

    if "stop" in d:
        print(f"\n{d['stop']}\n\nroute: stopped before searching · no answer model call")
        print(f"rewrite: {t_rewrite:.0f} ms · ${meta['cost_usd']:.6f}")
        return

    if args.show_sources:
        for row in d["rows"]:
            print(f"\n--- source: {row[0]} | {row[1]} ---\n{row[5]}")
    if d.get("sql"):
        print(f"  sql: {d['sql']}\n  params: {d['params']}")

    r = run_answer(args.question, d["rows"], "rag", model, CHAT_MODEL, CachePolicy())
    print(f"\n{r['answer']}\n")
    for sid, key in r["cited"]:
        print(f"  [{sid}] {key}")
    if r["truncated"]:
        print("  WARNING: answer was cut off (hit max_tokens)")
    total_cost = meta["cost_usd"] + r["cost_usd"]
    print(f"\nroute: {d['route']} · {len(d['rows'])} sources · plan: {plan.route}"
          f"{' / ' + plan.stat if plan.stat else ''}")
    print(f"rewrite: {t_rewrite:.0f} ms, {meta['input_tokens']} in / {meta['output_tokens']} out"
          f" (cache read {meta['cache_read']}) · answer: TTFT {r['ttft_ms']:.0f} ms, total {r['total_ms']:.0f} ms")
    print(f"cost: rewrite ${meta['cost_usd']:.6f} + answer ${r['cost_usd']:.6f} = ${total_cost:.6f}")


if __name__ == "__main__":
    main()
