"""
The stats route: answers that must be CALCULATED from every ball, not found in a chunk.

Safety: the model never writes SQL. It only picks a stat name and filters
(from a fixed list); this file turns them into SQL from fixed templates, with
values passed as parameters (never pasted into the SQL text). So no wrong
joins, no injection, no runaway queries.
"""

# Leaderboards: who leads the season (or a match / phase / team) at something.
#   key: (title, player column, value expression, order, team column)
LEADERBOARDS = {
    "most_sixes":   ("Most sixes",   "batter", "count(*) filter (where is_six)",   "desc", "batting_team"),
    "most_fours":   ("Most fours",   "batter", "count(*) filter (where is_four)",  "desc", "batting_team"),
    "most_runs":    ("Most runs",    "batter", "sum(batter_runs)",                 "desc", "batting_team"),
    "most_wickets": ("Most wickets", "bowler", "count(*) filter (where bowler_wicket)", "desc", "bowling_team"),
    "best_economy": ("Best economy rate (runs per over)", "bowler",
                     "round(sum(bowler_runs) * 6.0 / nullif(count(*) filter (where is_legal), 0), 2)",
                     "asc", "bowling_team"),
}
# One player's totals.
PLAYER_STATS = {"player_batting", "player_bowling"}
ALL_STATS = sorted(LEADERBOARDS) + sorted(PLAYER_STATS)


def _where(season, match_id=None, phase=None, team=None, team_col=None):
    clauses, params = ["season = %(season)s"], {"season": season}
    if match_id:
        clauses.append("match_id = %(match_id)s")
        params["match_id"] = match_id
    if phase:
        clauses.append("phase = %(phase)s")
        params["phase"] = phase
    if team and team_col:
        clauses.append(f"{team_col} = %(team)s")          # column name comes from our template, not the user
        params["team"] = team
    return " and ".join(clauses), params


def build_query(stat, season, match_id=None, phase=None, team=None, player=None, limit=5):
    """-> (title, sql, params). Raises ValueError for anything not in the fixed list."""
    if stat in LEADERBOARDS:
        title, who, value, order, team_col = LEADERBOARDS[stat]
        where, params = _where(season, match_id, phase, team, team_col)
        # Economy needs a minimum number of balls, or one tidy over tops the list.
        having = ""
        if stat == "best_economy":
            having = "having count(*) filter (where is_legal) >= %(min_balls)s"
            params["min_balls"] = 12 if match_id else 60
        sql = (f"select {who} as player, {value} as value from deliveries where {where} "
               f"group by {who} {having} order by value {order}, player limit %(limit)s")
        params["limit"] = limit
        return title, sql, params

    if stat in PLAYER_STATS:
        if not player:
            raise ValueError(f"{stat} needs a player")
        where, params = _where(season, match_id, phase)
        params["player"] = player
        if stat == "player_batting":
            title = f"Batting: {player}"
            # One pass over every ball the player batted on OR was out on
            # (a run out can happen while he's the non-striker).
            sql = (f"select count(distinct (match_id, innings)) filter (where batter = %(player)s) as innings, "
                   f"sum(batter_runs) filter (where batter = %(player)s) as runs, "
                   f"count(*) filter (where batter = %(player)s and not is_wide) as balls_faced, "
                   f"count(*) filter (where batter = %(player)s and is_four) as fours, "
                   f"count(*) filter (where batter = %(player)s and is_six) as sixes, "
                   f"count(*) filter (where player_out = %(player)s) as dismissals "
                   f"from deliveries where {where} and (batter = %(player)s or player_out = %(player)s)")
        else:
            title = f"Bowling: {player}"
            sql = (f"select count(distinct (match_id, innings)) as innings, "
                   f"count(*) filter (where is_legal) as balls, sum(bowler_runs) as runs_conceded, "
                   f"count(*) filter (where bowler_wicket) as wickets, "
                   f"round(sum(bowler_runs) * 6.0 / nullif(count(*) filter (where is_legal), 0), 2) as economy "
                   f"from deliveries where {where} and bowler = %(player)s")
        return title, sql, params

    raise ValueError(f"unknown stat {stat!r}; allowed: {ALL_STATS}")


def run(cur, stat, **filters):
    """Run one stat. Returns (title, column names, rows, sql, params)."""
    title, sql, params = build_query(stat, **filters)
    cur.execute(sql, params)
    cols = [d.name for d in cur.description]
    return title, cols, cur.fetchall(), sql, params


def describe_match(m):
    """'Final (Match 74), Royal Challengers Bengaluru vs Punjab Kings, 2025-06-03'"""
    stage = m["stage"] if m["stage"] != "League" else "League match"
    return f"{stage} (Match {m['match_number']}), {m['team1']} vs {m['team2']}, {m['match_date']}"


def as_source(title, cols, rows, filters, match=None):
    """Format the result as one source for the answer model. It says plainly
    that it was computed from ALL balls, so the model may state totals from it.
    The scope is written in words: the answer model can't know that
    match_id 1473511 is the Final unless the source says so."""
    parts = []
    for k, v in filters.items():
        if not v or k == "limit":
            continue
        if k == "match_id" and match:
            parts.append(f"match = {describe_match(match)}")
        else:
            parts.append(f"{k}={v}")
    scope = ", ".join(parts)
    lines = [f"Computed from all ball-by-ball data ({scope}): {title}"]
    if not rows or all(v is None for v in rows[0]):
        lines.append("No matching balls found.")
    elif cols[:2] == ["player", "value"]:
        lines += [f"{i}. {p}: {v}" for i, (p, v) in enumerate(rows, start=1)]
    else:
        lines += [f"{c}: {v}" for c, v in zip(cols, rows[0])]
    return "\n".join(lines)
