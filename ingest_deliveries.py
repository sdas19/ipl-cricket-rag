"""
Fill the structured tables (matches, deliveries, players) from Cricsheet JSON.
No AI and no embeddings: just copying and calculating, so it's fast and free.

  python ingest_deliveries.py --season 2025 --dry-run   # counts only
  python ingest_deliveries.py --season 2025             # replace that season's rows

Safe to re-run: the season's rows are deleted and re-inserted in ONE
transaction, so the tables are never half-filled.
"""

import argparse
import time
from pathlib import Path

from chunker import BOWLER_WICKETS, load_match, phase_of, result_text

MATCH_COLS = ["match_id", "season", "match_date", "match_number", "stage", "team1", "team2",
              "venue", "winner", "result", "player_of_match"]
BALL_COLS = ["match_id", "season", "innings", "seq", "over_number", "ball_label", "phase",
             "batting_team", "bowling_team", "batter", "bowler", "non_striker",
             "batter_runs", "extra_runs", "total_runs", "bowler_runs", "is_legal", "is_wide", "is_four", "is_six",
             "is_wicket", "wicket_kind", "player_out", "fielder", "bowler_wicket"]
PLAYER_COLS = ["season", "name", "cricsheet_id", "teams"]


def parse_match(data, match_id):
    """One Cricsheet match -> (match row, ball rows, {player: (id, set of teams)})."""
    info = data["info"]
    season = str(info.get("season", ""))
    event = info.get("event", {})
    teams = info["teams"]
    match = {
        "match_id": match_id,
        "season": season,
        "match_date": info["dates"][0],
        "match_number": event.get("match_number"),      # playoffs have none: numbered later
        "stage": event.get("stage") or "League",
        "team1": teams[0],
        "team2": teams[1],
        "venue": info.get("venue"),
        "winner": info.get("outcome", {}).get("winner"),
        "result": result_text(info),
        "player_of_match": info.get("player_of_match", []),
    }

    registry = info.get("registry", {}).get("people", {})
    players = {}
    for team, names in info.get("players", {}).items():   # the playing XIs
        for name in names:
            players.setdefault(name, (registry.get(name), set()))[1].add(team)

    balls = []
    for innings_no, inn in enumerate(data.get("innings", []), start=1):
        if inn.get("super_over"):
            continue
        batting = inn["team"]
        bowling = teams[1] if batting == teams[0] else teams[0]
        seq = 0
        for ov in inn.get("overs", []):
            legal_in_over = 0
            for d in ov["deliveries"]:
                seq += 1
                ex = d.get("extras", {})
                is_legal = "wides" not in ex and "noballs" not in ex
                if is_legal:
                    legal_in_over += 1
                r = d["runs"]
                wickets = d.get("wickets", [])
                w = wickets[0] if wickets else {}
                fielders = [f.get("name") for f in w.get("fielders", []) if f.get("name")]
                balls.append({
                    "match_id": match_id, "season": season, "innings": innings_no, "seq": seq,
                    "over_number": ov["over"] + 1,
                    "ball_label": f"{ov['over']}.{legal_in_over if is_legal else legal_in_over + 1}",
                    "phase": phase_of(ov["over"] + 1, info.get("match_type")),
                    "batting_team": batting, "bowling_team": bowling,
                    "batter": d["batter"], "bowler": d["bowler"], "non_striker": d.get("non_striker"),
                    "batter_runs": r.get("batter", 0), "extra_runs": r.get("extras", 0), "total_runs": r["total"],
                    "bowler_runs": r["total"] - ex.get("byes", 0) - ex.get("legbyes", 0) - ex.get("penalty", 0),
                    "is_legal": is_legal,
                    "is_wide": "wides" in ex,
                    "is_four": r.get("batter") == 4 and not r.get("non_boundary", False),
                    "is_six": r.get("batter") == 6,
                    "is_wicket": bool(wickets),
                    "wicket_kind": w.get("kind"), "player_out": w.get("player_out"),
                    "fielder": fielders[0] if fielders else None,
                    "bowler_wicket": any(x.get("kind") in BOWLER_WICKETS for x in wickets),
                })
    return match, balls, players


def number_playoffs(matches):
    """Cricsheet gives playoffs a stage name but no number. Number them after
    the league, in date order (IPL 2025: Qualifier 1 = 71 ... Final = 74)."""
    league_max = max((m["match_number"] or 0) for m in matches)
    playoffs = sorted((m for m in matches if m["match_number"] is None), key=lambda m: m["match_date"])
    for i, m in enumerate(playoffs, start=1):
        m["match_number"] = league_max + i
    return matches


def collect(data_dir, season):
    matches, balls, players = [], [], {}
    for path in sorted(Path(data_dir).glob("*.json")):
        data, match_id = load_match(path)
        if not str(data["info"].get("season", "")).startswith(season):
            continue
        m, b, p = parse_match(data, match_id)
        matches.append(m)
        balls += b
        for name, (pid, teams) in p.items():
            players.setdefault(name, (pid, set()))[1].update(teams)
    number_playoffs(matches)
    player_rows = [{"season": season, "name": n, "cricsheet_id": pid, "teams": sorted(t)}
                   for n, (pid, t) in sorted(players.items())]
    return matches, balls, player_rows


def save(matches, balls, players, season):
    """Replace the season in one transaction. COPY sends all rows in one stream:
    far faster than one INSERT per row over the internet."""
    from common import connect

    with connect() as conn, conn.cursor() as cur:
        cur.execute("delete from deliveries where season = %s", (season,))
        cur.execute("delete from players where season = %s", (season,))
        cur.execute("delete from matches where season = %s", (season,))
        for table, cols, rows in (("matches", MATCH_COLS, matches),
                                  ("deliveries", BALL_COLS, balls),
                                  ("players", PLAYER_COLS, players)):
            with cur.copy(f"copy {table} ({', '.join(cols)}) from stdin") as copy:
                for row in rows:
                    copy.write_row([row[c] for c in cols])
    # leaving the 'with' block commits all three tables together


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/ipl_json")
    ap.add_argument("--season", default="2025")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    matches, balls, players = collect(args.data_dir, args.season)
    stages = sorted({(m["match_number"], m["stage"]) for m in matches if m["stage"] != "League"})
    print(f"{len(matches)} matches, {len(balls):,} balls, {len(players)} players")
    print("Playoffs:", ", ".join(f"Match {n} = {s}" for n, s in stages))
    if args.dry_run:
        return
    t0 = time.perf_counter()
    save(matches, balls, players, args.season)
    print(f"Saved in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
