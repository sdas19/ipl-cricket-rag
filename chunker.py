"""
Turn one Cricsheet match file (ball-by-ball JSON) into chunks for RAG.

Two kinds of chunk per match:
  - level "over":  one per over per innings. A context header (match, innings,
                   over, phase, score situation) + one line per ball.
  - level "match": one per match. Result, toss, top batters and bowlers.

No AI here: everything is copied from the data (layer 1) or calculated
by code (layer 2). Run it on one file to see the chunks:

    python chunker.py data/ipl_json/1473511.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# Dismissals the bowler gets credit for (run outs etc. are not the bowler's).
BOWLER_WICKETS = {"bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"}


def phase_of(over_no, match_type):
    """T20 phases by over number (1-based). None for other formats."""
    if match_type not in ("T20", "IT20"):
        return None
    if over_no <= 6:
        return "powerplay"
    if over_no <= 15:
        return "middle"
    return "death"


def dismissal_text(w, bowler):
    """Cricsheet wicket -> scorecard style, e.g. 'c Jadeja b Ashwin'."""
    kind = w.get("kind", "out")
    fielders = [f.get("name", "sub") for f in w.get("fielders", [])]
    f = fielders[0] if fielders else ""
    return {
        "caught": f"c {f} b {bowler}",
        "caught and bowled": f"c & b {bowler}",
        "bowled": f"b {bowler}",
        "lbw": f"lbw b {bowler}",
        "stumped": f"st {f} b {bowler}",
        "run out": f"run out ({f})" if f else "run out",
        "hit wicket": f"hit wicket b {bowler}",
    }.get(kind, kind)


def describe_ball(d):
    """One delivery -> a short commentary-style description."""
    r, extras = d["runs"], d.get("extras", {})
    parts = []
    if "wides" in extras:
        parts.append(f"wide (+{extras['wides']})")
    if "noballs" in extras:
        parts.append("no ball")
    bat = r.get("batter", 0)
    if bat == 6:
        parts.append("SIX")
    elif bat == 4 and not r.get("non_boundary"):
        parts.append("FOUR")
    elif bat:
        parts.append(f"{bat} run" + ("s" if bat > 1 else ""))
    for k, name in (("byes", "bye"), ("legbyes", "leg bye")):
        if k in extras:
            parts.append(f"{extras[k]} {name}" + ("s" if extras[k] > 1 else ""))
    for w in d.get("wickets", []):
        parts.append(f"OUT! {w['player_out']} {dismissal_text(w, d['bowler'])}")
    return ", ".join(parts) if parts else "no run"


def match_meta(data, match_id):
    info = data["info"]
    event = info.get("event", {})
    return {
        "match_id": match_id,
        "match_date": info["dates"][0],
        "season": str(info.get("season", "")),
        "competition": event.get("name"),
        "stage": event.get("stage"),
        "match_number": event.get("match_number"),
        "venue": info.get("venue"),
        "teams": info["teams"],
        "match_type": info.get("match_type"),
        "registry": info.get("registry", {}).get("people", {}),
    }


def match_title(m):
    """Line 1 of every header, e.g.
    'Indian Premier League 2025 · Final · RCB vs PBKS · Narendra Modi Stadium · 2025-06-03'"""
    label = m["stage"] or (f"Match {m['match_number']}" if m["match_number"] else None)
    bits = [f"{m['competition'] or 'Match'} {m['season']}".strip()]
    if label:
        bits.append(label)
    bits += [" vs ".join(m["teams"]), m["venue"] or "", m["match_date"]]
    return " · ".join(b for b in bits if b)


def result_text(info):
    o = info.get("outcome", {})
    if "winner" in o:
        by = o.get("by", {})
        if "runs" in by:
            txt = f"{o['winner']} won by {by['runs']} runs"
        elif "wickets" in by:
            txt = f"{o['winner']} won by {by['wickets']} wickets"
        else:
            txt = f"{o['winner']} won"
        if o.get("method"):
            txt += f" ({o['method']})"
        return txt
    if o.get("result") == "tie":
        return "Match tied" + (f"; {o['eliminator']} won the super over" if o.get("eliminator") else "")
    if o.get("result") == "no result":
        return "No result"
    return "Result unknown"


def chunk_match(data, match_id):
    """Return the list of chunks (dicts) for one match."""
    m = match_meta(data, match_id)
    info = data["info"]
    title = match_title(m)
    is_t20 = m["match_type"] in ("T20", "IT20")
    chunks = []
    innings_summaries = []
    first_innings_total = None

    for idx, inn in enumerate(data.get("innings", []), start=1):
        if inn.get("super_over"):
            continue  # kept out for simplicity; the match chunk mentions the result
        team = inn["team"]
        target = inn.get("target", {}).get("runs")
        if target is None and idx == 2 and first_innings_total is not None:
            target = first_innings_total + 1

        runs = wkts = legal_balls = 0
        bat = defaultdict(lambda: {"runs": 0, "balls": 0, "4s": 0, "6s": 0})
        bowl = defaultdict(lambda: {"balls": 0, "runs": 0, "wkts": 0})

        for ov in inn.get("overs", []):
            over_no = ov["over"] + 1               # Cricsheet counts overs from 0
            start_runs, start_wkts, start_balls = runs, wkts, legal_balls
            lines, players, legal_in_over = [], set(), 0

            for d in ov["deliveries"]:
                extras = d.get("extras", {})
                is_legal = "wides" not in extras and "noballs" not in extras
                if is_legal:
                    legal_in_over += 1
                    legal_balls += 1
                label = f"{ov['over']}.{legal_in_over if is_legal else legal_in_over + 1}"
                lines.append(f"{label} {d['bowler']} to {d['batter']}: {describe_ball(d)}")

                r = d["runs"]
                runs += r["total"]
                wkts += len(d.get("wickets", []))
                players.update([d["batter"], d["bowler"], d["non_striker"]])
                for w in d.get("wickets", []):
                    players.update(f.get("name") for f in w.get("fielders", []) if f.get("name"))

                # Scorecard stats for the match-level chunk.
                b = bat[d["batter"]]
                b["runs"] += r.get("batter", 0)
                if "wides" not in extras:
                    b["balls"] += 1
                if r.get("batter") == 4 and not r.get("non_boundary"):
                    b["4s"] += 1
                if r.get("batter") == 6:
                    b["6s"] += 1
                bw = bowl[d["bowler"]]
                bw["balls"] += 1 if is_legal else 0
                bw["runs"] += r["total"] - extras.get("byes", 0) - extras.get("legbyes", 0) - extras.get("penalty", 0)
                bw["wkts"] += sum(1 for w in d.get("wickets", []) if w.get("kind") in BOWLER_WICKETS)

            phase = phase_of(over_no, m["match_type"])
            situation = f"Score at start of over: {start_runs}/{start_wkts}"
            if target:
                situation = f"chasing {target} · " + situation
                if is_t20:
                    situation += f", need {target - start_runs} off {120 - start_balls} balls"
            header = "\n".join([
                title,
                f"Innings {idx}: {team} batting · Over {over_no}"
                + (f" ({phase})" if phase else "") + f" · {situation}",
            ])
            footer = (f"End of over {over_no}: {team} {runs}/{wkts} "
                      f"({runs - start_runs} runs, {wkts - start_wkts} wicket"
                      f"{'' if wkts - start_wkts == 1 else 's'} in the over)")
            players.discard(None)
            chunks.append({
                **{k: m[k] for k in ("match_id", "match_date", "season", "competition", "stage", "venue", "teams")},
                "level": "over",
                "innings": idx,
                "over_start": over_no,
                "over_end": over_no,
                "phase": phase,
                "players": sorted(players),
                "player_ids": sorted({m["registry"][p] for p in players if p in m["registry"]}),
                "content": header + "\n---\n" + "\n".join(lines) + "\n" + footer,
            })

        if idx == 1:
            first_innings_total = runs
        top_bat = sorted(bat.items(), key=lambda kv: (-kv[1]["runs"], kv[1]["balls"]))[:3]
        top_bowl = sorted(bowl.items(), key=lambda kv: (-kv[1]["wkts"], kv[1]["runs"]))[:2]
        overs_txt = f"{legal_balls // 6}.{legal_balls % 6}" if legal_balls % 6 else str(legal_balls // 6)
        innings_summaries.append(
            f"Innings {idx}: {team} {runs}/{wkts} in {overs_txt} overs. "
            + "Top batters: " + ", ".join(f"{n} {s['runs']} ({s['balls']} balls, {s['4s']}x4, {s['6s']}x6)" for n, s in top_bat)
            + ". Top bowlers: " + ", ".join(f"{n} {s['wkts']}/{s['runs']}" for n, s in top_bowl) + "."
        )

    toss = info.get("toss", {})
    pom = ", ".join(info.get("player_of_match", [])) or "not recorded"
    match_content = "\n".join([
        title,
        f"Toss: {toss.get('winner', '?')} won the toss and chose to {toss.get('decision', '?')}.",
        f"Result: {result_text(info)}. Player of the match: {pom}.",
        *innings_summaries,
    ])
    everyone = sorted({p for c in chunks for p in c["players"]})
    chunks.append({
        **{k: m[k] for k in ("match_id", "match_date", "season", "competition", "stage", "venue", "teams")},
        "level": "match",
        "innings": 0,
        "over_start": 0,
        "over_end": 0,
        "phase": None,
        "players": everyone,
        "player_ids": sorted({m["registry"][p] for p in everyone if p in m["registry"]}),
        "content": match_content,
    })
    return chunks


def load_match(path):
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f), path.stem


if __name__ == "__main__":
    data, match_id = load_match(sys.argv[1])
    chunks = chunk_match(data, match_id)
    print(f"{len(chunks)} chunks for match {match_id}\n")
    for c in chunks[:2] + chunks[-1:]:
        print(f"--- {c['level']} | innings {c['innings']} | over {c['over_start']} ---")
        print(c["content"], "\n")
