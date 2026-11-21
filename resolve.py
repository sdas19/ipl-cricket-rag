"""
Turn the names in a question into exact values from our data.

  teams:   "RCB", "bangalore", "Royal Challengers" -> "Royal Challengers Bengaluru"
  players: "Virat Kohli", "Kohli", "King Kohli"    -> "V Kohli"   (Cricsheet's spelling)
           "Ashok Dinda"                            -> not in our data (stop: no search)
  match:   stage "Final" / match 57 / two teams     -> one match_id

Pure functions (no database, no AI): cheap, instant, and easy to test.
"""

# Short forms and common names users type -> Cricsheet's full team names.
TEAM_ALIASES = {
    "Royal Challengers Bengaluru": ["rcb", "bangalore", "bengaluru", "royal challengers", "royal challengers bangalore"],
    "Mumbai Indians": ["mi", "mumbai"],
    "Chennai Super Kings": ["csk", "chennai"],
    "Kolkata Knight Riders": ["kkr", "kolkata"],
    "Punjab Kings": ["pbks", "punjab", "kxip", "kings xi punjab"],
    "Delhi Capitals": ["dc", "delhi"],
    "Sunrisers Hyderabad": ["srh", "hyderabad", "sunrisers"],
    "Rajasthan Royals": ["rr", "rajasthan"],
    "Gujarat Titans": ["gt", "gujarat", "titans"],
    "Lucknow Super Giants": ["lsg", "lucknow"],
}
_TEAM_LOOKUP = {a: full for full, aliases in TEAM_ALIASES.items() for a in aliases + [full.lower()]}

# Nicknames fans use that no spelling rule can work out.
NICKNAMES = {"king kohli": "V Kohli", "hitman": "RG Sharma", "thala": "MS Dhoni", "sky": "SA Yadav"}


def resolve_team(text):
    return _TEAM_LOOKUP.get((text or "").strip().lower())


def resolve_player(name, known):
    """name: as the user (or the rewriter) wrote it. known: Cricsheet names for the season.
    Returns (resolved_name or None, candidates). One candidate = resolved;
    several = ambiguous (ask the user); none = not in our data."""
    q = " ".join(name.strip().lower().replace(".", " ").split())
    if q in NICKNAMES and NICKNAMES[q] in known:
        return NICKNAMES[q], [NICKNAMES[q]]
    for k in known:                                   # exact, ignoring case
        if k.lower() == q:
            return k, [k]
    parts = q.split()
    if not parts:
        return None, []
    surname = parts[-1]
    # Same surname: "Virat Kohli" / "Kohli" -> "V Kohli"
    cands = [k for k in known if k.lower().split()[-1] == surname]
    if not cands:
        # Single-word names in the data, e.g. "Priyansh Arya" typed as "Arya" is covered above;
        # here: the typed word appears anywhere in a name ("Shashank" -> "Shashank Singh").
        cands = [k for k in known if surname in k.lower().split()]
    if len(cands) > 1 and len(parts) > 1:
        # Narrow by first letter: "Rohit Sharma" -> "RG Sharma", not "Abhishek Sharma"
        first = parts[0][0]
        narrowed = [k for k in cands if k.lower().split()[0].startswith(first)]
        if narrowed:
            cands = narrowed
    if len(cands) == 1:
        return cands[0], cands
    return None, sorted(cands)


def resolve_match(matches, season=None, stage=None, match_number=None, teams=()):
    """matches: dicts from the matches table. Returns (match or None, note).
    No match details in the question -> (None, None): a season-wide question."""
    pool = [m for m in matches if not season or m["season"] == season]
    # One team on its own ("RCB's top scorer") describes a TEAM, not a match:
    # only two teams, a stage or a match number point at a specific match.
    if not (stage or match_number or len(teams) >= 2):
        return None, None
    if stage:
        pool = [m for m in pool if m["stage"].lower() == stage.lower()]
    if match_number:
        pool = [m for m in pool if m["match_number"] == match_number]
    for t in teams:
        pool = [m for m in pool if t in (m["team1"], m["team2"])]
    if not pool:
        return None, "no match in our data fits that description"
    pool.sort(key=lambda m: str(m["match_date"]))
    if len(pool) == 1:
        return pool[0], None
    # Several fit (e.g. "the MI vs CSK match" happened twice): assume the latest, and say so.
    latest = pool[-1]
    return latest, (f"{len(pool)} matches fit; using the latest "
                    f"(Match {latest['match_number']}, {latest['match_date']})")
