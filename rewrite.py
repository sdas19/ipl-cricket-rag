"""
Query rewriting: ONE cheap model call turns a messy question (English, Hindi,
Hinglish, nicknames, "the final", "last over") into a structured Plan.

The Plan is validated with Pydantic. Structured output here is "request +
validate + retry": we ask for JSON, check it against the schema, and if it
doesn't fit, send the error back once. Code downstream only ever sees a
valid Plan.
"""

import re
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field, ValidationError

from caching import CachePolicy, build_messages, cache_counts, cost_usd
from stats import ALL_STATS


class Plan(BaseModel):
    route: Literal["lookup", "stats", "search", "out_of_scope"] = Field(
        description="lookup: a specific over/innings/match result; stats: needs counting across balls "
                    "(most, total, best, a player's figures); search: describe/how/what happened; "
                    "out_of_scope: not about IPL cricket data")
    season: str = Field("2025", description="IPL season year")
    stage: Optional[Literal["Final", "Qualifier 1", "Qualifier 2", "Eliminator"]] = None
    match_number: Optional[int] = None
    teams: List[str] = Field(default_factory=list, description="teams mentioned, as written or short codes (RCB, MI)")
    players: List[str] = Field(default_factory=list, description="players mentioned, expanded to full names")
    innings: Optional[int] = Field(None, ge=1, le=2)
    # "last" is a POSITION, not a number: an innings can end before over 20
    # (a chase completed early, a team bowled out), so the lookup finds it in the data.
    over: Optional[Union[Literal["last"], Annotated[int, Field(ge=1, le=20)]]] = None
    phase: Optional[Literal["powerplay", "middle", "death"]] = None
    stat: Optional[str] = Field(None, description=f"for route=stats, one of: {', '.join(ALL_STATS)}")
    search_text: str = Field(description="the question rewritten in clear English")
    language: Literal["english", "hindi", "hinglish"] = "english"


REWRITE_RULES = f"""You convert cricket questions about IPL matches into a JSON plan. You do not answer the question.

Return ONLY a JSON object with these fields:
- route: "lookup" | "stats" | "search" | "out_of_scope"
- season: IPL season year as text, e.g. "2025"
- stage: "Final" | "Qualifier 1" | "Qualifier 2" | "Eliminator" | null (null = a league match or not stated)
- match_number: integer or null
- teams: list of teams mentioned, as written or short codes (RCB, MI)
- players: list of players mentioned, expanded to full names
- innings: 1 | 2 | null
- over: 1-20 | "last" | null
- phase: "powerplay" | "middle" | "death" | null
- stat: for route "stats", one of {", ".join(ALL_STATS)}; otherwise null
- search_text: the question rewritten in clear English
- language: "english" | "hindi" | "hinglish"

Rules:
- route "stats" for anything that needs counting or ranking: most/top/highest/lowest/best, totals, averages, economy, strike rate, "how many", or one player's figures for a match or season. Set "stat" to one of: {", ".join(ALL_STATS)}. For one player's figures use player_batting or player_bowling and put the player in "players".
- route "lookup" when the question points at a specific over, innings or a match's result/scorecard (e.g. "who bowled the last over of the final", "who won match 57").
- route "search" for descriptive questions: how, why, what happened, describe (e.g. "how did Kohli get out").
- route "out_of_scope" if it isn't about IPL cricket.
- "the final" / "final match" means stage "Final".
- "last over" / "final over" / "aakhri over" means over "last" (never 20: an innings can end early). If the question says which innings or team, set innings; "of the chase" / "second innings" means innings 2; otherwise leave innings null (it then means the last over of the match).
- Powerplay = overs 1-6, middle = 7-15, death = 16-20.
- Expand first names and nicknames to the player's full name when you are sure: "Rohit" -> "Rohit Sharma", "Bumrah" -> "Jasprit Bumrah". Keep names you don't recognise exactly as written.
- Teams may be short codes (RCB, MI, CSK, KKR, PBKS, DC, SRH, RR, GT, LSG); copy them as written.
- Default season is "2025" unless the question says another year.
- Hindi/Hinglish words: kisne = who, daala/daali = bowled, kitne = how many, run bane = runs scored, kaise = how, aakhri = last, sabse zyada = most, chhakke = sixes, chauke = fours, wicket liye = took wickets.

Examples:
Q: final me last over kisne daala aur kitne run bane?
{{"route": "lookup", "season": "2025", "stage": "Final", "teams": [], "players": [], "over": "last", "search_text": "Who bowled the last over of the final and how many runs were scored?", "language": "hinglish"}}
Q: Who hit the most sixes in the death overs this season?
{{"route": "stats", "season": "2025", "stat": "most_sixes", "phase": "death", "teams": [], "players": [], "search_text": "Most sixes in death overs in the 2025 season", "language": "english"}}
Q: How did Virat Kohli get out in the final?
{{"route": "search", "season": "2025", "stage": "Final", "teams": [], "players": ["Virat Kohli"], "search_text": "How did Virat Kohli get out in the final?", "language": "english"}}"""


def parse_plan(text):
    """Pull the JSON object out of the model's reply and validate it."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in the reply")
    return Plan.model_validate_json(match.group(0))


def rewrite(question, model, model_id, policy=CachePolicy(), in_price=0.0, out_price=0.0):
    """-> (Plan, meta). One call; one retry with the validation error if needed."""
    messages, info = build_messages(REWRITE_RULES, "", f"Q: {question}", model_id, policy)
    meta = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cost_usd": 0.0}
    for attempt in range(2):
        reply = model.invoke(messages)
        usage = getattr(reply, "usage_metadata", None) or {}
        read, _ = cache_counts(usage)
        meta["calls"] += 1
        meta["input_tokens"] += usage.get("input_tokens", 0)
        meta["output_tokens"] += usage.get("output_tokens", 0)
        meta["cache_read"] += read
        meta["cost_usd"] += cost_usd(usage, info["family"], in_price, out_price)
        text = reply.content if isinstance(reply.content, str) else "".join(
            b.get("text", "") for b in reply.content if isinstance(b, dict))
        try:
            return parse_plan(text), meta
        except (ValueError, ValidationError) as e:
            if attempt == 1:
                raise
            # Send the error back so the model can fix its own JSON.
            from langchain_core.messages import AIMessage, HumanMessage
            messages = messages + [AIMessage(content=text),
                                   HumanMessage(content=f"That JSON was invalid: {str(e)[:300]}. Return only the corrected JSON.")]
