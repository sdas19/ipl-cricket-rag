"""
QUERYING phase (retrieval only, no answer yet): question -> embedding ->
nearest chunks. Shows exactly what the model would be given.

  python ask.py --list-matches                                   # find match ids
  python ask.py "Who won the final and by how much?"
  python ask.py "How did Kohli get out?" --match 1473511          # filter first, then rank
  python ask.py "Bumrah ne death overs me kaisa daala?" --k 5
"""

import argparse
import json
import time

from common import EMBED_MODEL, PROJECT_DIR, connect, get_embeddings, to_pgvector

CACHE_FILE = PROJECT_DIR / ".cache" / "query_embeddings.json"


def embed_question(question, use_cache=True):
    """Embed the question, reusing a saved vector for a repeated question.
    The key includes the model, so a different model never reuses old vectors."""
    key = f"{EMBED_MODEL}::{question.strip().lower()}"
    cache = {}
    if use_cache and CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text())
        if key in cache:
            return cache[key], True
    t0 = time.perf_counter()
    emb = get_embeddings()                      # imports LangChain + creates the client (setup)
    embed_question.setup_ms = (time.perf_counter() - t0) * 1000
    vec = emb.embed_query(question)             # the actual network call
    if use_cache:
        cache[key] = vec
        CACHE_FILE.parent.mkdir(exist_ok=True)
        CACHE_FILE.write_text(json.dumps(cache))
    return vec, False

SEARCH = """
select match_id, level, innings, over_start, over_end, content,
       embedding <=> %(q)s::vector as distance          -- cosine distance: 0 = same meaning
from chunks
where embed_model = %(model)s                            -- never compare across models
  and (%(match)s::text is null or match_id = %(match)s)  -- optional filter: one match
order by distance
limit %(k)s
"""

LIST = """
select match_id, match_date, coalesce(stage, ''), array_to_string(teams, ' vs ')
from chunks where level = 'match' order by match_date
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--match", help="only search this match_id")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--list-matches", action="store_true")
    ap.add_argument("--full", action="store_true", help="print each chunk in full (exactly what a model would get)")
    ap.add_argument("--no-cache", action="store_true", help="always call the embedding model (to measure latency)")
    args = ap.parse_args()

    with connect() as conn, conn.cursor() as cur:
        if args.list_matches:
            cur.execute(LIST)
            for mid, date, stage, teams in cur.fetchall():
                print(f"{mid}  {date}  {stage:<12} {teams}")
            return
        if not args.question:
            ap.error("give a question, or --list-matches")

        # The question goes through the SAME embedding model as the chunks.
        embed_question.setup_ms = 0.0
        t0 = time.perf_counter()
        q_vec, cached = embed_question(args.question, use_cache=not args.no_cache)
        t_setup = embed_question.setup_ms
        t_embed = (time.perf_counter() - t0) * 1000 - t_setup   # the API call only

        t0 = time.perf_counter()
        cur.execute(SEARCH, {"q": to_pgvector(q_vec), "model": EMBED_MODEL,
                             "match": args.match, "k": args.k})
        rows = cur.fetchall()
        t_search = (time.perf_counter() - t0) * 1000

    timing = "embed 0 ms (cached)" if cached else f"setup {t_setup:.0f} ms, embed call {t_embed:.0f} ms"
    print(f'\nQ: {args.question}   ({timing}, search {t_search:.0f} ms)\n')
    for rank, (mid, level, inn, o1, o2, content, dist) in enumerate(rows, start=1):
        key = f"{mid} | {'match' if level == 'match' else f'{inn} | {o1}-{o2}'}"
        lines = content.splitlines()
        print(f"#{rank}  distance {dist:.3f}   key {key}")
        if args.full:
            show = lines                                   # everything the model would see
        elif level == "over":
            # Header + the over's key moments (wickets, boundaries) + end-of-over score,
            # so a preview never hides the line that answers the question.
            balls = lines[3:-1]
            moments = [ln for ln in balls if any(w in ln for w in ("OUT!", "SIX", "FOUR"))]
            show = lines[:2] + (moments or balls[:3]) + lines[-1:]
        else:
            show = lines[:4]
        for ln in show:
            print("    " + ln)
        print()


if __name__ == "__main__":
    main()
