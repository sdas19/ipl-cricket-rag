"""
QUERYING phase (retrieval only, no answer yet): question -> embedding ->
nearest chunks. Shows exactly what the model would be given.

  python ask.py --list-matches                                   # find match ids
  python ask.py "Who won the final and by how much?"
  python ask.py "How did Kohli get out?" --match 1473511          # filter first, then rank
  python ask.py "Bumrah ne death overs me kaisa daala?" --k 5
"""

import argparse
import time

from common import EMBED_MODEL, connect, get_embeddings, to_pgvector

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
        t0 = time.perf_counter()
        q_vec = get_embeddings().embed_query(args.question)
        t_embed = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        cur.execute(SEARCH, {"q": to_pgvector(q_vec), "model": EMBED_MODEL,
                             "match": args.match, "k": args.k})
        rows = cur.fetchall()
        t_search = (time.perf_counter() - t0) * 1000

    print(f'\nQ: {args.question}   (embed {t_embed:.0f} ms, search {t_search:.0f} ms)\n')
    for rank, (mid, level, inn, o1, o2, content, dist) in enumerate(rows, start=1):
        key = f"{mid} | {'match' if level == 'match' else f'{inn} | {o1}-{o2}'}"
        lines = content.splitlines()
        print(f"#{rank}  distance {dist:.3f}   key {key}")
        print("    " + lines[0])
        print("    " + lines[1])
        for ln in lines[3:6] if level == "over" else lines[2:4]:
            print("    " + ln)
        print()


if __name__ == "__main__":
    main()
