"""
INDEXING phase: match files -> chunks -> embeddings -> Supabase (pgvector).

  python ingest.py --season 2025 --dry-run   # free: just shows chunks + cost
  python ingest.py --season 2025 --limit 3   # 3 matches, to test
  python ingest.py --season 2025             # the whole season (safe to re-run)

Works in batches of matches: embed a batch, save it, commit, move on.
A crash loses at most one batch, and re-running skips matches already saved.
"""

import argparse
import time
from pathlib import Path

import psycopg

from chunker import chunk_match, load_match
from common import EMBED_MODEL, connect, get_embeddings, to_pgvector

UPSERT = """
insert into chunks (match_id, match_date, season, competition, stage, venue, teams,
                    level, innings, over_start, over_end, phase, players, player_ids,
                    content, embedding, embed_model)
values (%(match_id)s, %(match_date)s, %(season)s, %(competition)s, %(stage)s, %(venue)s, %(teams)s,
        %(level)s, %(innings)s, %(over_start)s, %(over_end)s, %(phase)s, %(players)s, %(player_ids)s,
        %(content)s, %(embedding)s::vector, %(embed_model)s)
on conflict (match_id, level, innings, over_start) do update set
  content = excluded.content, embedding = excluded.embedding, embed_model = excluded.embed_model,
  players = excluded.players, player_ids = excluded.player_ids, phase = excluded.phase
"""

# A match is "done" once its match-level chunk is saved with this model.
# Each batch is saved in one transaction, so all of a match's rows or none.
DONE = "select match_id from chunks where level = 'match' and embed_model = %s"


def collect_matches(data_dir, season, limit):
    """Read every match file of that season -> list of (match_id, chunks)."""
    matches = []
    for path in sorted(Path(data_dir).glob("*.json")):
        data, match_id = load_match(path)
        if season and not str(data["info"].get("season", "")).startswith(season):
            continue
        matches.append((match_id, chunk_match(data, match_id)))
        if limit and len(matches) >= limit:
            break
    return matches


def save_batch(rows, attempts=3):
    """Save one batch in one transaction; reconnect and retry if the connection drops."""
    for attempt in range(1, attempts + 1):
        try:
            with connect() as conn, conn.cursor() as cur:
                cur.executemany(UPSERT, rows)      # bulk: far fewer round trips than row by row
            return                                  # leaving the 'with' block commits
        except psycopg.OperationalError as e:
            if attempt == attempts:
                raise
            print(f"  connection problem ({str(e).splitlines()[0]}), retrying in {2 * attempt}s...")
            time.sleep(2 * attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/ipl_json")
    ap.add_argument("--season", default="2025", help="e.g. 2025; Cricsheet's 'season' field ('' = all)")
    ap.add_argument("--limit", type=int, default=0, help="max matches (0 = all)")
    ap.add_argument("--batch", type=int, default=10, help="matches per batch")
    ap.add_argument("--dry-run", action="store_true", help="no API calls, no database")
    args = ap.parse_args()

    matches = collect_matches(args.data_dir, args.season, args.limit)
    all_chunks = [c for _, cs in matches for c in cs]
    est_tokens = sum(len(c["content"]) for c in all_chunks) // 4   # ~4 characters per token
    print(f"{len(matches)} matches -> {len(all_chunks)} chunks, ~{est_tokens:,} tokens "
          f"(~${est_tokens * 0.02 / 1e6:.4f} at $0.02 per 1M tokens)")

    if args.dry_run:
        for c in all_chunks[:1] + all_chunks[-1:]:
            print(f"\n--- {c['match_id']} | {c['level']} | innings {c['innings']} | over {c['over_start']} ---")
            print(c["content"])
        return

    # Resume: skip matches already saved with this embedding model.
    with connect() as conn, conn.cursor() as cur:
        cur.execute(DONE, (EMBED_MODEL,))
        done = {r[0] for r in cur.fetchall()}
    todo = [(mid, cs) for mid, cs in matches if mid not in done]
    if done:
        print(f"Skipping {len(matches) - len(todo)} matches already saved.")
    if not todo:
        print("Nothing to do.")
        return

    emb = get_embeddings()
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    saved_matches = len(matches) - len(todo)
    t_start = time.perf_counter()

    for n, batch in enumerate(batches, start=1):
        chunks = [c for _, cs in batch for c in cs]
        vectors = emb.embed_documents([c["content"] for c in chunks])     # 1) embed this batch
        rows = [{**c, "embedding": to_pgvector(v), "embed_model": EMBED_MODEL}
                for c, v in zip(chunks, vectors)]
        save_batch(rows)                                                   # 2) save + commit
        saved_matches += len(batch)
        print(f"batch {n}/{len(batches)}: {len(batch)} matches, {len(rows)} chunks saved "
              f"({saved_matches}/{len(matches)} matches done)")

    print(f"Finished in {time.perf_counter() - t_start:.0f}s.")


if __name__ == "__main__":
    main()
