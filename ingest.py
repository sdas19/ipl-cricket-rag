"""
INDEXING phase: match files -> chunks -> embeddings -> Supabase (pgvector).

  python ingest.py --data-dir data/ipl_json --season 2025 --dry-run   # free: just shows chunks + cost
  python ingest.py --data-dir data/ipl_json --season 2025 --limit 3   # 3 matches, to test
  python ingest.py --data-dir data/ipl_json --season 2025             # the whole season
"""

import argparse
import time
from pathlib import Path

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


def collect_chunks(data_dir, season, limit):
    """Read every match file of that season and chunk it."""
    chunks, matches = [], 0
    for path in sorted(Path(data_dir).glob("*.json")):
        data, match_id = load_match(path)
        if season and not str(data["info"].get("season", "")).startswith(season):
            continue
        chunks += chunk_match(data, match_id)
        matches += 1
        if limit and matches >= limit:
            break
    return chunks, matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/ipl_json")
    ap.add_argument("--season", default="2025", help="e.g. 2025; Cricsheet's 'season' field")
    ap.add_argument("--limit", type=int, default=0, help="max matches (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="no API calls, no database")
    args = ap.parse_args()

    chunks, matches = collect_chunks(args.data_dir, args.season, args.limit)
    est_tokens = sum(len(c["content"]) for c in chunks) // 4   # ~4 characters per token
    print(f"{matches} matches -> {len(chunks)} chunks, ~{est_tokens:,} tokens "
          f"(~${est_tokens * 0.02 / 1e6:.4f} at $0.02 per 1M tokens)")

    if args.dry_run:
        for c in chunks[:1] + chunks[-1:]:
            print(f"\n--- {c['match_id']} | {c['level']} | innings {c['innings']} | over {c['over_start']} ---")
            print(c["content"])
        return

    # 1) Embed every chunk's text (batched: 100 texts per API call).
    t0 = time.perf_counter()
    vectors = get_embeddings().embed_documents([c["content"] for c in chunks])
    print(f"Embedded {len(vectors)} chunks in {time.perf_counter() - t0:.1f}s "
          f"({len(vectors[0])} numbers each)")

    # 2) Store chunk + labels + vector in one row each.
    with connect() as conn, conn.cursor() as cur:
        for c, v in zip(chunks, vectors):
            cur.execute(UPSERT, {**c, "embedding": to_pgvector(v), "embed_model": EMBED_MODEL})
    print(f"Saved {len(chunks)} rows to the chunks table.")


if __name__ == "__main__":
    main()
