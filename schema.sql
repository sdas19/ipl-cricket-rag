-- Run once in Supabase: SQL Editor -> paste -> Run.

-- pgvector adds the "vector" column type and distance operators.
-- It stores and compares numbers only; it has no embedding model.
create extension if not exists vector;

create table if not exists chunks (
  id           bigserial primary key,

  -- Label columns: where this chunk came from.
  -- Used three ways: filters at query time, citations, and eval labels.
  match_id     text not null,           -- Cricsheet file name, e.g. '1473511'
  match_date   date,
  season       text,                    -- '2025'
  competition  text,                    -- 'Indian Premier League'
  stage        text,                    -- 'Final', 'Qualifier 1', or null
  venue        text,
  teams        text[],
  level        text not null,           -- 'over' or 'match'
  innings      int  not null,           -- 1 or 2 (0 for a match-level chunk)
  over_start   int  not null,           -- 1-based: over 18 = balls 17.1 to 17.6
  over_end     int  not null,
  phase        text,                    -- powerplay / middle / death
  players      text[],                  -- names, for display
  player_ids   text[],                  -- Cricsheet registry ids, for filters

  -- What gets embedded and shown to the model: context header + ball lines.
  content      text not null,

  -- Keyword side of hybrid search (BM25-style). 'simple' = no stemming,
  -- which keeps player names intact.
  tsv          tsvector generated always as (to_tsvector('simple', content)) stored,

  -- Meaning side. 1536 numbers = openai/text-embedding-3-small.
  embedding    vector(1536),
  embed_model  text,                    -- which model made the vector

  -- One row per (match, level, innings, over): re-running ingest updates
  -- the row instead of duplicating it.
  unique (match_id, level, innings, over_start)
);

-- Filter indexes (B-tree / GIN): "which match / which date / which player".
create index if not exists chunks_match_idx   on chunks (match_id, innings, over_start);
create index if not exists chunks_date_idx    on chunks (match_date);
create index if not exists chunks_tsv_idx     on chunks using gin (tsv);
create index if not exists chunks_players_idx on chunks using gin (player_ids);

-- Vector index (HNSW). Not needed for one season (~3,000 rows: exact search
-- is milliseconds). We add it later to compare exact vs approximate search.
-- create index chunks_emb_hnsw on chunks using hnsw (embedding vector_cosine_ops);
