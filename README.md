# Cricket RAG (Build 2)

Ask questions about IPL matches. Ball-by-ball data from [Cricsheet](https://cricsheet.org),
chunked per over with a context header, embedded via OpenRouter, stored in Supabase pgvector.

## Files

| File | Phase | What it does |
| --- | --- | --- |
| `schema.sql` | setup | `chunks` table: label columns + content + keyword index + vector column |
| `chunker.py` | indexing | Cricsheet JSON -> over chunks and a match summary chunk (no AI) |
| `common.py` | both | settings, embedding model, database connection |
| `ingest.py` | indexing | chunks -> embeddings -> upsert into Supabase |
| `ask.py` | querying | question -> embedding -> nearest chunks (retrieval only) |
| `caching.py` | answering | provider-neutral prompt caching: one request, converted per model family |
| `answer.py` | answering | sources + question -> chat model -> answer with [S#] citations; `rag` or whole-`match` mode |
| `schema_3b.sql` | setup | `matches`, `deliveries` (one row per ball), `players` tables |
| `ingest_deliveries.py` | indexing | Cricsheet JSON -> matches, deliveries, players (no AI; COPY in one transaction) |
| `resolve.py` | routing | team aliases, player names ("Virat Kohli" -> "V Kohli"), which match |
| `stats.py` | routing | fixed SQL templates for calculated answers (most sixes, economy, a player's season) |
| `rewrite.py` | routing | one cheap model call: question -> validated Plan (route, filters, English text) |
| `router.py` | routing | rewrite -> name check -> match -> lookup / stats / search -> answer |
| `search.py` | routing | keyword query (OR + synonyms), Reciprocal Rank Fusion, reranker call |
| `golden.csv` | evals | 36 questions, 11 types, English + Hinglish, with expected route, match, keys, facts |
| `eval.py` | evals | runs the golden set; scores routing, retrieval and answers by question type |
| `test_chunker.py` | tests | chunker on a hand-made match |
| `test_router.py` | tests | ingest, resolve, stats, rewrite and routing (fakes, no network) |
| `test_eval.py` | tests | search helpers and eval scoring |
| `test_answer.py` | tests | caching adapter, cost maths, citations, prompt layout (fake chat model) |

## Run

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in OPENROUTER_API_KEY and DATABASE_URL
pytest -q
python ingest.py --season 2025 --dry-run
python ingest.py --season 2025
python ask.py "Who won the final and by how much?"
python answer.py "Who won the 2025 final and by how much?"
python answer.py "Who bowled the last over?" --mode match --match 1473511
python ingest_deliveries.py --season 2025
python router.py "final me last over kisne daala aur kitne run bane?" --show-plan
python eval.py --search-mode vector --out vector.json
python eval.py --search-mode hybrid_rerank --out hybrid_rerank.json
```
