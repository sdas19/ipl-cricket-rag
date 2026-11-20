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
| `test_chunker.py` | tests | chunker on a hand-made match |
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
```
