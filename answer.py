"""
The "A" and "G" of RAG: put the retrieved chunks into a prompt and have a
chat model answer ONLY from them, citing its sources.

Two modes:
  rag   (default) retrieve the top-k chunks for the question, send those.
  match send the WHOLE match (~6k tokens) as a stable, cacheable context.
        Repeated questions about the same match reuse the cached prefix.

  python answer.py "Who won the 2025 final and by how much?"
  python answer.py "How did Virat Kohli get out?" --match 1473511
  python answer.py "Ashok Dinda bowling"
  python answer.py "Who bowled the last over?" --mode match --match 1473511
  python answer.py "Who bowled the last over?" --mode match --match 1473511 --no-cache
"""

import argparse
import re
import time

from caching import CachePolicy, build_messages, cache_counts, cost_usd
from common import CHAT_IN_PRICE, CHAT_MODEL, CHAT_OUT_PRICE, connect, get_chat_model

IDK = "I don't know based on the available match data."

# Stable instructions: identical on every call (so they can be cached too).
RULES = f"""You answer questions about IPL cricket matches using ONLY the numbered sources provided.

Rules:
1. Use only facts stated in the sources. Never add facts from your own cricket knowledge, even if you are sure.
2. Cite every fact with its source id in square brackets, e.g. [S3]. Cite only ids that appear in the sources.
3. Copy numbers (runs, balls, overs, scores, wickets) exactly as written.
4. First check the sources are about the match the question asks about (match name, stage such as "Final", teams and date are in each source's header). If the question names a match and no source is from it, reply with exactly: "{IDK}" Then say the sources are from other matches.
5. If the sources do not contain the answer, reply with exactly: "{IDK}" Then, in one short sentence, say what is missing.
6. If the question needs a total, ranking or comparison across many matches or a whole season (e.g. most sixes, highest average), and the sources only show some overs, reply with exactly: "{IDK}" Then say these sources are only a sample of overs. Do not list partial figures from the sample. Exception: a source that starts with "Computed from all ball-by-ball data" is complete for the scope it states; answer totals and rankings from it directly.
7. Answer in the same language style as the question (English, Hindi or Hinglish), in 1-4 sentences.

How to read the sources:
- Ball labels count completed overs: "14.5" is the 5th ball of the 15th over, and balls 18.1 to 18.6 make up the 19th over (shown as "Over 19" in headers). This is normal cricket notation, not an inconsistency. Over numbers in headers are 1-based. The last over of an innings is the highest over number in it; an innings can end before over 20 (a chase completed early, a team bowled out).
- "c X b Y" = caught by X off bowler Y; "c & b Y" = caught and bowled by Y; "lbw b Y" = leg before wicket to Y.
- "Innings 2 ... chasing N" means the second team needed N to win."""

COLUMNS = "match_id, level, innings, over_start, over_end, content"


def key_of(row):
    mid, level, inn, o1, o2 = row[:5]
    if level == "stats":                     # a computed table: its "key" is what was computed
        return mid
    return f"{mid} | match" if level == "match" else f"{mid} | {inn} | {o1}-{o2}"


def fetch_rag(question, match, k):
    """Retrieval exactly as in ask.py: embed the question, nearest chunks."""
    from ask import SEARCH, embed_question
    from common import EMBED_MODEL, to_pgvector

    vec, _ = embed_question(question)
    with connect() as conn, conn.cursor() as cur:
        cur.execute(SEARCH, {"q": to_pgvector(vec), "model": EMBED_MODEL, "match": match, "k": k})
        return [r[:6] for r in cur.fetchall()]


def fetch_match(match_id):
    """Every chunk of one match, in reading order: summary first, then each innings over by over."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"select {COLUMNS} from chunks where match_id = %s "
            "order by (level = 'match') desc, innings, over_start",
            (match_id,),
        )
        return cur.fetchall()


def format_sources(rows, compact=False):
    """Number the chunks [S1], [S2]... and remember which key each number points to.
    compact=True (one match): print the match title once instead of on every chunk."""
    labels, parts = {}, []
    title = rows[0][5].splitlines()[0] if (compact and rows) else None
    if title:
        parts.append(title)
    for i, row in enumerate(rows, start=1):
        sid = f"S{i}"
        labels[sid] = key_of(row)
        text = row[5]
        if title and text.startswith(title):
            text = text[len(title):].lstrip("\n")
        parts.append(f"[{sid}]\n{text}")
    return "\n\n".join(parts), labels


def parse_citations(answer, labels):
    """[S2] -> its key. Ids the model invented (not in the sources) are flagged."""
    ids = list(dict.fromkeys(re.findall(r"\[(S\d+)\]", answer)))
    return [(i, labels[i]) for i in ids if i in labels], [i for i in ids if i not in labels]


def run_answer(question, rows, mode, model, model_id, policy):
    """Build the prompt, stream the answer, measure everything."""
    sources, labels = format_sources(rows, compact=(mode == "match"))
    if mode == "match":
        # The whole match is the same for every question about it: stable, cacheable.
        stable, dynamic = "Sources:\n\n" + sources, f"Question: {question}"
    else:
        # Retrieved chunks change with every question: they go in the changing part.
        stable, dynamic = "", f"Sources:\n\n{sources}\n\nQuestion: {question}"
    messages, info = build_messages(RULES, stable, dynamic, model_id, policy)

    start, ttft, full = time.perf_counter(), None, None
    for chunk in model.stream(messages):
        if ttft is None and chunk.content:
            ttft = (time.perf_counter() - start) * 1000
        full = chunk if full is None else full + chunk
    total = (time.perf_counter() - start) * 1000

    text = full.content if full else ""
    if isinstance(text, list):
        text = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in text)
    usage = getattr(full, "usage_metadata", None) or {}
    read, write = cache_counts(usage)
    # Why the output stopped: "stop" = finished, "length" = hit max_tokens (cut off).
    finish = (getattr(full, "response_metadata", None) or {}).get("finish_reason")
    cited, invented = parse_citations(text, labels)
    return {
        "answer": text.strip(),
        "cited": cited,
        "invented_citations": invented,
        "said_idk": IDK.lower()[:12] in text.lower(),
        "finish_reason": finish,
        "truncated": finish == "length",
        "n_sources": len(rows),
        "cache_mode": info["mode"],
        "family": info["family"],
        "input_tokens": usage.get("input_tokens", 0),
        "cache_read": read,
        "cache_write": write,
        "output_tokens": usage.get("output_tokens", 0),
        "ttft_ms": ttft if ttft is not None else total,
        "total_ms": total,
        "cost_usd": cost_usd(usage, info["family"], CHAT_IN_PRICE, CHAT_OUT_PRICE),
        "prompt": messages,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--mode", choices=["rag", "match"], default="rag")
    ap.add_argument("--match", help="match_id: filter for rag mode, required for match mode")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-cache", action="store_true", help="don't request prompt caching")
    ap.add_argument("--show-prompt", action="store_true", help="print exactly what the model receives")
    args = ap.parse_args()

    if args.mode == "match":
        if not args.match:
            ap.error("--mode match needs --match <match_id> (see: python ask.py --list-matches)")
        rows = fetch_match(args.match)
    else:
        rows = fetch_rag(args.question, args.match, args.k)
    if not rows:
        print("No chunks found.")
        return

    r = run_answer(args.question, rows, args.mode, get_chat_model(), CHAT_MODEL,
                   CachePolicy(enabled=not args.no_cache))

    if args.show_prompt:
        for m in r["prompt"]:
            content = m.content if isinstance(m.content, str) else "\n".join(
                b["text"] + ("\n   <-- cache breakpoint" if "cache_control" in b else "") for b in m.content)
            print(f"===== {m.type} =====\n{content}\n")

    print(f"\nQ: {args.question}\n\n{r['answer']}\n")
    if r["truncated"]:
        print("  WARNING: answer was cut off (hit max_tokens). Raise max_tokens in get_chat_model().\n")
    if r["cited"]:
        print("Sources cited:")
        for sid, key in r["cited"]:
            print(f"  [{sid}] {key}")
    if r["invented_citations"]:
        print(f"  WARNING: cited ids not in the sources: {r['invented_citations']}")
    if not r["cited"] and not r["said_idk"]:
        print("  WARNING: no citations")
    print(f"\n{args.mode} mode · {r['n_sources']} sources · {CHAT_MODEL} · cache {r['cache_mode']}")
    print(f"tokens: input {r['input_tokens']:,} (cache read {r['cache_read']:,}, "
          f"cache write {r['cache_write']:,}) · output {r['output_tokens']:,}")
    price = f"${r['cost_usd']:.6f}" if CHAT_IN_PRICE else "set CHAT_IN_PRICE / CHAT_OUT_PRICE in .env"
    print(f"TTFT {r['ttft_ms']:.0f} ms · total {r['total_ms']:.0f} ms · cost {price}")


if __name__ == "__main__":
    main()
