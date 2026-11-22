"""
Search modes for the router's search route:

  vector         meaning only (what we had)
  keyword        exact words only (the tsv column)
  hybrid         both, merged with Reciprocal Rank Fusion (RRF)
  hybrid_rerank  hybrid candidates, then a reranker reads question + chunk together

Pure helpers here (keyword query, RRF, rerank call); the SQL lives in router.Database.
"""

import re

MODES = ("vector", "keyword", "hybrid", "hybrid_rerank")

# Words that carry no search signal. Keyword search should look for what makes
# a chunk special ("kohli", "out", "omarzai"), not "how", "did", "the".
STOPWORDS = set("""a an the and or of in on at to for from by with is was were be been did do does how what who
whom which when where why this that these those it its his her their there me my we our you your i
ka ki ke me mein ne se ko hai tha thi the kya kaise kaisa kisne kaun""".split())


# Query expansion: the data's own words for what users say. Keyword search only
# matches exact words, and the chunks always write "OUT!", never "dismissed".
SYNONYMS = {"dismissed": "out", "dismissal": "out", "wicket": "out", "wickets": "out",
            "sixes": "six", "chhakke": "six", "chhakka": "six", "fours": "four", "chauke": "four",
            "boundaries": "four six", "aakhri": "last"}


def keyword_query(text):
    """'How did Virat Kohli get out in the final?' -> 'virat or kohli or get or out or final'

    websearch_to_tsquery joins plain words with AND: every word must appear,
    so a question's filler words ("how", "did") would match nothing. Joining the
    useful words with OR lets ts_rank reward chunks that contain MORE of them."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    words = [x for w in words for x in SYNONYMS.get(w, w).split()]
    keep = [w for w in words if w not in STOPWORDS and len(w) > 1]
    return " or ".join(dict.fromkeys(keep))          # keep order, drop repeats


def row_key(row):
    return tuple(row[:4])                             # match_id, level, innings, over_start


def rrf(*ranked_lists, k=60):
    """Reciprocal Rank Fusion: score = sum of 1 / (k + rank) over the lists a row appears in.
    A row ranked high in BOTH lists beats one ranked first in only one."""
    scores, rows = {}, {}
    for ranked in ranked_lists:
        for rank, row in enumerate(ranked, start=1):
            key = row_key(row)
            rows[key] = row
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    order = sorted(scores, key=lambda key: -scores[key])
    return [rows[key] for key in order], [scores[key] for key in order]


def rerank(query, rows, top_n, model, api_key, base_url, post=None):
    """Ask a rerank model to score each (question, chunk) pair; return rows in its order.
    -> (rows, scores, cost). `post` can be swapped for a fake in tests."""
    if not rows:
        return [], [], 0.0
    if post is None:
        import httpx

        def post(url, payload):
            r = httpx.post(url, json=payload, timeout=20,
                           headers={"Authorization": f"Bearer {api_key}"})
            r.raise_for_status()
            return r.json()

    data = post(f"{base_url}/rerank", {"model": model, "query": query,
                                       "documents": [row[5] for row in rows], "top_n": top_n})
    results = data["results"]                         # sorted by relevance, each with the input index
    cost = (data.get("usage") or {}).get("cost") or 0.0
    return [rows[r["index"]] for r in results], [r["relevance_score"] for r in results], cost
