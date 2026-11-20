"""Tests for caching.py and answer.py, with a fake chat model (free, no network).
Run:  python -m pytest -q"""

import os

from langchain_core.messages import AIMessageChunk

from answer import RULES, format_sources, parse_citations, run_answer
from caching import CachePolicy, build_messages, cost_usd, model_family
from chunker import chunk_match
from test_chunker import MATCH

ROWS = [(c["match_id"], c["level"], c["innings"], c["over_start"], c["over_end"], c["content"])
        for c in chunk_match(MATCH, "1473511")]


class FakeChat:
    """Streams a fixed answer; the last chunk carries token usage, like a real provider."""
    def __init__(self, text, usage, finish="stop"):
        self.text, self.usage, self.finish, self.seen = text, usage, finish, None

    def stream(self, messages):
        self.seen = messages
        words = self.text.split(" ")
        for i, w in enumerate(words):
            last = i == len(words) - 1
            usage = None
            if last:
                usage = {**self.usage, "total_tokens": self.usage["input_tokens"] + self.usage["output_tokens"]}
            yield AIMessageChunk(content=w + ("" if last else " "), usage_metadata=usage,
                                 response_metadata={"finish_reason": self.finish} if last else {})


# ---- caching.py -------------------------------------------------------------

def test_model_family():
    assert model_family("anthropic/claude-haiku-5.5") == "anthropic"
    assert model_family("openai/gpt-5-mini") == "openai"
    assert model_family("google/gemini-3-flash") == "google"
    assert model_family("deepseek/deepseek-v4-flash") == "deepseek"
    assert model_family("meta-llama/llama-4") == "unknown"


def test_anthropic_gets_marker_on_last_stable_block():
    msgs, info = build_messages("RULES", "WHOLE MATCH", "Q?", "anthropic/claude-haiku-5.5", CachePolicy())
    blocks = msgs[0].content
    assert info == {"family": "anthropic", "mode": "explicit"}
    assert [b["text"] for b in blocks] == ["RULES", "WHOLE MATCH"]
    assert "cache_control" not in blocks[0]
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}
    assert msgs[1].content == "Q?"                       # the changing part stays after the breakpoint


def test_one_hour_ttl_and_rules_only():
    msgs, _ = build_messages("RULES", "", "Q?", "anthropic/x", CachePolicy(ttl="1h"))
    assert msgs[0].content[-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_automatic_off_and_unsupported_are_plain_text():
    for model, mode in [("openai/gpt-5-mini", "automatic"), ("meta-llama/llama-4", "unsupported")]:
        msgs, info = build_messages("RULES", "CTX", "Q?", model, CachePolicy())
        assert info["mode"] == mode and msgs[0].content == "RULES\n\nCTX"
    _, info = build_messages("RULES", "CTX", "Q?", "anthropic/x", CachePolicy(enabled=False))
    assert info["mode"] == "off"


def test_marker_survives_langchain_to_openrouter():
    from langchain_openai import ChatOpenAI
    msgs, _ = build_messages("RULES", "CTX", "Q?", "anthropic/x", CachePolicy())
    body = ChatOpenAI(model="anthropic/x", api_key="k", base_url="https://openrouter.ai/api/v1") \
        ._get_request_payload(msgs)
    assert body["messages"][0]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_cache_aware_cost():
    # 6,000 input tokens: 5,800 read from cache, 200 fresh; 40 output. $1 in / $5 out.
    usage = {"input_tokens": 6000, "output_tokens": 40,
             "input_token_details": {"cache_read": 5800, "cache_creation": 0}}
    hit = cost_usd(usage, "anthropic", 1.0, 5.0)
    assert abs(hit - (200 * 1 + 5800 * 0.1 + 40 * 5) / 1e6) < 1e-12
    miss = cost_usd({"input_tokens": 6000, "output_tokens": 40}, "anthropic", 1.0, 5.0)
    assert hit < miss / 4                                # cached call is >4x cheaper
    write = cost_usd({"input_tokens": 6000, "output_tokens": 0,
                      "input_token_details": {"cache_read": 0, "cache_creation": 6000}}, "anthropic", 1.0, 0)
    assert round(write * 1e6) == 7500                    # first call pays 1.25x


# ---- answer.py --------------------------------------------------------------

def test_sources_are_numbered_and_compact_mode_prints_title_once():
    text, labels = format_sources(ROWS)
    assert labels == {"S1": "1473511 | 1 | 1-1", "S2": "1473511 | 2 | 18-18", "S3": "1473511 | match"}
    title = ROWS[0][5].splitlines()[0]
    assert text.count(title) == 3
    compact, _ = format_sources(ROWS, compact=True)
    assert compact.count(title) == 1 and compact.startswith(title)


def test_citations_map_to_keys_and_invented_ids_are_flagged():
    _, labels = format_sources(ROWS)
    cited, invented = parse_citations("Kohli was out [S1]. RCB won [S3][S1]. Also [S9].", labels)
    assert cited == [("S1", "1473511 | 1 | 1-1"), ("S3", "1473511 | match")]
    assert invented == ["S9"]


def test_rag_mode_puts_sources_in_the_changing_part():
    fake = FakeChat("RCB won by 6 runs [S3].", {"input_tokens": 900, "output_tokens": 9})
    r = run_answer("Who won?", ROWS, "rag", fake, "anthropic/x", CachePolicy())
    system, user = fake.seen
    assert "[S1]" not in "".join(b["text"] for b in system.content)   # system = rules only
    assert user.content.startswith("Sources:") and user.content.endswith("Question: Who won?")
    assert r["cited"] == [("S3", "1473511 | match")] and not r["said_idk"]
    assert r["ttft_ms"] <= r["total_ms"]


def test_match_mode_caches_the_whole_match():
    usage = {"input_tokens": 6000, "output_tokens": 30,
             "input_token_details": {"cache_read": 5900, "cache_creation": 0}}
    fake = FakeChat("JR Hazlewood bowled it [S2].", usage)
    r = run_answer("Who bowled the last over?", ROWS, "match", fake, "anthropic/x", CachePolicy())
    system, user = fake.seen
    assert system.content[0]["text"] == RULES
    assert "[S2]" in system.content[1]["text"] and "cache_control" in system.content[1]
    assert user.content == "Question: Who bowled the last over?"
    assert (r["cache_mode"], r["cache_read"], r["cache_write"]) == ("explicit", 5900, 0)


def test_idk_is_detected():
    fake = FakeChat("I don't know based on the available match data. Dinda is not in the sources.",
                    {"input_tokens": 800, "output_tokens": 15})
    r = run_answer("Ashok Dinda bowling", ROWS, "rag", fake, "anthropic/x", CachePolicy())
    assert r["said_idk"] and r["cited"] == []


def test_cut_off_answer_is_flagged():
    fake = FakeChat("Final ke aakhri over mein M Pathirana ne", {"input_tokens": 2000, "output_tokens": 400},
                    finish="length")
    r = run_answer("final me last over?", ROWS, "rag", fake, "anthropic/x", CachePolicy())
    assert r["truncated"] and r["finish_reason"] == "length"
    ok = run_answer("Who won?", ROWS, "rag", FakeChat("RCB [S3].", {"input_tokens": 9, "output_tokens": 3}),
                    "anthropic/x", CachePolicy())
    assert not ok["truncated"]


def test_chat_model_has_room_for_reasoning(monkeypatch):
    import common
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert common.get_chat_model().max_tokens == 1000


def test_rules_cover_wrong_match_and_samples():
    assert "no source is from it" in RULES
    assert "Do not list partial figures" in RULES
