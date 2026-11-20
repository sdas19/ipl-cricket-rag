"""
Provider-neutral prompt caching.

The caller says WHAT it wants ("cache the stable part") and passes the
stable part and the changing part separately. This module turns that into
whatever each model family needs:

  anthropic -> explicit: a cache_control marker on the last stable block
  openai, google, deepseek -> automatic: nothing to add, just keep stable first
  anything else -> unsupported: sent plainly, nothing cached

Caching can be REQUESTED, never guaranteed (prompt below the provider's
minimum length, cache expired, request routed to another provider...).
So every call reports both what was requested (mode) and what actually
happened (cache_read / cache_write tokens from the response).
"""

from dataclasses import dataclass
from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage

MODES = {"anthropic": "explicit", "openai": "automatic", "google": "automatic", "deepseek": "automatic"}

# Price of cached tokens relative to normal input tokens. Approximate:
# check the provider's pricing page for the model you use.
#   read  = tokens served from cache      write = tokens stored into cache
MULTIPLIERS = {
    "anthropic": {"read": 0.10, "write": 1.25},   # 5-minute cache; 1-hour writes cost 2x
    "openai": {"read": 0.50, "write": 1.00},
    "google": {"read": 0.25, "write": 1.00},
    "deepseek": {"read": 0.10, "write": 1.00},
}


@dataclass
class CachePolicy:
    enabled: bool = True
    ttl: Literal["5m", "1h"] = "5m"


def model_family(model_id: str) -> str:
    """'anthropic/claude-haiku-5.5' -> 'anthropic'. Works with or without the
    OpenRouter 'vendor/' prefix."""
    m = model_id.lower()
    if m.startswith("anthropic/") or "claude" in m:
        return "anthropic"
    if m.startswith("openai/") or m.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if m.startswith("google/") or "gemini" in m:
        return "google"
    if m.startswith("deepseek/") or "deepseek" in m:
        return "deepseek"
    return "unknown"


def build_messages(rules: str, stable: str, dynamic: str, model_id: str, policy: CachePolicy):
    """System message = rules + stable context (same on every call: cacheable).
    User message      = the part that changes on every call.
    Returns (messages, info) where info says what caching was requested."""
    family = model_family(model_id)
    mode = MODES.get(family, "unsupported") if policy.enabled else "off"

    if mode == "explicit":
        marker = {"type": "ephemeral"}
        if policy.ttl == "1h":
            marker["ttl"] = "1h"
        blocks = [{"type": "text", "text": rules}]
        if stable:
            blocks.append({"type": "text", "text": stable})
        blocks[-1]["cache_control"] = marker      # breakpoint: everything up to here is reusable
        system = SystemMessage(content=blocks)
    else:
        system = SystemMessage(content=rules + ("\n\n" + stable if stable else ""))

    return [system, HumanMessage(content=dynamic)], {"family": family, "mode": mode}


def cache_counts(usage: Optional[dict]):
    """LangChain normalises every provider's report into
    usage_metadata['input_token_details']: cache_read and cache_creation."""
    details = (usage or {}).get("input_token_details") or {}
    return details.get("cache_read", 0) or 0, details.get("cache_creation", 0) or 0


def cost_usd(usage: Optional[dict], family: str, in_price: float, out_price: float) -> float:
    """input_tokens INCLUDES cached tokens, so split it into fresh / read / written,
    each with its own price. Prices are $ per 1M tokens."""
    usage = usage or {}
    read, write = cache_counts(usage)
    fresh = max(0, usage.get("input_tokens", 0) - read - write)
    mult = MULTIPLIERS.get(family, {"read": 1.0, "write": 1.0})
    return (fresh * in_price
            + read * in_price * mult["read"]
            + write * in_price * mult["write"]
            + usage.get("output_tokens", 0) * out_price) / 1_000_000
