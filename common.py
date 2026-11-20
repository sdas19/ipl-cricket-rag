"""Shared setup: settings, the embedding model, and the database connection."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")  # always this folder's .env, wherever the script is run from


def setting(name):
    """Read a required setting, with a clear message if it's missing."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(
            f"{name} is not set. Create .env in {PROJECT_DIR} "
            f"(copy .env.example) and fill it in."
        )
    return value

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "openai/text-embedding-3-small")

# The chat model that writes answers (Lesson 3). Copy the exact id from
# openrouter.ai/models; prices are $ per 1M tokens, from the same page.
CHAT_MODEL = os.environ.get("CHAT_MODEL", "anthropic/claude-haiku-5.5")
CHAT_IN_PRICE = float(os.environ.get("CHAT_IN_PRICE", "0"))
CHAT_OUT_PRICE = float(os.environ.get("CHAT_OUT_PRICE", "0"))


def get_embeddings():
    """The embedding model. The SAME one must embed the chunks (ingest.py)
    and the questions (ask.py), or the search compares two different
    'meaning maps' and quietly returns wrong chunks."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        base_url=OPENROUTER_BASE_URL,            # OpenRouter speaks the OpenAI API format
        api_key=setting("OPENROUTER_API_KEY"),
        # LangChain normally splits text into OpenAI token ids itself before
        # sending. Non-OpenAI endpoints expect plain text, so switch that off.
        check_embedding_ctx_length=False,
        chunk_size=100,                          # texts per API call when embedding many
        # Never wait without a limit: the client's default is minutes per try.
        timeout=20,                              # give up on one request after 20 s
        max_retries=2,                           # then retry at most twice
    )


def get_chat_model(max_tokens=1000):
    """The answer-writing model, via OpenRouter."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=CHAT_MODEL,
        base_url=OPENROUTER_BASE_URL,
        api_key=setting("OPENROUTER_API_KEY"),
        temperature=0,           # same sources -> same answer: easier to test
        # Room for hidden reasoning + the answer. Reasoning tokens count too:
        # at 400 a Hinglish answer was cut off mid-word.
        max_tokens=max_tokens,
        stream_usage=True,       # token counts (incl. cache) arrive with the stream
        timeout=60,
        max_retries=2,
    )


def connect():
    import psycopg

    conn = psycopg.connect(
        setting("DATABASE_URL"),
        connect_timeout=10,          # fail fast instead of hanging
        keepalives=1,                # detect a dropped connection
        keepalives_idle=30,
        prepare_threshold=None,      # no prepared statements: safest through Supabase's pooler
    )
    # Waiting on another transaction's lock for >15 s is a stuck session,
    # not slow work: stop with a clear "lock timeout" instead of hanging.
    conn.execute("set lock_timeout = '15s'")
    conn.execute("set statement_timeout = '120s'")
    return conn


def to_pgvector(vec):
    """Python list of floats -> pgvector text format '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"
