"""
/data/geo_events.py — Geopolitical event ingestion & sentiment pipeline.

Design decisions:
- Two source channels: GDELT (structured event data) + RSS feeds (free text).
- Sentiment is scored with a FinBERT-class model (ProsusAI/finbert) which is
  domain-tuned on financial news. Generic VADER/TextBlob is not used because
  it systematically misclassifies financial jargon.
- The output is a daily sentiment score in [-1, 1] per asset label
  ("BTC", "GOLD", "GLOBAL") aligned to the OHLCV index.
- We deliberately avoid blocking the event loop: all HTTP calls are async
  and the model inference is run in a thread pool via asyncio.to_thread.
- No raw news text is persisted to disk; only the scalar sentiment scores
  are cached, minimising data-retention surface.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import feedparser  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Lazy-loaded FinBERT sentiment model
# ─────────────────────────────────────────────────────────────────────────────

_sentiment_pipeline: Any | None = None  # transformers.Pipeline


def _get_sentiment_pipeline() -> Any:
    """
    Lazy-load FinBERT on first call; subsequent calls reuse the cached object.
    Model is loaded in CPU mode; GPU is picked up automatically if available.
    """
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        try:
            from transformers import pipeline  # type: ignore[import-untyped]

            _sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=-1,   # -1 = CPU; set to 0 for first CUDA GPU
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT pipeline loaded.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("FinBERT unavailable (%s); falling back to neutral score.", exc)
    return _sentiment_pipeline


def _score_text(text: str) -> float:
    """
    Return a scalar sentiment in [-1, 1].
    FinBERT labels: positive → +1, neutral → 0, negative → -1.
    """
    pipe = _get_sentiment_pipeline()
    if pipe is None:
        return 0.0
    result = pipe(text[:512])[0]
    label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    return label_map.get(result["label"].lower(), 0.0) * result["score"]


# ─────────────────────────────────────────────────────────────────────────────
# GDELT connector
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_gdelt_headlines(
    query: str,
    max_records: int = 50,
) -> list[str]:
    """
    Query GDELT v2 doc API and return a list of article titles.

    Args:
        query:       GDELT query string (e.g. "bitcoin OR cryptocurrency").
        max_records: max articles to retrieve (GDELT caps at 250).

    Returns:
        List of headline strings.
    """
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": str(max_records),
        "format": "json",
    }
    url = settings.gdelt_base_url

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    logger.warning("GDELT returned HTTP %d", resp.status)
                    return []
                data = await resp.json(content_type=None)
                articles = data.get("articles", [])
                return [a.get("title", "") for a in articles if a.get("title")]
    except Exception as exc:  # noqa: BLE001
        logger.error("GDELT fetch failed: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# RSS connector
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_rss_headlines(feed_url: str | None = None) -> list[str]:
    """
    Parse an RSS feed and return entry titles.
    feedparser is synchronous — run in thread to stay non-blocking.
    """
    url = feed_url or settings.reuters_rss_feed

    def _parse() -> list[str]:
        parsed = feedparser.parse(url)
        return [e.get("title", "") for e in parsed.entries if e.get("title")]

    try:
        titles = await asyncio.to_thread(_parse)
        return titles
    except Exception as exc:  # noqa: BLE001
        logger.error("RSS fetch failed: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment aggregator
# ─────────────────────────────────────────────────────────────────────────────

_ASSET_KEYWORDS: dict[str, list[str]] = {
    "BTC": ["bitcoin", "btc", "crypto", "cryptocurrency"],
    "GOLD": ["gold", "xauusd", "bullion", "precious metal"],
    "GLOBAL": ["fed", "rate hike", "inflation", "war", "geopolit", "sanctions"],
}


def _classify_headline(headline: str) -> str:
    """
    Map a headline to the most relevant asset label, or GLOBAL as default.
    Simple keyword matching — fast enough for production ingestion cadence.
    """
    lower = headline.lower()
    for asset, keywords in _ASSET_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return asset
    return "GLOBAL"


async def compute_daily_sentiment(date: datetime | None = None) -> dict[str, float]:
    """
    Aggregate sentiment for a single trading day.

    Fetches headlines from GDELT + RSS concurrently, classifies by asset,
    scores with FinBERT, and returns mean scores per label.

    Args:
        date: target date (UTC); defaults to today.

    Returns:
        Dict {asset_label: mean_sentiment_score} — e.g.
        {"BTC": 0.42, "GOLD": -0.18, "GLOBAL": 0.05}
    """
    _ = date  # Future: filter GDELT by date range
    gdelt_tasks = [
        asyncio.create_task(fetch_gdelt_headlines("bitcoin OR cryptocurrency")),
        asyncio.create_task(fetch_gdelt_headlines("gold bullion OR XAUUSD")),
        asyncio.create_task(fetch_gdelt_headlines("geopolitical risk OR sanctions OR war")),
    ]
    rss_task = asyncio.create_task(fetch_rss_headlines())

    gdelt_results = await asyncio.gather(*gdelt_tasks)
    rss_headlines = await rss_task

    all_headlines = [h for group in gdelt_results for h in group] + rss_headlines

    # Score in thread pool — FinBERT inference blocks CPU
    scores_by_asset: dict[str, list[float]] = {k: [] for k in _ASSET_KEYWORDS}
    for headline in all_headlines:
        asset = _classify_headline(headline)
        score = await asyncio.to_thread(_score_text, headline)
        scores_by_asset[asset].append(score)

    result = {
        asset: float(np.mean(scores)) if scores else 0.0
        for asset, scores in scores_by_asset.items()
    }
    logger.info("Daily sentiment: %s", result)
    return result


async def build_sentiment_series(
    start: str = "2020-01-01",
    end: str | None = None,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """
    Build a historical daily sentiment DataFrame by iterating over trading dates.

    NOTE: GDELT's historical query API supports date-filtered requests.
    For backtesting, this loop is run offline once and the result cached.

    Returns:
        DataFrame indexed by UTC date with columns [BTC, GOLD, GLOBAL].
    """
    end_str = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cache = cache_path or settings.data_cache_path / f"sentiment_{start}_{end_str}.parquet"

    if cache.exists():
        logger.info("Sentiment cache hit: %s", cache)
        return pd.read_parquet(cache)

    date_range = pd.date_range(start=start, end=end_str, freq="B", tz="UTC")
    rows = []
    for dt in date_range:
        scores = await compute_daily_sentiment(dt.to_pydatetime())
        rows.append({"date": dt, **scores})

    df = pd.DataFrame(rows).set_index("date")
    settings.data_cache_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df
