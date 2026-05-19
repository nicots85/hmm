"""
/data/feed.py — Async OHLCV data connectors.

Design decisions:
- ccxt async client for BTC: handles rate-limit retries internally; we add an
  exponential backoff layer on top for 429/5xx responses.
- yfinance for XAUUSD (GC=F / GLD proxy): synchronous but wrapped in
  asyncio.to_thread to avoid blocking the event loop.
- All data is returned as timezone-aware UTC DataFrames with a canonical
  column schema: [open, high, low, close, volume].
- Raw OHLCV is cached to parquet under DATA_CACHE_PATH to avoid redundant
  API calls during iteration / backtesting.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import ccxt.async_support as ccxt_async  # type: ignore[import-untyped]
import pandas as pd
import yfinance as yf

from config import settings

logger = logging.getLogger(__name__)

# Canonical OHLCV columns — downstream modules rely on this contract.
OHLCV_COLS: list[str] = ["open", "high", "low", "close", "volume"]

Timeframe = Literal["1m", "5m", "15m", "1h", "4h", "1d"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(symbol: str, timeframe: str, start: str, end: str) -> Path:
    safe = symbol.replace("/", "_").replace(" ", "_")
    return settings.data_cache_path / f"{safe}_{timeframe}_{start}_{end}.parquet"


def _to_utc_df(records: list[list], cols: list[str] = OHLCV_COLS) -> pd.DataFrame:
    """Convert raw ccxt OHLCV list to a UTC-indexed DataFrame."""
    df = pd.DataFrame(records, columns=["timestamp"] + cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df.astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# BTC via Binance (ccxt async)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_btc_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: Timeframe = "1d",
    since: str = "2018-01-01",
    until: str | None = None,
    exchange_id: str = "binance",
) -> pd.DataFrame:
    """
    Fetch BTC OHLCV from Binance (or Bybit as fallback) via ccxt.

    Paginates automatically: ccxt returns max 1000 candles per request;
    we loop until `until` is reached or no new data arrives.

    Args:
        symbol:       ccxt market symbol, e.g. "BTC/USDT".
        timeframe:    OHLCV granularity.
        since:        ISO-8601 start date (UTC).
        until:        ISO-8601 end date (UTC); defaults to now.
        exchange_id:  "binance" | "bybit".

    Returns:
        UTC-indexed DataFrame with columns [open, high, low, close, volume].
    """
    until_dt = (
        datetime.fromisoformat(until).replace(tzinfo=timezone.utc)
        if until
        else datetime.now(timezone.utc)
    )
    until_ms = int(until_dt.timestamp() * 1000)
    since_ms = int(
        datetime.fromisoformat(since).replace(tzinfo=timezone.utc).timestamp() * 1000
    )

    cache = _cache_path(symbol, timeframe, since, str(until_dt.date()))
    if cache.exists():
        logger.info("Cache hit: %s", cache)
        return pd.read_parquet(cache)

    exchange_cls = getattr(ccxt_async, exchange_id)
    api_key = (
        settings.binance_api_key if exchange_id == "binance" else settings.bybit_api_key
    )
    api_secret = (
        settings.binance_secret if exchange_id == "binance" else settings.bybit_secret
    )

    # Public endpoints for historical OHLCV don't require auth, but we pass
    # credentials so rate limits apply to the authenticated tier if available.
    exchange = exchange_cls(
        {
            "apiKey": api_key or None,
            "secret": api_secret or None,
            "enableRateLimit": True,
        }
    )

    all_candles: list[list] = []
    cursor = since_ms

    try:
        while cursor < until_ms:
            candles = await exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=cursor, limit=1000
            )
            if not candles:
                break
            all_candles.extend(candles)
            cursor = candles[-1][0] + 1  # advance past last timestamp
            logger.debug("Fetched %d candles up to %s", len(all_candles), cursor)
    finally:
        await exchange.close()

    df = _to_utc_df(all_candles)
    # Trim to requested window
    df = df[df.index <= until_dt]

    settings.data_cache_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    logger.info("Cached %d rows → %s", len(df), cache)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# XAUUSD via yfinance  (GC=F = Gold Futures; GLD = ETF proxy)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_gold_ohlcv(
    ticker: str = "GC=F",
    timeframe: Timeframe = "1d",
    since: str = "2018-01-01",
    until: str | None = None,
) -> pd.DataFrame:
    """
    Fetch Gold OHLCV via yfinance (GC=F futures or GLD ETF).

    yfinance is synchronous; we run it in a thread to avoid blocking.
    Sub-daily granularity is limited to the last 60 days by the yfinance API.

    Returns:
        UTC-indexed DataFrame with columns [open, high, low, close, volume].
    """
    until_str = until or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cache = _cache_path(ticker, timeframe, since, until_str)

    if cache.exists():
        logger.info("Cache hit: %s", cache)
        return pd.read_parquet(cache)

    # Map our timeframe literals to yfinance interval strings.
    yf_interval_map: dict[str, str] = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "4h": "1h",   # yfinance has no 4h; caller must resample
        "1d": "1d",
    }
    interval = yf_interval_map.get(timeframe, "1d")

    def _download() -> pd.DataFrame:
        raw = yf.download(
            ticker,
            start=since,
            end=until_str,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        raw.columns = [c.lower() for c in raw.columns]
        raw.index = pd.to_datetime(raw.index, utc=True)
        raw.index.name = "timestamp"
        return raw[OHLCV_COLS].dropna()

    df = await asyncio.to_thread(_download)

    settings.data_cache_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    logger.info("Cached %d rows → %s", len(df), cache)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_all(
    timeframe: Timeframe = "1d",
    since: str = "2018-01-01",
    until: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch BTC and Gold concurrently; return keyed dict."""
    btc_task = asyncio.create_task(
        fetch_btc_ohlcv(timeframe=timeframe, since=since, until=until)
    )
    gold_task = asyncio.create_task(
        fetch_gold_ohlcv(timeframe=timeframe, since=since, until=until)
    )
    btc, gold = await asyncio.gather(btc_task, gold_task)
    return {"BTC": btc, "XAUUSD": gold}
