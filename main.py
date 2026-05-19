"""
main.py — Pipeline orchestrator.

Execution sequence:
  1. Load config and ensure storage directories exist.
  2. Fetch OHLCV data for BTC and XAUUSD (async, parallel).
  3. Feature engineering on both assets.
  4. Select optimal HMM n_regimes via BIC; train HMM.
  5. Compute geopolitical sentiment series (cached).
  6. Run core strategy signal generation on the most recent bar.
  7. Execute walk-forward backtest; print metrics report.
  8. Persist model artefacts.

Usage:
    python main.py [--mode live|backtest] [--asset BTC|XAUUSD|all]

Environment:
    Copy .env.example → .env and populate credentials before running.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("main")


async def run_pipeline(asset: str = "all", mode: str = "backtest") -> None:
    """Main async pipeline."""
    settings.ensure_dirs()

    # ── 1. Data ingestion ──────────────────────────────────────────────────
    from data.feed import fetch_all, Timeframe
    logger.info("Fetching OHLCV data...")
    data = await fetch_all(
        timeframe="1d",
        since=settings.backtest_start,
        until=settings.backtest_end,
    )

    # ── 2. Feature engineering ─────────────────────────────────────────────
    from data.features import FeatureEngineer
    fe = FeatureEngineer()
    enriched: dict[str, object] = {}
    observations: dict[str, object] = {}

    assets_to_process = ["BTC", "XAUUSD"] if asset == "all" else [asset]
    for sym in assets_to_process:
        if sym not in data:
            logger.warning("No data for %s, skipping.", sym)
            continue
        enriched[sym] = fe.transform(data[sym])  # type: ignore[arg-type]
        observations[sym] = fe.get_hmm_observations(enriched[sym])  # type: ignore[arg-type]
        logger.info("%s: %d enriched bars.", sym, len(enriched[sym]))  # type: ignore[arg-type]

    # ── 3. HMM training ────────────────────────────────────────────────────
    from models.hmm_regimes import HMMRegimeDetector, select_optimal_n_regimes
    import numpy as np

    hmm_models: dict[str, HMMRegimeDetector] = {}
    for sym in assets_to_process:
        if sym not in observations:
            continue
        obs_arr = observations[sym]  # type: ignore[index]
        logger.info("Selecting optimal HMM k for %s...", sym)
        best_k, bic_scores = select_optimal_n_regimes(obs_arr, k_range=(2, 6))  # type: ignore[arg-type]
        detector = HMMRegimeDetector(n_regimes=best_k)
        detector.fit(obs_arr)  # type: ignore[arg-type]
        save_path = detector.save()
        logger.info("%s: HMM saved to %s", sym, save_path)
        hmm_models[sym] = detector

    # ── 4. Geopolitical sentiment ──────────────────────────────────────────
    from data.geo_events import build_sentiment_series
    logger.info("Building sentiment series (this may take a while on first run)...")
    try:
        sentiment_df = await build_sentiment_series(
            start=settings.backtest_start, end=settings.backtest_end
        )
        logger.info("Sentiment series: %d days.", len(sentiment_df))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Sentiment unavailable (%s); using neutral.", exc)
        import pandas as pd
        sentiment_df = pd.DataFrame()

    # ── 5. Strategy evaluation (latest bar) ───────────────────────────────
    from models.wyckoff import WyckoffAnalyser
    from strategies.core_strategy import CoreStrategy
    from strategies.inefficiencies import TemporalArbitrageFilter

    for sym in assets_to_process:
        if sym not in hmm_models:
            continue
        logger.info("Evaluating strategy signal for %s...", sym)
        strategy = CoreStrategy(
            hmm_detector=hmm_models[sym],
            wyckoff=WyckoffAnalyser(),
            arb_filter=TemporalArbitrageFilter(min_win_rate=0.60),
        )
        sentiment_score = (
            float(sentiment_df[sym if sym != "XAUUSD" else "GOLD"].iloc[-1])
            if not sentiment_df.empty and (sym if sym != "XAUUSD" else "GOLD") in sentiment_df.columns
            else 0.0
        )
        signal = strategy.evaluate(
            asset=sym,
            hmm_observations=observations[sym],  # type: ignore[arg-type]
            ohlcv_df=enriched[sym],  # type: ignore[arg-type]
            sentiment_score=sentiment_score,
        )
        logger.info(
            "%s signal: direction=%+d, confidence=%.2f, regime=%s",
            sym, signal.direction, signal.confidence, signal.hmm_regime
        )
        for r in signal.rationale:
            logger.info("  → %s", r)

    # ── 6. Backtest + WFO ─────────────────────────────────────────────────
    if mode == "backtest":
        from backtest.engine import BacktestEngine
        import pandas as pd

        for sym in assets_to_process:
            if sym not in enriched:
                continue
            edf = enriched[sym]  # type: ignore[index]
            close = edf["close"]  # type: ignore[index]

            # Placeholder signal series: +1 on bull regime bars, -1 on bear
            obs_arr = observations[sym]  # type: ignore[index]
            regimes = hmm_models[sym].predict_regimes(obs_arr)
            regime_labels = np.array([
                hmm_models[sym]._regime_labels.get(int(r), "") for r in regimes
            ])
            signals = pd.Series(
                np.where(np.char.startswith(regime_labels, "bull"), 1,
                         np.where(np.char.startswith(regime_labels, "bear"), -1, 0)),
                index=close.index,
            )

            sent_series = (
                sentiment_df.get("BTC" if sym == "BTC" else "GOLD", pd.Series())
                if not sentiment_df.empty else pd.Series()
            )

            engine = BacktestEngine(close=close, signals=signals, sentiment=sent_series)
            logger.info("Running WFO for %s (5 splits)...", sym)
            wfo = engine.walk_forward(n_splits=5)
            for i, res in enumerate(wfo):
                logger.info("WFO OOS split %d: %s", i + 1, res.summary())


def main() -> None:
    parser = argparse.ArgumentParser(description="HMM Trading Bot")
    parser.add_argument("--mode", choices=["live", "backtest"], default="backtest")
    parser.add_argument("--asset", choices=["BTC", "XAUUSD", "all"], default="all")
    args = parser.parse_args()

    if args.mode == "live":
        logger.warning("Live mode is not yet implemented. Running in backtest mode.")
        args.mode = "backtest"

    asyncio.run(run_pipeline(asset=args.asset, mode=args.mode))


if __name__ == "__main__":
    main()
