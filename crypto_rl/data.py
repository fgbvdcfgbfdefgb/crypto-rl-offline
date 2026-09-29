"""Offline price/context/news data access with causal feature engineering."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SYMBOLS = ("BTCUSDT", "ETHUSDT", "LTCUSDT")
MARKET_FEATURES = (
    "ret_1m",
    "ret_5m",
    "ret_15m",
    "ret_60m",
    "candle_body",
    "candle_range",
    "volume_z",
    "quote_volume_z",
    "trades_z",
    "realized_vol_30",
    "realized_vol_120",
    "rsi_14",
    "minute_sin",
    "minute_cos",
)
CONTEXT_FEATURES = (
    "fear_greed_scaled",
    "vix_scaled",
    "dollar_index_scaled",
    "sp500_return_clipped",
    "ten_year_yield_scaled",
    "fed_funds_scaled",
    "active_addresses_z",
    "transaction_count_z",
    "hash_rate_z",
    "market_cap_return_clipped",
    "asset_daily_return_clipped",
    "asset_realized_vol_scaled",
)
NEWS_FEATURES = ("sentiment", "positive_prob", "negative_prob", "news_count_scaled")


def _rolling_z(series: pd.Series, window: int, minimum: int = 20) -> pd.Series:
    shifted = series.shift(1)
    mean = shifted.rolling(window, min_periods=minimum).mean()
    std = shifted.rolling(window, min_periods=minimum).std().replace(0.0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8, 8)


def engineer_market_features(frame: pd.DataFrame) -> np.ndarray:
    """Build only backward-looking features; the row's candle is known at its close."""
    close = frame["close"].astype(float).clip(lower=1e-12)
    log_close = np.log(close)
    ret_1m = log_close.diff()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = ((100.0 - 100.0 / (1.0 + rs)).fillna(50.0) - 50.0) / 50.0
    minute = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute

    features = pd.DataFrame(
        {
            "ret_1m": ret_1m,
            "ret_5m": log_close.diff(5) / np.sqrt(5.0),
            "ret_15m": log_close.diff(15) / np.sqrt(15.0),
            "ret_60m": log_close.diff(60) / np.sqrt(60.0),
            "candle_body": (frame["close"] - frame["open"]) / close,
            "candle_range": (frame["high"] - frame["low"]) / close,
            "volume_z": _rolling_z(np.log1p(frame["volume"].astype(float)), 240),
            "quote_volume_z": _rolling_z(np.log1p(frame["quote_volume"].astype(float)), 240),
            "trades_z": _rolling_z(np.log1p(frame["trade_count"].astype(float)), 240),
            "realized_vol_30": ret_1m.rolling(30, min_periods=5).std() * np.sqrt(30.0),
            "realized_vol_120": ret_1m.rolling(120, min_periods=20).std() * np.sqrt(120.0),
            "rsi_14": rsi,
            "minute_sin": np.sin(2 * np.pi * minute / 1440.0),
            "minute_cos": np.cos(2 * np.pi * minute / 1440.0),
        }
    )
    # Return-like fields are scaled to useful neural-network magnitudes and clipped.
    for col in ("ret_1m", "ret_5m", "ret_15m", "ret_60m", "candle_body", "candle_range"):
        features[col] = features[col] * 100.0
    for col in ("realized_vol_30", "realized_vol_120"):
        features[col] = features[col] * 100.0
    return (
        features.loc[:, MARKET_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(-10.0, 10.0)
        .to_numpy(dtype=np.float32)
    )


@dataclass(frozen=True)
class EpisodeKey:
    symbol: str
    date: pd.Timestamp


@dataclass
class DayData:
    key: EpisodeKey
    timestamps: np.ndarray
    prices: np.ndarray
    market_features: np.ndarray
    daily_context: np.ndarray


class OfflineMarketData:
    """Reads one day at a time from yearly Parquet files.

    The catalog is created when assets are packaged. Only complete 1,440-minute UTC
    days are eligible by default. Daily context is pre-lagged by its builder, so a
    row dated D contains only information available at the start of D.
    """

    def __init__(
        self,
        prices_dir: str | Path,
        catalog_path: str | Path,
        context_path: str | Path,
        lookback: int = 60,
        require_complete_days: bool = True,
        split: str = "train",
        train_fraction: float = 0.8,
    ) -> None:
        self.prices_dir = Path(prices_dir)
        self.lookback = int(lookback)
        catalog = pd.read_csv(catalog_path, parse_dates=["date"])
        catalog = catalog[catalog["symbol"].isin(SYMBOLS)].copy()
        if require_complete_days:
            catalog = catalog[catalog["rows"] == 1440]
        catalog["date"] = pd.to_datetime(catalog["date"], utc=True).dt.normalize()
        catalog = catalog.sort_values(["symbol", "date"]).reset_index(drop=True)
        self.catalog = self._split_catalog(catalog, split, train_fraction)
        if self.catalog.empty:
            raise ValueError(f"No eligible days in catalog for split={split!r}")

        context_file = Path(context_path)
        if context_file.exists():
            context = pd.read_parquet(context_file)
            context["date"] = pd.to_datetime(context["date"], utc=True).dt.normalize()
            context = context.set_index(["symbol", "date"])
            self.context = context
        else:
            self.context = pd.DataFrame()

    @staticmethod
    def _split_catalog(catalog: pd.DataFrame, split: str, fraction: float) -> pd.DataFrame:
        if split == "all":
            return catalog
        parts: list[pd.DataFrame] = []
        for _, group in catalog.groupby("symbol", sort=False):
            cut = max(1, min(len(group) - 1, int(len(group) * fraction)))
            parts.append(group.iloc[:cut] if split == "train" else group.iloc[cut:])
        return pd.concat(parts, ignore_index=True)

    @property
    def observation_shape(self) -> tuple[int, int]:
        return self.lookback, len(MARKET_FEATURES)

    def sample_key(self, rng: np.random.Generator) -> EpisodeKey:
        # Uniform over assets first, then days, to avoid BTC's longer history dominating.
        symbols = self.catalog["symbol"].unique()
        symbol = str(rng.choice(symbols))
        choices = self.catalog[self.catalog["symbol"] == symbol]
        row = choices.iloc[int(rng.integers(0, len(choices)))]
        return EpisodeKey(symbol=symbol, date=pd.Timestamp(row["date"]))

    def keys(self) -> Iterable[EpisodeKey]:
        for row in self.catalog.itertuples(index=False):
            yield EpisodeKey(str(row.symbol), pd.Timestamp(row.date))

    def _read_range(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        years = range(start.year, end.year + 1)
        pieces: list[pd.DataFrame] = []
        for year in years:
            path = self.prices_dir / f"{symbol}_{year}.parquet"
            if not path.exists():
                continue
            part = pd.read_parquet(
                path,
                filters=[("timestamp", ">=", start.to_pydatetime()), ("timestamp", "<", end.to_pydatetime())],
            )
            pieces.append(part)
        if not pieces:
            raise FileNotFoundError(f"No price data for {symbol} in {start}..{end}")
        frame = pd.concat(pieces, ignore_index=True)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")

    def _daily_context(self, key: EpisodeKey) -> np.ndarray:
        if self.context.empty or (key.symbol, key.date) not in self.context.index:
            return np.zeros(len(CONTEXT_FEATURES), dtype=np.float32)
        row = self.context.loc[(key.symbol, key.date)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        return np.asarray([float(row.get(c, 0.0) or 0.0) for c in CONTEXT_FEATURES], dtype=np.float32)

    def load_day(self, key: EpisodeKey) -> DayData:
        day_start = key.date
        day_end = day_start + pd.Timedelta(days=1)
        raw_start = day_start - pd.Timedelta(minutes=max(self.lookback + 240, 300))
        frame = self._read_range(key.symbol, raw_start, day_end)

        # Enforce a regular minute grid. Missing pre-roll candles are causally forward-filled;
        # missing volumes/trades become zero. Eligible episode days themselves are complete.
        grid = pd.date_range(raw_start, day_end - pd.Timedelta(minutes=1), freq="min", tz="UTC")
        frame = frame.set_index("timestamp").reindex(grid)
        for col in ("open", "high", "low", "close"):
            frame[col] = frame[col].ffill()
        for col in ("volume", "quote_volume", "trade_count", "taker_buy_base", "taker_buy_quote"):
            if col not in frame:
                frame[col] = 0.0
            frame[col] = frame[col].fillna(0.0)
        frame = frame.rename_axis("timestamp").reset_index()
        all_features = engineer_market_features(frame)
        mask = (frame["timestamp"] >= day_start) & (frame["timestamp"] < day_end)
        episode_idx = np.flatnonzero(mask.to_numpy())
        if len(episode_idx) != 1440:
            raise ValueError(f"Expected 1440 bars for {key}, got {len(episode_idx)}")
        first = int(episode_idx[0])
        # Keep lookback-1 rows before minute zero so every observation has a full window.
        start = first - (self.lookback - 1)
        end = int(episode_idx[-1]) + 1
        if start < 0:
            raise ValueError(f"Insufficient pre-roll for {key}")
        return DayData(
            key=key,
            timestamps=frame.loc[first:end - 1, "timestamp"].to_numpy(),
            prices=frame.loc[first:end - 1, "close"].to_numpy(dtype=np.float64),
            market_features=all_features[start:end],
            daily_context=self._daily_context(key),
        )
