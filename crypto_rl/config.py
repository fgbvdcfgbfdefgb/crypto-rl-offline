"""Configuration loading and validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    prices_dir: str = "data/prices"
    catalog_path: str = "data/price_catalog.csv"
    context_path: str = "data/context/daily_market_context.parquet"
    news_path: str = "data/news/crypto_news.parquet"
    model_dir: str = "models/financial_sentiment"
    lookback: int = 60
    require_complete_days: bool = True


@dataclass
class EnvironmentConfig:
    initial_balance: float = 2000.0
    fee_rate: float = 0.001
    slippage_bps: float = 2.0
    target_allocations: list[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    turnover_penalty: float = 0.002
    drawdown_penalty: float = 0.05
    reward_scale: float = 100.0


@dataclass
class PPOConfig:
    hidden_size: int = 128
    learning_rate: float = 3.0e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 6
    minibatch_size: int = 256


@dataclass
class TrainConfig:
    episodes: int = 1000
    seed: int = 1337
    split: str = "train"
    train_fraction: float = 0.8
    max_news_per_day: int = 64
    output_dir: str = "runs"
    checkpoint_every: int = 1
    plot_every: int = 1
    cpu_agents_if_no_gpu: int = 1
    sentiment_batch_size: int = 32
    sentiment_threads: int = 4
    strict_offline_model: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _merge_dataclass(obj: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(obj, key):
            raise ValueError(f"Unknown configuration key: {type(obj).__name__}.{key}")
        setattr(obj, key, value)
    return obj


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = Config()
    for section in ("data", "environment", "ppo", "train"):
        if section in raw:
            _merge_dataclass(getattr(cfg, section), raw[section] or {})
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    if cfg.data.lookback < 2:
        raise ValueError("data.lookback must be at least 2")
    if cfg.environment.initial_balance <= 0:
        raise ValueError("environment.initial_balance must be positive")
    allocations = cfg.environment.target_allocations
    if not allocations or any(not 0.0 <= x <= 1.0 for x in allocations):
        raise ValueError("target_allocations must contain values in [0, 1]")
    if sorted(allocations) != allocations:
        raise ValueError("target_allocations must be sorted")
    if not 0 < cfg.train.train_fraction < 1:
        raise ValueError("train.train_fraction must be in (0, 1)")
    if cfg.train.split not in {"train", "validation", "all"}:
        raise ValueError("train.split must be train, validation, or all")
