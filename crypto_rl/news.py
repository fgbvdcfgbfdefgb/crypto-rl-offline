"""Offline CPU financial-news sentiment service.

A single CPU worker owns the packaged transformer and serves all GPU agents. For
minute t, agents receive only news published before the start of t's UTC hour;
future headlines from the same episode day never enter the observation.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import NEWS_FEATURES


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def assemble_offline_model(model_dir: str | Path) -> Path:
    """Reassemble a <100 MiB Git-safe sharded weight file, without networking.

    During training, ``CRYPTO_RL_RUNTIME_MODEL_DIR`` points into the writable run
    directory. This lets the checked-out/synchronized repository itself be read-only.
    """
    source_dir = Path(model_dir)
    manifest_path = source_dir / "chunks.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing offline model manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime_value = os.environ.get("CRYPTO_RL_RUNTIME_MODEL_DIR")
    runtime_dir = Path(runtime_value) if runtime_value else source_dir
    runtime_dir.mkdir(parents=True, exist_ok=True)
    # Transformers expects tokenizer/config files beside the reconstructed weights.
    if runtime_dir != source_dir:
        for name in ("config.json", "merges.txt", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"):
            source = source_dir / name
            target_metadata = runtime_dir / name
            if not target_metadata.exists() or target_metadata.stat().st_size != source.stat().st_size:
                shutil.copy2(source, target_metadata)

    target = runtime_dir / manifest["target"]
    expected = manifest["sha256"]
    if target.exists() and target.stat().st_size == manifest["size"] and sha256_file(target) == expected:
        return runtime_dir

    missing = [item["name"] for item in manifest["parts"] if not (source_dir / item["name"]).exists()]
    if missing:
        raise FileNotFoundError(f"Missing model chunks: {', '.join(missing)}")
    fd, temporary_name = tempfile.mkstemp(prefix="model.", suffix=".assembling", dir=runtime_dir)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as out:
            for item in manifest["parts"]:
                part = source_dir / item["name"]
                if part.stat().st_size != item["size"] or sha256_file(part) != item["sha256"]:
                    raise ValueError(f"Corrupt offline model chunk: {part}")
                with part.open("rb") as handle:
                    while block := handle.read(8 * 1024 * 1024):
                        out.write(block)
        if temporary.stat().st_size != manifest["size"] or sha256_file(temporary) != expected:
            raise ValueError("Reassembled model checksum does not match manifest")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return runtime_dir


class TransformerSentiment:
    def __init__(self, model_dir: str | Path, batch_size: int, threads: int) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(max(1, int(threads)))
        path = assemble_offline_model(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
        self.model.eval().to("cpu")
        self.torch = torch
        self.batch_size = max(1, int(batch_size))
        labels = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.negative_idx = next((i for i, v in labels.items() if "neg" in v), 0)
        self.positive_idx = next((i for i, v in labels.items() if "pos" in v), len(labels) - 1)

    def score(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 3), dtype=np.float32)
        output: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=192,
            )
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits
                probs = self.torch.softmax(logits, dim=-1).cpu().numpy()
            # Standardize output columns to negative, neutral remainder, positive.
            neg = probs[:, self.negative_idx]
            pos = probs[:, self.positive_idx]
            neutral = np.maximum(0.0, 1.0 - neg - pos)
            output.append(np.stack((neg, neutral, pos), axis=1).astype(np.float32))
        return np.concatenate(output, axis=0)


class LexiconFallback:
    """Deterministic emergency fallback; not used when strict_offline_model=true."""

    POSITIVE = {"gain", "gains", "rise", "rises", "rally", "approval", "adoption", "bullish", "surge", "record", "growth"}
    NEGATIVE = {"loss", "losses", "fall", "falls", "crash", "ban", "hack", "fraud", "bearish", "lawsuit", "liquidation"}

    def score(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            tokens = set(str(text).lower().replace("-", " ").split())
            p = len(tokens & self.POSITIVE)
            n = len(tokens & self.NEGATIVE)
            denom = max(1, p + n)
            pos = 0.1 + 0.8 * p / denom if p + n else 0.1
            neg = 0.1 + 0.8 * n / denom if p + n else 0.1
            rows.append([neg, max(0.0, 1.0 - pos - neg), pos])
        return np.asarray(rows, dtype=np.float32)


def _load_cache(path: Path) -> dict[tuple[str, str], np.ndarray]:
    cache: dict[tuple[str, str], np.ndarray] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                profile = np.asarray(row["profile"], dtype=np.float32)
                if profile.shape == (24, len(NEWS_FEATURES)):
                    cache[(row["symbol"], row["date"])] = profile
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return cache


def _make_profile(
    news: pd.DataFrame,
    scorer: TransformerSentiment | LexiconFallback,
    symbol: str,
    date: str,
    max_news: int,
) -> np.ndarray:
    asset = symbol.replace("USDT", "")
    start = pd.Timestamp(date, tz="UTC")
    final_cutoff = start + pd.Timedelta(hours=23)
    subset = news[
        (news["asset"] == asset)
        & (news["timestamp"] >= start - pd.Timedelta(days=1))
        & (news["timestamp"] < final_cutoff)
    ].sort_values("timestamp")
    # Bound worst-case CPU work while retaining the most recent, actionable stories.
    subset = subset.tail(max(4 * max_news, max_news)).copy()
    if subset.empty:
        return np.zeros((24, len(NEWS_FEATURES)), dtype=np.float32)
    probabilities = scorer.score(subset["title"].fillna("").astype(str).tolist())
    subset["neg"] = probabilities[:, 0]
    subset["pos"] = probabilities[:, 2]
    profile = np.zeros((24, len(NEWS_FEATURES)), dtype=np.float32)
    for hour in range(24):
        cutoff = start + pd.Timedelta(hours=hour)
        available = subset[(subset["timestamp"] < cutoff) & (subset["timestamp"] >= cutoff - pd.Timedelta(days=1))].tail(max_news)
        if available.empty:
            continue
        recency_hours = (cutoff - available["timestamp"]).dt.total_seconds().to_numpy() / 3600.0
        weights = np.exp(-recency_hours / 12.0)
        weights = weights / max(weights.sum(), 1e-12)
        pos = float(np.sum(available["pos"].to_numpy() * weights))
        neg = float(np.sum(available["neg"].to_numpy() * weights))
        profile[hour] = np.asarray(
            [pos - neg, pos, neg, min(1.0, np.log1p(len(available)) / np.log1p(max_news))],
            dtype=np.float32,
        )
    return profile


def sentiment_worker(
    request_queue: Any,
    response_queues: list[Any],
    status_queue: Any,
    news_path: str,
    model_dir: str,
    cache_path: str,
    max_news: int,
    batch_size: int,
    threads: int,
    strict: bool,
) -> None:
    """Multiprocessing entry point; intentionally imports model libraries in child."""
    try:
        news = pd.read_parquet(news_path, columns=["asset", "timestamp", "title"])
        news["timestamp"] = pd.to_datetime(news["timestamp"], utc=True)
        try:
            scorer: TransformerSentiment | LexiconFallback = TransformerSentiment(model_dir, batch_size, threads)
            scorer_name = "offline-transformer"
        except Exception as exc:
            if strict:
                raise
            scorer = LexiconFallback()
            scorer_name = f"lexicon-fallback ({type(exc).__name__})"
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache = _load_cache(cache_file)
        status_queue.put(("ready", {"scorer": scorer_name, "cached_profiles": len(cache), "news_rows": len(news)}))
        while True:
            message = request_queue.get()
            if message is None:
                break
            agent_id, request_id, symbol, date = message
            key = (str(symbol), str(date))
            try:
                if key not in cache:
                    cache[key] = _make_profile(news, scorer, key[0], key[1], max_news)
                    with cache_file.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"symbol": key[0], "date": key[1], "profile": cache[key].tolist()}) + "\n")
                response_queues[agent_id].put((request_id, cache[key], None))
            except Exception as exc:
                response_queues[agent_id].put((request_id, None, f"{type(exc).__name__}: {exc}"))
    except Exception as exc:
        status_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def request_profile(
    request_queue: Any,
    response_queue: Any,
    agent_id: int,
    request_id: int,
    symbol: str,
    date: str,
    timeout: float = 600.0,
) -> np.ndarray:
    request_queue.put((agent_id, request_id, symbol, date))
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for CPU sentiment worker")
        try:
            returned_id, profile, error = response_queue.get(timeout=min(remaining, 5.0))
        except queue.Empty:
            continue
        if returned_id != request_id:
            raise RuntimeError(f"Sentiment response mismatch: expected {request_id}, got {returned_id}")
        if error:
            raise RuntimeError(error)
        return np.asarray(profile, dtype=np.float32)
