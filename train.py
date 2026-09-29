#!/usr/bin/env python3
"""Run resumable PPO training using only files in this repository."""
from __future__ import annotations

import argparse
import os
import queue
import sys
from dataclasses import asdict
from pathlib import Path

# Force all Hugging Face components offline before child processes import them.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from crypto_rl.config import load_config
from crypto_rl.news import sentiment_worker
from crypto_rl.trainer import agent_worker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml", help="YAML configuration path")
    parser.add_argument("--episodes", type=int, help="Override total episodes per agent")
    parser.add_argument("--agents", type=int, help="Override worker count (default: one per GPU, otherwise config)")
    parser.add_argument("--cpu-only", action="store_true", help="Do not use CUDA even when it is available")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing agent checkpoints")
    parser.add_argument(
        "--allow-sentiment-fallback",
        action="store_true",
        help="Use a small lexicon only if the packaged transformer is corrupt (not recommended)",
    )
    return parser.parse_args()


def resolve(root: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else root / path)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    cfg = load_config(resolve(root, args.config))
    if args.episodes is not None:
        if args.episodes < 1:
            raise ValueError("--episodes must be positive")
        cfg.train.episodes = args.episodes
    if args.allow_sentiment_fallback:
        cfg.train.strict_offline_model = False

    gpu_count = 0 if args.cpu_only else torch.cuda.device_count()
    if args.agents is not None and args.agents < 1:
        raise ValueError("--agents must be positive")
    if gpu_count:
        agent_count = args.agents or gpu_count
        if agent_count > gpu_count:
            raise ValueError(f"Requested {agent_count} agents but only {gpu_count} CUDA devices are visible")
        gpu_ids: list[int | None] = list(range(agent_count))
    else:
        agent_count = args.agents or cfg.train.cpu_agents_if_no_gpu
        gpu_ids = [None] * agent_count

    context = torch.multiprocessing.get_context("spawn")
    request_queue = context.Queue(maxsize=max(8, agent_count * 2))
    response_queues = [context.Queue(maxsize=2) for _ in range(agent_count)]
    status_queue = context.Queue(maxsize=2)
    runtime_root = Path(resolve(root, cfg.train.output_dir))
    cache_path = str(runtime_root / "sentiment_cache.jsonl")
    os.environ["CRYPTO_RL_RUNTIME_MODEL_DIR"] = str(runtime_root / "runtime_model" / "financial_sentiment")
    sentiment = context.Process(
        name="offline-sentiment-cpu",
        target=sentiment_worker,
        args=(
            request_queue,
            response_queues,
            status_queue,
            resolve(root, cfg.data.news_path),
            resolve(root, cfg.data.model_dir),
            cache_path,
            cfg.train.max_news_per_day,
            cfg.train.sentiment_batch_size,
            cfg.train.sentiment_threads,
            cfg.train.strict_offline_model,
        ),
    )
    sentiment.start()
    print("Starting packaged financial-news model on CPU (network access is disabled)...", flush=True)
    try:
        status, detail = status_queue.get(timeout=600)
    except queue.Empty:
        sentiment.terminate()
        sentiment.join()
        raise TimeoutError("Offline sentiment model did not become ready within 10 minutes")
    if status != "ready":
        sentiment.join(timeout=5)
        raise RuntimeError(f"Offline sentiment worker failed: {detail}")
    print(f"Sentiment worker ready: {detail}", flush=True)
    print(
        f"Launching {agent_count} independent PPO agent(s): "
        + (", ".join(f"cuda:{x}" for x in gpu_ids) if gpu_count else "CPU"),
        flush=True,
    )

    agents = []
    for agent_id, gpu_id in enumerate(gpu_ids):
        process = context.Process(
            name=f"ppo-agent-{agent_id}",
            target=agent_worker,
            args=(
                agent_id,
                gpu_id,
                cfg,
                str(root),
                request_queue,
                response_queues[agent_id],
                not args.no_resume,
            ),
        )
        process.start()
        agents.append(process)

    exit_code = 0
    try:
        for process in agents:
            process.join()
            if process.exitcode != 0:
                exit_code = process.exitcode or 1
    except KeyboardInterrupt:
        print("Interrupt received. The last completed episode checkpoints remain resumable.", file=sys.stderr)
        exit_code = 130
        for process in agents:
            if process.is_alive():
                process.terminate()
        for process in agents:
            process.join(timeout=10)
    finally:
        try:
            request_queue.put(None, timeout=2)
        except Exception:
            pass
        sentiment.join(timeout=30)
        if sentiment.is_alive():
            sentiment.terminate()
            sentiment.join()
    if exit_code:
        print("One or more agent workers failed; inspect the traceback above.", file=sys.stderr)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
