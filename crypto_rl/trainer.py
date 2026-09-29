"""Independent multi-device PPO agent workers and resumable checkpoints."""
from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import Config
from .data import OfflineMarketData
from .env import CryptoTradingEnv
from .model import ActorCritic
from .news import request_profile
from .plotting import save_episode_plot


def _under(root: Path, path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _append_metrics(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _save_timeline(env: CryptoTradingEnv, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trade_by_step = {int(t["step"]): t for t in env.trades}
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["minute", "timestamp", "price", "equity", "cash", "units", "action", "side", "notional", "fee"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        assert env.day is not None
        for minute in range(len(env.equity_curve)):
            trade = trade_by_step.get(minute, {})
            writer.writerow(
                {
                    "minute": minute,
                    "timestamp": str(env.day.timestamps[minute]),
                    "price": env.price_curve[minute],
                    "equity": env.equity_curve[minute],
                    "cash": env.cash_curve[minute],
                    "units": env.units_curve[minute],
                    "action": "" if minute == 0 or minute - 1 >= len(env.action_curve) else env.action_curve[minute - 1],
                    "side": trade.get("side", "hold"),
                    "notional": trade.get("notional", 0.0),
                    "fee": trade.get("fee", 0.0),
                }
            )


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    observations: np.ndarray,
    actions: np.ndarray,
    old_log_probs: np.ndarray,
    values: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    cfg: Config,
    device: torch.device,
) -> dict[str, float]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + cfg.ppo.gamma * next_value * nonterminal - values[t]
        last_advantage = delta + cfg.ppo.gamma * cfg.ppo.gae_lambda * nonterminal * last_advantage
        advantages[t] = last_advantage
        next_value = float(values[t])
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs_t = torch.as_tensor(observations, dtype=torch.float32, device=device)
    action_t = torch.as_tensor(actions, dtype=torch.long, device=device)
    old_log_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=device)
    return_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
    advantage_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    count = len(rewards)
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0}

    model.train()
    for _ in range(cfg.ppo.update_epochs):
        permutation = torch.randperm(count, device=device)
        for start in range(0, count, cfg.ppo.minibatch_size):
            index = permutation[start : start + cfg.ppo.minibatch_size]
            log_prob, entropy, predicted_value = model.evaluate_actions(obs_t[index], action_t[index])
            ratio = torch.exp(log_prob - old_log_t[index])
            unclipped = ratio * advantage_t[index]
            clipped = torch.clamp(ratio, 1.0 - cfg.ppo.clip_ratio, 1.0 + cfg.ppo.clip_ratio) * advantage_t[index]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * (predicted_value - return_t[index]).pow(2).mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + cfg.ppo.value_coef * value_loss - cfg.ppo.entropy_coef * entropy_mean
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.ppo.max_grad_norm)
            optimizer.step()
            totals["policy_loss"] += float(policy_loss.detach().cpu())
            totals["value_loss"] += float(value_loss.detach().cpu())
            totals["entropy"] += float(entropy_mean.detach().cpu())
            totals["updates"] += 1
    updates = max(1, int(totals.pop("updates")))
    return {key: value / updates for key, value in totals.items()}


def agent_worker(
    agent_id: int,
    gpu_id: int | None,
    cfg: Config,
    project_root: str,
    request_queue: Any,
    response_queue: Any,
    resume: bool,
) -> None:
    root = Path(project_root)
    seed = cfg.train.seed + 10_003 * agent_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(1)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) // max(1, cfg.train.cpu_agents_if_no_gpu))))

    data = OfflineMarketData(
        _under(root, cfg.data.prices_dir),
        _under(root, cfg.data.catalog_path),
        _under(root, cfg.data.context_path),
        lookback=cfg.data.lookback,
        require_complete_days=cfg.data.require_complete_days,
        split=cfg.train.split,
        train_fraction=cfg.train.train_fraction,
    )
    env = CryptoTradingEnv(
        lookback=cfg.data.lookback,
        initial_balance=cfg.environment.initial_balance,
        fee_rate=cfg.environment.fee_rate,
        slippage_bps=cfg.environment.slippage_bps,
        target_allocations=cfg.environment.target_allocations,
        turnover_penalty=cfg.environment.turnover_penalty,
        drawdown_penalty=cfg.environment.drawdown_penalty,
        reward_scale=cfg.environment.reward_scale,
    )
    model = ActorCritic(cfg.data.lookback, env.action_dim, cfg.ppo.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ppo.learning_rate, eps=1e-5)
    agent_dir = _under(root, cfg.train.output_dir) / f"agent_{agent_id:02d}"
    checkpoint_path = agent_dir / "checkpoint.pt"
    start_episode = 1

    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_episode = int(checkpoint["episode"]) + 1
        if "numpy_rng_state" in checkpoint:
            rng.bit_generator.state = checkpoint["numpy_rng_state"]
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        print(f"[agent {agent_id}] resumed at episode {start_episode} on {device}", flush=True)
    else:
        print(f"[agent {agent_id}] starting fresh on {device}", flush=True)

    for episode in range(start_episode, cfg.train.episodes + 1):
        started = time.time()
        key = data.sample_key(rng)
        day = data.load_day(key)
        profile = request_profile(
            request_queue,
            response_queue,
            agent_id,
            episode,
            key.symbol,
            str(key.date.date()),
        )
        observation = env.reset(day, profile)
        observations: list[np.ndarray] = []
        actions: list[int] = []
        log_probs: list[float] = []
        values: list[float] = []
        rewards: list[float] = []
        dones: list[bool] = []
        terminated = False
        model.eval()
        while not terminated:
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
            action_t, log_prob_t, value_t = model.act(obs_tensor)
            action = int(action_t.item())
            result = env.step(action)
            observations.append(observation)
            actions.append(action)
            log_probs.append(float(log_prob_t.item()))
            values.append(float(value_t.item()))
            rewards.append(float(result.reward))
            dones.append(bool(result.terminated))
            observation = result.observation
            terminated = result.terminated

        losses = ppo_update(
            model,
            optimizer,
            np.asarray(observations, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(log_probs, dtype=np.float32),
            np.asarray(values, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
            cfg,
            device,
        )
        summary = env.summary()
        summary.update(losses)
        summary.update(
            {
                "agent_id": agent_id,
                "episode": episode,
                "device": str(device),
                "seed": seed,
                "wall_seconds": float(time.time() - started),
                "reward_sum": float(np.sum(rewards)),
            }
        )
        episode_dir = agent_dir / "episodes"
        stem = f"episode_{episode:06d}_{key.symbol}_{key.date.date()}"
        if episode % cfg.train.plot_every == 0:
            save_episode_plot(env, summary, profile, episode_dir / f"{stem}.png", agent_id, episode)
        _save_timeline(env, episode_dir / f"{stem}.csv")
        report = dict(summary)
        report["trades"] = env.trades
        report["news_profile_by_utc_hour"] = profile.tolist()
        report["target_allocations"] = cfg.environment.target_allocations
        _atomic_json(report, episode_dir / f"{stem}.json")
        _append_metrics(agent_dir / "metrics.csv", summary)

        if episode % cfg.train.checkpoint_every == 0 or episode == cfg.train.episodes:
            checkpoint = {
                "format_version": 1,
                "episode": episode,
                "agent_id": agent_id,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "numpy_rng_state": rng.bit_generator.state,
                "python_rng_state": random.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "config": {
                    "lookback": cfg.data.lookback,
                    "action_dim": env.action_dim,
                    "hidden_size": cfg.ppo.hidden_size,
                },
            }
            _atomic_torch_save(checkpoint, checkpoint_path)
        print(
            f"[agent {agent_id}] episode {episode}/{cfg.train.episodes} "
            f"{key.symbol} {key.date.date()} profit=${summary['profit']:.2f} "
            f"return={summary['return_pct']:+.2f}% trades={summary['trade_count']} "
            f"time={summary['wall_seconds']:.1f}s",
            flush=True,
        )
