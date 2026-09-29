# Offline Multi-GPU Crypto PPO Trainer

A fully offline, resumable reinforcement-learning research project for minute-level **BTC, ETH, and LTC spot trading**. Each episode randomly selects one asset and one complete UTC day, starts with a simulated **$2,000**, trades for 1,439 minute-to-minute transitions, and writes an audit PNG, JSON report, and minute timeline.

> **Research only — not financial advice.** This simulator cannot establish that a strategy will be profitable in live markets. It omits order-book depth, latency, partial fills, taxes, exchange outages, changing fee tiers, and market impact. Never connect an unvalidated policy to real funds.

## What is included

- 14,189,722 Binance spot one-minute candles, partitioned by symbol/year as Parquet.
- 55,610 BTC/ETH/LTC historical news headlines.
- Daily lagged market context from Fear & Greed, FRED macro series, Coin Metrics community on-chain data, and prior-day market aggregates.
- A packaged 82M-parameter financial-news sentiment transformer for CPU inference.
- A dependency-light PPO implementation in PyTorch—no Gym or Stable-Baselines dependency.
- Automatic independent-agent parallelism: one process per visible GPU.
- A separate shared CPU sentiment process serving every GPU agent.
- Atomic checkpointing after every completed episode; rerun the same command to resume.
- A PNG + exact JSON + minute CSV for every episode.

All training-time model and dataset access is local. `train.py` sets Hugging Face/Transformers offline mode and contains no download path.

## Why the packaged model is not Qwen 8B

The requested 8B model would require roughly 16 GB of weights at BF16 and cannot fit in ordinary GitHub files. The user authorized a more suitable alternative, so this repository packages [`mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis), an Apache-2.0, 82M-parameter model trained specifically for financial sentiment.

It is a better fit for this narrow job: CPU inference is much faster, RAM use is dramatically lower, and its 328,499,560-byte weight file can be split into four normal Git objects below GitHub's 100 MiB per-file limit. The first run reassembles the file locally and verifies its SHA-256; it does **not** download anything.

## Offline asset coverage

| Asset | Minute-price coverage (UTC) | Rows |
|---|---:|---:|
| BTCUSDT | 2017-08-17 → 2026-09-28 | 4,786,399 |
| ETHUSDT | 2017-08-17 → 2026-09-28 | 4,786,398 |
| LTCUSDT | 2017-12-13 → 2026-09-28 | 4,616,925 |

News coverage is 2017-09-23 through 2025-12-03: 31,162 BTC rows, 21,721 ETH rows, and 2,727 LTC rows. Missing-news hours receive zero-valued news features. See [`docs/PROVENANCE.md`](docs/PROVENANCE.md) and [`data/manifest.json`](data/manifest.json) for sources, exact revisions, sizes, and SHA-256 checksums.

## Run on the synced Snowflake workspace

The supplied package inventory already contains the required libraries. From the repository root:

```bash
# Fast presence/size check (no network)
python verify_assets.py

# Optional full SHA-256 check; slower
python verify_assets.py --full

# Train; resumes existing runs/agent_XX/checkpoint.pt automatically
python train.py --config configs/default.yaml
```

Useful overrides:

```bash
# Quick smoke run, one episode per agent
python train.py --episodes 1

# Force CPU mode
python train.py --cpu-only --agents 1 --episodes 10

# Start fresh instead of loading checkpoints
python train.py --no-resume
```

When CUDA is available, the default worker count is `torch.cuda.device_count()`. GPU 0 runs agent 0, GPU 1 runs agent 1, and so on. These agents are intentionally independent: each has its own seed, optimizer, replay trajectory, checkpoint, metrics file, and episode artifacts. With no CUDA device, one CPU PPO agent runs by default. Adjust `cpu_agents_if_no_gpu` only if sufficient RAM/CPU is available.

### First-run behavior

`models/financial_sentiment/model.safetensors.part-*` is reassembled into `runs/runtime_model/financial_sentiment/model.safetensors`. Every part and the final file are checksum-verified. Using the writable run directory also supports a read-only synced repository. The reconstructed file is git-ignored with the rest of `runs/`. The sentiment model then remains resident in one CPU process while PPO agents use GPUs.

## Episode semantics

1. Sample BTC, ETH, or LTC uniformly.
2. Sample one eligible complete day from that asset's chronological training split.
3. Start with $2,000 cash and zero coin.
4. At each minute close, choose a target allocation from `[0%, 25%, 50%, 75%, 100%]`.
5. Execute with configurable fee and slippage, then mark equity at the next minute close.
6. Update PPO after the day ends and atomically save the checkpoint.

The action is a target allocation rather than an unconstrained order quantity; cash and coin units cannot become negative. The default environment is long-only spot trading—no margin and no shorting.

## Causality and leakage controls

- Market features use the current and earlier completed candles only.
- Daily context dated `D` contains completed daily/on-chain/macro information lagged from `D-1`.
- News at hour `H` contains only headlines timestamped before `H:00`, weighted by recency over the preceding 24 hours.
- Train/validation splitting is chronological within each asset. It is never randomized across time.
- The included unit test changes future prices and verifies that earlier features do not change.

The CPU sentiment worker may precompute all 24 hourly profiles for efficiency, but each observation receives only its corresponding causal cutoff. No full-day sentiment is exposed at minute zero.

## Output layout

```text
runs/
├── sentiment_cache.jsonl
└── agent_00/
    ├── checkpoint.pt
    ├── metrics.csv
    └── episodes/
        ├── episode_000001_BTCUSDT_2024-01-01.png
        ├── episode_000001_BTCUSDT_2024-01-01.json
        └── episode_000001_BTCUSDT_2024-01-01.csv
```

Each PNG contains price and buy/sell markers, equity/cash, drawdown, target allocation, news sentiment by hour, start/end balance, profit, return, fees, turnover, trade count, and diagnostic Sharpe. JSON and CSV sidecars preserve exact values so the PNG is not the only audit record.

## Resume behavior

A checkpoint is saved with a temporary file followed by an atomic rename after every completed episode. It includes:

- policy/value parameters,
- optimizer state,
- completed episode number,
- Python, NumPy, and PyTorch random states,
- architecture metadata.

If interrupted mid-day, the next run resumes from the prior completed episode and replays only the interrupted day. Existing episode artifacts are not overwritten because numbering resumes from the checkpoint.

## Configuration

Edit [`configs/default.yaml`](configs/default.yaml) to change episode count, trading friction, PPO hyperparameters, lookback, train split, output paths, sentiment batch size, or plotting/checkpoint frequency.

For an out-of-sample run, copy the config and set:

```yaml
train:
  split: validation
  output_dir: validation_runs
```

Do not tune hyperparameters on validation results and then describe those same results as unseen performance.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Repository size

The repository is intentionally large because the target training workspace must not download anything. It uses normal Git objects—not Git LFS—so a one-time Git synchronization receives the actual files. Every committed file is below 100 MiB. The checkout is about 1.1 GB plus the reconstructed 329 MB model file.

## License and attribution

Project code is MIT-licensed. Third-party model/data artifacts retain their own terms; see [`docs/PROVENANCE.md`](docs/PROVENANCE.md). Binance market data may be subject to Binance's terms and local rules. News-derived signals can contain errors, duplicates, source bias, and retrospective corrections.
