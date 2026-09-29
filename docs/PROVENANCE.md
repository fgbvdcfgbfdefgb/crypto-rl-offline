# Offline asset provenance

Assets were packaged on 2026-09-29 UTC. `data/manifest.json` is the machine-readable source of exact sizes, coverage, and SHA-256 values.

## Minute OHLCV

- Provider: Binance public spot market-data archive
- Base URL: `https://data.binance.vision/data/spot/`
- Pairs: `BTCUSDT`, `ETHUSDT`, `LTCUSDT`
- Interval: one minute
- Source form: monthly ZIP archives plus available daily ZIPs after the last complete month
- Packaged form: yearly Parquet, Zstandard compression, 1,440-row groups
- Preserved fields: UTC timestamp, open, high, low, close, base volume, quote volume, trade count, taker-buy base volume, taker-buy quote volume

Binance changed archive timestamps from milliseconds to microseconds in 2025. The builder detects the unit by magnitude and normalizes every timestamp to UTC. Consumers remain responsible for Binance's terms and any applicable local rules.

## Historical crypto news

- Upstream repository: `https://github.com/soheilrahsaz/cryptoNewsDataset`
- Source artifact: `csvOutput/news_currencies_source_joinedResult.rar`
- Upstream license: CC0-1.0
- Packaged subset: rows explicitly tagged BTC, ETH, or LTC
- Fields: upstream ID, asset, UTC publication time, title, description, source domain, URL, and upstream vote counters

The data originates from CryptoPanic-linked sources. Headlines may be duplicated, mislabeled, corrected later, or unavailable for some dates. The trainer uses title text only and imposes timestamp cutoffs to avoid exposing later same-day headlines.

## Financial sentiment model

- Upstream: `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`
- Pinned revision: `ae0eab9ad336d7d548e0efe394b07c04bcaf6e91`
- Upstream task: three-class financial-news sentiment
- Architecture: DistilRoBERTa sequence classifier, approximately 82M parameters
- License declared by upstream: Apache-2.0
- Original weight file: `model.safetensors`
- Size: 328,499,560 bytes
- SHA-256: `c0b61385e4482edd179b69042c014dcb53a79431784f34a0171f5d43b092feaa`

The weight file is split into four sub-100 MiB normal Git files. `crypto_rl.news.assemble_offline_model` verifies every part and the final file before Transformers loads it with `local_files_only=True`.

## Daily context

Every value exposed to an episode on date D is lagged or otherwise available no later than the start of D.

### Alternative.me

- Dataset: Crypto Fear & Greed Index
- Endpoint used by the maintainer builder: `https://api.alternative.me/fng/?limit=0&format=json`
- Feature: `(value - 50) / 50`, clipped to [-1, 1], lagged one day

### Federal Reserve Economic Data (FRED)

Downloaded as public CSV series:

- `VIXCLS`: CBOE Volatility Index
- `DTWEXBGS`: Nominal Broad U.S. Dollar Index
- `SP500`: S&P 500
- `DGS10`: 10-Year Treasury Constant Maturity Rate
- `DFF`: Effective Federal Funds Rate

Values are forward-filled across non-reporting days and lagged one day. The S&P 500 signal is a clipped return; the others use bounded level transforms or causal rolling standardization.

### Coin Metrics Community API

- Endpoint: `https://community-api.coinmetrics.io/v4/timeseries/asset-metrics`
- Assets: BTC, ETH, LTC
- Frequency: daily
- Metrics: `CapMrktCurUSD`, `TxCnt`, `AdrActCnt`, `HashRate`

Transaction count, active addresses, and hash rate are log-transformed and causally standardized. Market-cap return is clipped/scaled. All are lagged one day. Missing metrics (for example, an unsupported early history) become zero only after alignment.

### Prior-day exchange aggregates

Prior-day asset return and realized volatility are derived from the packaged Binance minute bars and lagged one day.

## Rebuilding

`tools/build_offline_assets.py` is a maintainer/provenance utility that requires internet access and a `7zz` binary with RAR5 support. It is never invoked by `train.py`. Rebuilding on a later date can change upstream coverage and therefore checksums; review licensing and data-quality changes before publishing a rebuilt repository.
