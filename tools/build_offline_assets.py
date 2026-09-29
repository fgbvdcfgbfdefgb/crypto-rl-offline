#!/usr/bin/env python3
"""Maintainer-only networked builder for the repository's offline assets.

Do NOT run this on the training machine. The committed Parquet/model chunks are
already sufficient for fully offline training. This script documents provenance
and makes the package reproducible for maintainers with network access.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BINANCE_DOWNLOAD = "https://data.binance.vision/"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "LTCUSDT")
KLINE_COLUMNS = (
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trade_count", "taker_buy_base", "taker_buy_quote", "ignore",
)
SESSION = requests.Session()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def get(url: str, *, stream: bool = False, attempts: int = 5) -> requests.Response:
    for attempt in range(attempts):
        try:
            response = SESSION.get(url, timeout=(20, 180), stream=stream)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def list_s3(prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        response = SESSION.get(S3, params=params, timeout=90)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        namespace = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
        keys.extend(node.text for node in root.findall("s:Contents/s:Key", namespace) if node.text)
        truncated = root.findtext("s:IsTruncated", default="false", namespaces=namespace) == "true"
        if not truncated:
            break
        token = root.findtext("s:NextContinuationToken", namespaces=namespace)
    return keys


def read_kline_zip(key: str) -> pd.DataFrame:
    content = get(BINANCE_DOWNLOAD + key).content
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV in {key}, found {csv_names}")
        frame = pd.read_csv(archive.open(csv_names[0]), header=None, names=KLINE_COLUMNS, low_memory=False)
    frame["open_time"] = pd.to_numeric(frame["open_time"], errors="coerce")
    frame = frame.dropna(subset=["open_time"]).copy()
    raw_time = frame["open_time"].astype("int64")
    # Binance spot archives use microseconds beginning in 2025; older files use milliseconds.
    timestamp = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    micro = raw_time > 100_000_000_000_000
    timestamp.loc[micro] = pd.to_datetime(raw_time.loc[micro], unit="us", utc=True)
    timestamp.loc[~micro] = pd.to_datetime(raw_time.loc[~micro], unit="ms", utc=True)
    output = pd.DataFrame({"timestamp": timestamp})
    for col in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"):
        output[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    output["trade_count"] = pd.to_numeric(frame["trade_count"], errors="coerce").fillna(0).astype("int32")
    return output.dropna(subset=["timestamp", "open", "high", "low", "close"]).drop_duplicates("timestamp")


def build_prices() -> tuple[pd.DataFrame, dict]:
    out_dir = ROOT / "data" / "prices"
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_parts: list[pd.DataFrame] = []
    coverage: dict[str, dict] = {}
    yesterday = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)
    for symbol in SYMBOLS:
        prefix = f"data/spot/monthly/klines/{symbol}/1m/"
        monthly = sorted(key for key in list_s3(prefix) if key.endswith(".zip"))
        if not monthly:
            raise RuntimeError(f"No Binance monthly archives for {symbol}")
        latest_month = pd.Period(Path(monthly[-1]).stem.rsplit("-", 2)[-2] + "-" + Path(monthly[-1]).stem.rsplit("-", 1)[-1], freq="M")
        daily_start = latest_month.end_time.tz_localize("UTC").normalize() + pd.Timedelta(days=1)
        daily = []
        for date in pd.date_range(daily_start, yesterday, freq="D"):
            key = f"data/spot/daily/klines/{symbol}/1m/{symbol}-1m-{date.date()}.zip"
            head = SESSION.head(BINANCE_DOWNLOAD + key, timeout=30)
            if head.status_code == 200:
                daily.append(key)
        all_keys = monthly + daily
        by_year: dict[int, list[str]] = {}
        for key in all_keys:
            name = Path(key).stem
            year = int(name.split("-")[2])
            by_year.setdefault(year, []).append(key)
        symbol_frames = []
        for year, keys in sorted(by_year.items()):
            pieces = []
            for index, key in enumerate(keys, 1):
                print(f"[{symbol} {year}] {index}/{len(keys)} {Path(key).name}", flush=True)
                pieces.append(read_kline_zip(key))
            frame = pd.concat(pieces, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp")
            frame = frame[frame["timestamp"].dt.year == year].reset_index(drop=True)
            path = out_dir / f"{symbol}_{year}.parquet"
            frame.to_parquet(path, index=False, compression="zstd", row_group_size=1440)
            day_counts = frame.groupby(frame["timestamp"].dt.normalize()).size().rename("rows").rename_axis("date").reset_index()
            day_counts.insert(0, "symbol", symbol)
            day_counts["file"] = str(path.relative_to(ROOT))
            catalog_parts.append(day_counts)
            symbol_frames.append(frame[["timestamp", "open", "high", "low", "close", "volume"]])
        combined = pd.concat(symbol_frames, ignore_index=True)
        coverage[symbol] = {
            "start": str(combined["timestamp"].min()),
            "end": str(combined["timestamp"].max()),
            "rows": int(len(combined)),
            "source": "Binance spot 1m public data",
        }
    catalog = pd.concat(catalog_parts, ignore_index=True)
    catalog.to_csv(ROOT / "data" / "price_catalog.csv", index=False)
    return catalog, coverage


def read_fred(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd=2017-01-01&coed={pd.Timestamp.now(tz='UTC').date()}"
    frame = pd.read_csv(io.BytesIO(get(url).content))
    frame.columns = ["date", series_id]
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame[series_id] = pd.to_numeric(frame[series_id], errors="coerce")
    return frame.set_index("date")[series_id].sort_index()


def coinmetrics() -> pd.DataFrame:
    url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    params = {
        "assets": "btc,eth,ltc",
        "metrics": "CapMrktCurUSD,TxCnt,AdrActCnt,HashRate",
        "frequency": "1d",
        "page_size": 10000,
    }
    rows = []
    while True:
        response = SESSION.get(url, params=params, timeout=180)
        response.raise_for_status()
        payload = response.json()
        rows.extend(payload.get("data", []))
        token = payload.get("next_page_token")
        if not token:
            break
        params["next_page_token"] = token
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame.pop("time"), utc=True).dt.normalize()
    frame["symbol"] = frame.pop("asset").str.upper() + "USDT"
    for col in ("CapMrktCurUSD", "TxCnt", "AdrActCnt", "HashRate"):
        frame[col] = pd.to_numeric(frame.get(col), errors="coerce")
    return frame.sort_values(["symbol", "date"])


def causal_rolling_z(series: pd.Series, window: int = 365) -> pd.Series:
    mean = series.rolling(window, min_periods=30).mean()
    std = series.rolling(window, min_periods=30).std().replace(0.0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)


def build_context(catalog: pd.DataFrame) -> dict:
    daily_parts = []
    for path in sorted((ROOT / "data" / "prices").glob("*USDT_*.parquet")):
        symbol = path.name.split("_")[0]
        frame = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close", "volume"])
        frame["date"] = frame["timestamp"].dt.normalize()
        frame["minute_return"] = np.log(frame["close"].clip(lower=1e-12)).diff()
        grouped = frame.groupby("date", sort=True)
        daily = grouped.agg(open=("open", "first"), close=("close", "last"), volume=("volume", "sum"))
        daily["asset_daily_return"] = np.log(daily["close"] / daily["open"])
        daily["asset_realized_vol"] = grouped["minute_return"].std() * np.sqrt(1440.0)
        daily["symbol"] = symbol
        daily_parts.append(daily.reset_index())
    daily_assets = pd.concat(daily_parts, ignore_index=True).sort_values(["symbol", "date"])

    fng_payload = get("https://api.alternative.me/fng/?limit=0&format=json").json()
    fng = pd.DataFrame(fng_payload["data"])
    fng["date"] = pd.to_datetime(pd.to_numeric(fng["timestamp"]), unit="s", utc=True).dt.normalize()
    fng["fear_greed"] = pd.to_numeric(fng["value"], errors="coerce")
    fng = fng.set_index("date")["fear_greed"].sort_index()
    macro = pd.concat(
        {
            "vix": read_fred("VIXCLS"),
            "dollar": read_fred("DTWEXBGS"),
            "sp500": read_fred("SP500"),
            "dgs10": read_fred("DGS10"),
            "dff": read_fred("DFF"),
        },
        axis=1,
    ).sort_index().ffill()
    macro["sp500_return"] = np.log(macro["sp500"] / macro["sp500"].shift(1))
    metrics = coinmetrics()

    rows = []
    for symbol, asset in daily_assets.groupby("symbol", sort=False):
        asset = asset.set_index("date").sort_index()
        index = pd.date_range(asset.index.min(), asset.index.max() + pd.Timedelta(days=1), freq="D", tz="UTC")
        base = pd.DataFrame(index=index)
        # Lag all completed-day asset and on-chain data by one day to prevent leakage.
        base = base.join(asset[["asset_daily_return", "asset_realized_vol"]].shift(1))
        chain = metrics[metrics["symbol"] == symbol].set_index("date").sort_index()
        chain_features = pd.DataFrame(index=chain.index)
        chain_features["active_addresses_z"] = causal_rolling_z(np.log1p(chain["AdrActCnt"])).shift(1)
        chain_features["transaction_count_z"] = causal_rolling_z(np.log1p(chain["TxCnt"])).shift(1)
        chain_features["hash_rate_z"] = causal_rolling_z(np.log1p(chain["HashRate"])).shift(1)
        chain_features["market_cap_return"] = np.log(chain["CapMrktCurUSD"] / chain["CapMrktCurUSD"].shift(1)).shift(1)
        base = base.join(chain_features)
        base = base.join(macro.shift(1)).join(fng.rename("fear_greed").shift(1)).ffill()
        result = pd.DataFrame(index=base.index)
        result["fear_greed_scaled"] = ((base["fear_greed"] - 50.0) / 50.0).clip(-1, 1)
        result["vix_scaled"] = ((base["vix"] - 20.0) / 20.0).clip(-2, 3)
        result["dollar_index_scaled"] = causal_rolling_z(base["dollar"])
        result["sp500_return_clipped"] = (base["sp500_return"] / 0.03).clip(-3, 3)
        result["ten_year_yield_scaled"] = ((base["dgs10"] - 3.0) / 3.0).clip(-2, 3)
        result["fed_funds_scaled"] = ((base["dff"] - 3.0) / 3.0).clip(-2, 3)
        result["active_addresses_z"] = base["active_addresses_z"]
        result["transaction_count_z"] = base["transaction_count_z"]
        result["hash_rate_z"] = base["hash_rate_z"]
        result["market_cap_return_clipped"] = (base["market_cap_return"] / 0.1).clip(-3, 3)
        result["asset_daily_return_clipped"] = (base["asset_daily_return"] / 0.1).clip(-3, 3)
        result["asset_realized_vol_scaled"] = (base["asset_realized_vol"] / 0.05).clip(0, 5)
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result.insert(0, "symbol", symbol)
        result = result.rename_axis("date").reset_index()
        rows.append(result)
    context = pd.concat(rows, ignore_index=True)
    path = ROOT / "data" / "context" / "daily_market_context.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    context.to_parquet(path, index=False, compression="zstd")
    return {
        "start": str(context["date"].min()),
        "end": str(context["date"].max()),
        "rows": int(len(context)),
        "sources": ["Alternative.me Fear & Greed", "FRED", "Coin Metrics Community", "lagged Binance aggregates"],
    }


def build_news(seven_zip: str) -> dict:
    source_url = "https://raw.githubusercontent.com/soheilrahsaz/cryptoNewsDataset/main/csvOutput/news_currencies_source_joinedResult.rar"
    with tempfile.TemporaryDirectory() as temp_name:
        temp = Path(temp_name)
        archive = temp / "news.rar"
        with get(source_url, stream=True) as response, archive.open("wb") as handle:
            for block in response.iter_content(1024 * 1024):
                handle.write(block)
        subprocess.run([seven_zip, "x", "-y", str(archive), f"-o{temp}"], check=True, stdout=subprocess.DEVNULL)
        csv_path = temp / "news_currencies_source_joinedResult.csv"
        frame = pd.read_csv(
            csv_path,
            usecols=["id", "title", "description", "sourceDomain", "newsDatetime", "url", "negative", "positive", "important", "currencies"],
            low_memory=False,
        )
    frame["timestamp"] = pd.to_datetime(frame["newsDatetime"], utc=True, errors="coerce")
    frame["currencies"] = frame["currencies"].fillna("").str.split(",")
    frame = frame.explode("currencies")
    frame["asset"] = frame["currencies"].str.strip().str.upper()
    frame = frame[frame["asset"].isin(["BTC", "ETH", "LTC"]) & frame["timestamp"].notna()]
    frame = frame.rename(columns={"sourceDomain": "source"})
    output = frame[["id", "asset", "timestamp", "title", "description", "source", "url", "negative", "positive", "important"]]
    output = output.sort_values(["asset", "timestamp", "id"]).drop_duplicates(["asset", "id"])
    path = ROOT / "data" / "news" / "crypto_news.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(path, index=False, compression="zstd")
    return {
        "start": str(output["timestamp"].min()),
        "end": str(output["timestamp"].max()),
        "rows": int(len(output)),
        "rows_by_asset": {k: int(v) for k, v in output.groupby("asset").size().items()},
        "source": "soheilrahsaz/cryptoNewsDataset (CC0-1.0)",
    }


def build_model() -> dict:
    repository = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    revision = "ae0eab9ad336d7d548e0efe394b07c04bcaf6e91"
    out = ROOT / "models" / "financial_sentiment"
    out.mkdir(parents=True, exist_ok=True)
    small_files = ["README.md", "config.json", "merges.txt", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"]
    for name in small_files:
        content = get(f"https://huggingface.co/{repository}/resolve/{revision}/{name}").content
        target_name = "UPSTREAM_README.md" if name == "README.md" else name
        (out / target_name).write_bytes(content)
    weights = out / "model.safetensors"
    with get(f"https://huggingface.co/{repository}/resolve/{revision}/model.safetensors", stream=True) as response, weights.open("wb") as handle:
        for block in response.iter_content(8 * 1024 * 1024):
            handle.write(block)
    expected = "c0b61385e4482edd179b69042c014dcb53a79431784f34a0171f5d43b092feaa"
    if sha256(weights) != expected:
        raise ValueError("Downloaded sentiment model hash did not match the pinned revision")
    parts = []
    chunk_size = 90 * 1024 * 1024
    with weights.open("rb") as source:
        index = 0
        while block := source.read(chunk_size):
            name = f"model.safetensors.part-{index:03d}"
            path = out / name
            path.write_bytes(block)
            parts.append({"name": name, "size": path.stat().st_size, "sha256": sha256(path)})
            index += 1
    manifest = {
        "target": "model.safetensors",
        "size": weights.stat().st_size,
        "sha256": expected,
        "parts": parts,
        "upstream": repository,
        "revision": revision,
        "license": "Apache-2.0",
    }
    (out / "chunks.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "LICENSE").write_text(get("https://www.apache.org/licenses/LICENSE-2.0.txt").text, encoding="utf-8")
    weights.unlink()
    return {"upstream": repository, "revision": revision, "weight_bytes": manifest["size"], "parts": len(parts), "license": "Apache-2.0"}


def build_manifest(coverage: dict) -> None:
    paths = [ROOT / "data" / "price_catalog.csv", ROOT / "data" / "context" / "daily_market_context.parquet", ROOT / "data" / "news" / "crypto_news.parquet"]
    paths.extend(sorted((ROOT / "data" / "prices").glob("*.parquet")))
    model_dir = ROOT / "models" / "financial_sentiment"
    paths.extend(sorted(path for path in model_dir.iterdir() if path.is_file() and path.name != "model.safetensors"))
    files = [{"path": str(path.relative_to(ROOT)), "size": path.stat().st_size, "sha256": sha256(path)} for path in paths]
    payload = {
        "created_utc": str(pd.Timestamp.now(tz="UTC")),
        "coverage": coverage,
        "files": files,
    }
    (ROOT / "data" / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seven-zip", required=True, help="Path to a 7zz binary with RAR5 support")
    args = parser.parse_args()
    catalog, prices = build_prices()
    coverage = {
        "prices": prices,
        "context": build_context(catalog),
        "news": build_news(args.seven_zip),
        "model": build_model(),
    }
    build_manifest(coverage)
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    main()
