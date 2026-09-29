from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class AssetTests(unittest.TestCase):
    def test_manifest_sizes(self) -> None:
        manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
        self.assertGreater(len(manifest["files"]), 30)
        for item in manifest["files"]:
            path = ROOT / item["path"]
            self.assertTrue(path.exists(), item["path"])
            self.assertEqual(path.stat().st_size, item["size"], item["path"])

    def test_model_parts_are_git_safe(self) -> None:
        model = json.loads((ROOT / "models" / "financial_sentiment" / "chunks.json").read_text(encoding="utf-8"))
        self.assertEqual(sum(item["size"] for item in model["parts"]), model["size"])
        for item in model["parts"]:
            self.assertLess(item["size"], 100 * 1024 * 1024)

    def test_price_catalog_contains_complete_days_for_all_assets(self) -> None:
        catalog = pd.read_csv(ROOT / "data" / "price_catalog.csv")
        complete = catalog[catalog["rows"] == 1440]
        self.assertEqual(set(complete["symbol"]), {"BTCUSDT", "ETHUSDT", "LTCUSDT"})
        self.assertGreater(len(complete), 9000)

    def test_news_assets(self) -> None:
        news = pd.read_parquet(ROOT / "data" / "news" / "crypto_news.parquet", columns=["asset"])
        self.assertEqual(set(news["asset"]), {"BTC", "ETH", "LTC"})
        self.assertGreater(len(news), 50_000)


if __name__ == "__main__":
    unittest.main()
