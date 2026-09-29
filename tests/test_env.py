from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from crypto_rl.data import CONTEXT_FEATURES, MARKET_FEATURES, DayData, EpisodeKey, engineer_market_features
from crypto_rl.env import CryptoTradingEnv


class EnvironmentTests(unittest.TestCase):
    def make_day(self, lookback: int = 60) -> DayData:
        timestamps = pd.date_range("2024-01-01", periods=1440, freq="min", tz="UTC").to_numpy()
        prices = np.linspace(100.0, 110.0, 1440)
        features = np.zeros((lookback - 1 + 1440, len(MARKET_FEATURES)), dtype=np.float32)
        return DayData(
            key=EpisodeKey("BTCUSDT", pd.Timestamp("2024-01-01", tz="UTC")),
            timestamps=timestamps,
            prices=prices,
            market_features=features,
            daily_context=np.zeros(len(CONTEXT_FEATURES), dtype=np.float32),
        )

    def make_env(self) -> CryptoTradingEnv:
        return CryptoTradingEnv(60, 2000.0, 0.001, 2.0, [0.0, 0.5, 1.0], 0.0, 0.0, 100.0)

    def test_episode_length_and_nonnegative_balances(self) -> None:
        env = self.make_env()
        obs = env.reset(self.make_day(), np.zeros((24, 4), dtype=np.float32))
        self.assertEqual(obs.shape, (env.observation_dim,))
        done = False
        steps = 0
        while not done:
            result = env.step(steps % env.action_dim)
            done = result.terminated
            steps += 1
            self.assertGreaterEqual(result.info["cash"], -1e-7)
            self.assertGreaterEqual(result.info["units"], -1e-12)
        self.assertEqual(steps, 1439)
        self.assertEqual(len(env.equity_curve), 1440)

    def test_transaction_costs_reduce_flat_market_equity(self) -> None:
        day = self.make_day()
        day.prices[:] = 100.0
        env = self.make_env()
        env.reset(day)
        env.step(2)  # 100% allocation
        result = env.step(0)  # back to cash
        self.assertLess(result.info["equity"], 2000.0)
        self.assertGreater(env.total_fees, 0.0)

    def test_features_do_not_look_forward(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=400, freq="min", tz="UTC")
        base = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": np.linspace(100, 105, 400),
                "high": np.linspace(101, 106, 400),
                "low": np.linspace(99, 104, 400),
                "close": np.linspace(100, 105, 400),
                "volume": np.arange(400) + 1,
                "quote_volume": (np.arange(400) + 1) * 100,
                "trade_count": np.arange(400) + 10,
            }
        )
        altered = base.copy()
        altered.loc[300:, "close"] *= 5
        original_features = engineer_market_features(base)
        altered_features = engineer_market_features(altered)
        np.testing.assert_allclose(original_features[:300], altered_features[:300], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
