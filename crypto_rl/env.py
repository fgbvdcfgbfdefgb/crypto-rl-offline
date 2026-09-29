"""A causal, long-only spot-crypto environment with realistic frictions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import CONTEXT_FEATURES, MARKET_FEATURES, NEWS_FEATURES, DayData

PORTFOLIO_FEATURES = ("cash_fraction", "asset_fraction", "allocation", "drawdown")
STATE_DIM = len(PORTFOLIO_FEATURES) + len(CONTEXT_FEATURES) + len(NEWS_FEATURES)


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    info: dict


class CryptoTradingEnv:
    """One episode is one complete UTC day of minute bars.

    Action i targets ``target_allocations[i]`` of current equity in the selected
    asset. Orders execute at the current minute close plus/minus slippage and a
    fee. The next observation/reward uses the next minute close.
    """

    def __init__(
        self,
        lookback: int,
        initial_balance: float,
        fee_rate: float,
        slippage_bps: float,
        target_allocations: list[float],
        turnover_penalty: float,
        drawdown_penalty: float,
        reward_scale: float,
    ) -> None:
        self.lookback = int(lookback)
        self.initial_balance = float(initial_balance)
        self.fee_rate = float(fee_rate)
        self.slippage = float(slippage_bps) / 10_000.0
        self.target_allocations = np.asarray(target_allocations, dtype=np.float64)
        self.turnover_penalty = float(turnover_penalty)
        self.drawdown_penalty = float(drawdown_penalty)
        self.reward_scale = float(reward_scale)
        self.day: DayData | None = None
        self.news_profile = np.zeros((24, len(NEWS_FEATURES)), dtype=np.float32)

    @property
    def action_dim(self) -> int:
        return len(self.target_allocations)

    @property
    def observation_dim(self) -> int:
        return self.lookback * len(MARKET_FEATURES) + STATE_DIM

    def reset(self, day: DayData, news_profile: np.ndarray | None = None) -> np.ndarray:
        self.day = day
        profile = np.asarray(news_profile if news_profile is not None else np.zeros((24, 4)), dtype=np.float32)
        if profile.shape != (24, len(NEWS_FEATURES)):
            raise ValueError(f"news_profile must have shape (24, {len(NEWS_FEATURES)}), got {profile.shape}")
        self.news_profile = profile
        self.t = 0
        self.cash = self.initial_balance
        self.units = 0.0
        self.total_fees = 0.0
        self.total_turnover = 0.0
        self.peak_equity = self.initial_balance
        self.max_drawdown = 0.0
        self.trades: list[dict] = []
        self.equity_curve = [self.initial_balance]
        self.cash_curve = [self.cash]
        self.units_curve = [self.units]
        self.price_curve = [float(day.prices[0])]
        self.action_curve: list[int] = []
        self.reward_curve: list[float] = []
        return self._observation()

    def _equity(self, price: float) -> float:
        return max(1e-12, self.cash + self.units * price)

    def _observation(self) -> np.ndarray:
        assert self.day is not None
        window = self.day.market_features[self.t : self.t + self.lookback]
        if window.shape != (self.lookback, len(MARKET_FEATURES)):
            raise RuntimeError(f"Bad feature window at step {self.t}: {window.shape}")
        price = float(self.day.prices[self.t])
        equity = self._equity(price)
        asset_value = self.units * price
        drawdown = max(0.0, 1.0 - equity / max(self.peak_equity, 1e-12))
        portfolio = np.asarray(
            [self.cash / equity, asset_value / equity, asset_value / equity, drawdown],
            dtype=np.float32,
        )
        timestamp = self.day.timestamps[self.t]
        hour = int(str(timestamp)[11:13])
        state = np.concatenate((portfolio, self.day.daily_context, self.news_profile[hour])).astype(np.float32)
        return np.concatenate((window.reshape(-1), state)).astype(np.float32)

    def step(self, action: int) -> StepResult:
        assert self.day is not None
        if not 0 <= int(action) < self.action_dim:
            raise ValueError(f"Invalid action {action}")
        if self.t >= len(self.day.prices) - 1:
            raise RuntimeError("Episode is already finished")

        action = int(action)
        price = float(self.day.prices[self.t])
        equity_before = self._equity(price)
        target_fraction = float(self.target_allocations[action])
        target_asset_value = target_fraction * equity_before
        current_asset_value = self.units * price
        desired_notional = target_asset_value - current_asset_value
        side = "hold"
        executed_notional = 0.0
        fee = 0.0

        if desired_notional > 1e-8 and self.cash > 1e-8:
            # Cost is notional plus fee. Cap the order so cash can never be negative.
            max_notional = self.cash / (1.0 + self.fee_rate)
            executed_notional = min(desired_notional, max_notional)
            execution_price = price * (1.0 + self.slippage)
            units_bought = executed_notional / execution_price
            fee = executed_notional * self.fee_rate
            self.cash -= executed_notional + fee
            self.units += units_bought
            side = "buy"
        elif desired_notional < -1e-8 and self.units > 1e-12:
            execution_price = price * (1.0 - self.slippage)
            desired_units = (-desired_notional) / execution_price
            units_sold = min(self.units, desired_units)
            executed_notional = units_sold * execution_price
            fee = executed_notional * self.fee_rate
            self.units -= units_sold
            self.cash += executed_notional - fee
            side = "sell"

        if self.cash < 0 and self.cash > -1e-7:
            self.cash = 0.0
        if self.units < 0 and self.units > -1e-12:
            self.units = 0.0
        if self.cash < -1e-7 or self.units < -1e-12:
            raise RuntimeError("Accounting invariant violated: negative cash or units")

        self.total_fees += fee
        turnover = abs(executed_notional) / max(equity_before, 1e-12)
        self.total_turnover += turnover
        if side != "hold":
            self.trades.append(
                {
                    "step": self.t,
                    "timestamp": str(self.day.timestamps[self.t]),
                    "side": side,
                    "market_price": price,
                    "notional": float(executed_notional),
                    "fee": float(fee),
                    "target_allocation": target_fraction,
                }
            )

        next_t = self.t + 1
        next_price = float(self.day.prices[next_t])
        equity_after = self._equity(next_price)
        self.peak_equity = max(self.peak_equity, equity_after)
        drawdown = max(0.0, 1.0 - equity_after / max(self.peak_equity, 1e-12))
        self.max_drawdown = max(self.max_drawdown, drawdown)
        log_return = np.log(max(equity_after, 1e-12) / max(equity_before, 1e-12))
        reward = self.reward_scale * (
            log_return - self.turnover_penalty * turnover - self.drawdown_penalty * drawdown / 1440.0
        )

        self.t = next_t
        self.equity_curve.append(float(equity_after))
        self.cash_curve.append(float(self.cash))
        self.units_curve.append(float(self.units))
        self.price_curve.append(next_price)
        self.action_curve.append(action)
        self.reward_curve.append(float(reward))
        terminated = self.t >= len(self.day.prices) - 1
        info = {
            "equity": float(equity_after),
            "cash": float(self.cash),
            "units": float(self.units),
            "price": next_price,
            "drawdown": float(drawdown),
            "turnover": float(turnover),
            "fee": float(fee),
            "side": side,
        }
        return StepResult(self._observation(), float(reward), terminated, info)

    def summary(self) -> dict:
        assert self.day is not None
        ending_equity = float(self.equity_curve[-1])
        profit = ending_equity - self.initial_balance
        returns = np.diff(np.log(np.maximum(np.asarray(self.equity_curve), 1e-12)))
        sharpe = 0.0
        if len(returns) > 1 and float(returns.std()) > 1e-12:
            sharpe = float(np.sqrt(365.0 * 1440.0) * returns.mean() / returns.std())
        return {
            "symbol": self.day.key.symbol,
            "date": str(self.day.key.date.date()),
            "starting_balance": self.initial_balance,
            "ending_equity": ending_equity,
            "profit": float(profit),
            "return_pct": float(100.0 * profit / self.initial_balance),
            "max_drawdown_pct": float(100.0 * self.max_drawdown),
            "total_fees": float(self.total_fees),
            "total_turnover": float(self.total_turnover),
            "trade_count": len(self.trades),
            "minute_sharpe_annualized": sharpe,
            "final_cash": float(self.cash),
            "final_units": float(self.units),
        }
