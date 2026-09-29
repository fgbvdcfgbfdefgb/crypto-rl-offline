"""Per-episode audit plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .env import CryptoTradingEnv


def save_episode_plot(
    env: CryptoTradingEnv,
    summary: dict,
    news_profile: np.ndarray,
    path: str | Path,
    agent_id: int,
    episode: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    price = np.asarray(env.price_curve)
    equity = np.asarray(env.equity_curve)
    cash = np.asarray(env.cash_curve)
    drawdown = 100.0 * (1.0 - equity / np.maximum.accumulate(equity))
    x = np.arange(len(price))

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    grid = fig.add_gridspec(4, 1, height_ratios=[2.0, 1.5, 1.2, 1.3])
    ax_price = fig.add_subplot(grid[0])
    ax_equity = fig.add_subplot(grid[1], sharex=ax_price)
    ax_risk = fig.add_subplot(grid[2], sharex=ax_price)
    ax_info = fig.add_subplot(grid[3])

    ax_price.plot(x, price, color="#1f77b4", linewidth=1.0, label="Close price")
    buys = [t for t in env.trades if t["side"] == "buy"]
    sells = [t for t in env.trades if t["side"] == "sell"]
    if buys:
        ax_price.scatter([t["step"] for t in buys], [t["market_price"] for t in buys], marker="^", s=28, color="#2ca02c", label="Buy", zorder=3)
    if sells:
        ax_price.scatter([t["step"] for t in sells], [t["market_price"] for t in sells], marker="v", s=28, color="#d62728", label="Sell", zorder=3)
    ax_price.set_ylabel("USDT")
    ax_price.set_title(f"Agent {agent_id} · Episode {episode} · {summary['symbol']} · {summary['date']}")
    ax_price.legend(loc="upper left", ncol=3)
    ax_price.grid(alpha=0.2)

    ax_equity.plot(x, equity, color="#9467bd", linewidth=1.2, label="Equity")
    ax_equity.plot(x, cash, color="#7f7f7f", linewidth=0.8, alpha=0.8, label="Cash")
    ax_equity.axhline(summary["starting_balance"], color="black", linestyle="--", linewidth=0.8, label="Start")
    ax_equity.set_ylabel("USDT")
    ax_equity.legend(loc="upper left", ncol=3)
    ax_equity.grid(alpha=0.2)

    ax_risk.fill_between(x, drawdown, color="#d62728", alpha=0.25, label="Drawdown %")
    ax_risk.plot(x, drawdown, color="#d62728", linewidth=0.8)
    allocations = np.asarray(env.target_allocations)[np.asarray(env.action_curve, dtype=int)] if env.action_curve else np.asarray([])
    allocation_line = np.r_[allocations[0] if len(allocations) else 0.0, allocations] * 100.0
    ax_alloc = ax_risk.twinx()
    ax_alloc.step(x, allocation_line, where="post", color="#ff7f0e", alpha=0.65, linewidth=0.8, label="Target allocation %")
    ax_risk.set_ylabel("Drawdown %")
    ax_alloc.set_ylabel("Allocation %")
    ax_risk.set_xlabel("UTC minute of episode")
    ax_risk.grid(alpha=0.2)

    ax_info.axis("off")
    sentiment = news_profile[:, 0]
    info = (
        f"Start: ${summary['starting_balance']:,.2f}     End: ${summary['ending_equity']:,.2f}     "
        f"Profit: ${summary['profit']:,.2f} ({summary['return_pct']:+.2f}%)\n"
        f"Trades: {summary['trade_count']}     Fees: ${summary['total_fees']:,.2f}     "
        f"Turnover: {summary['total_turnover']:.2f}x     Max drawdown: {summary['max_drawdown_pct']:.2f}%\n"
        f"Annualized minute Sharpe (diagnostic only): {summary['minute_sharpe_annualized']:.2f}     "
        f"Final cash: ${summary['final_cash']:,.2f}     Final units: {summary['final_units']:.8f}\n"
        f"Causal news sentiment by UTC hour — min {sentiment.min():+.3f}, mean {sentiment.mean():+.3f}, max {sentiment.max():+.3f}. "
        "A value at hour H uses only headlines published before H:00."
    )
    ax_info.text(0.01, 0.95, info, ha="left", va="top", fontsize=11, family="monospace")
    hours = np.arange(24)
    inset = ax_info.inset_axes([0.04, 0.05, 0.92, 0.38])
    inset.axhline(0, color="black", linewidth=0.6)
    inset.plot(hours, sentiment, marker="o", markersize=2.5, linewidth=1.0, color="#17becf")
    inset.set_xlim(0, 23)
    inset.set_ylabel("news score")
    inset.set_xlabel("UTC hour")
    inset.grid(alpha=0.2)

    fig.savefig(path, dpi=140)
    plt.close(fig)
