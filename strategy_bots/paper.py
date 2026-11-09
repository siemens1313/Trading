"""Paper account for the daily bot: fills at the given prices plus costs, saved as JSON.

It rebalances at most once per date, holds fractional shares, and never contacts a broker. Targets decided
at a close wait in `pending` until the next open, the same timing as the backtest. Because no strategy
looks ahead, an account replayed from its first day ends exactly where daily runs would have left it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .data import Panel


@dataclass
class PaperAccount:
    cash: float
    positions: dict[str, float] = field(default_factory=dict)
    last_rebalance: str | None = None
    history: list[dict] = field(default_factory=list)
    pending: dict | None = None  # {"decided": date, "targets": {symbol: weight}}, filled at the next open
    capital: float | None = None  # starting cash
    opened: str | None = None  # the close of the first decision
    marks: dict[str, float] = field(default_factory=dict)  # value at each close since then

    @classmethod
    def open(cls, path: Path, capital: float) -> PaperAccount:
        path = Path(path)
        if path.exists():
            return cls(**json.loads(path.read_text()))
        return cls(cash=capital, capital=capital)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1))

    def value(self, prices: dict[str, float]) -> float:
        missing = [s for s in self.positions if s not in prices]
        if missing:
            raise KeyError(f"no price for held symbol(s): {', '.join(missing)}")
        return self.cash + sum(shares * prices[s] for s, shares in self.positions.items())

    def rebalance(
        self, targets: dict[str, float], prices: dict[str, float], as_of: str, cost_bps: float = 5.0, band: float = 0.01
    ) -> list[dict]:
        """Trade to the target weights at the given prices. A second call for the same date does nothing.

        A trade worth less than `band` of equity is skipped unless it opens or closes a position, so price
        drift alone doesn't cause a stream of tiny trades.
        """
        if as_of == self.last_rebalance:
            return []
        equity = self.value(prices)
        fills = []
        for symbol in sorted(set(self.positions) | set(targets)):
            price = prices[symbol]
            have = self.positions.get(symbol, 0.0)
            want = round(targets.get(symbol, 0.0) * equity / price, 4)
            delta = round(want - have, 4)
            notional = delta * price
            if not delta or (abs(notional) < band * equity and bool(want) == bool(have)):
                continue
            fee = abs(notional) * cost_bps / 1e4
            self.cash -= notional + fee
            if want:
                self.positions[symbol] = want
            else:
                self.positions.pop(symbol, None)
            fills.append({"date": as_of, "symbol": symbol, "shares": delta, "price": price, "fee": round(fee, 2)})
        self.last_rebalance = as_of
        self.history.append({"date": as_of, "equity": round(equity, 2), "fills": len(fills)})
        return fills


def targets(symbols: list[str], row: np.ndarray) -> dict[str, float]:
    """A row of strategy weights as {symbol: weight}, without the zeros."""
    return {sym: round(float(w), 6) for sym, w in zip(symbols, row) if abs(w) > 1e-9}


def advance(
    account: PaperAccount, panel: Panel, weights: np.ndarray, cost_bps: float = 5.0, since: str | None = None
) -> list[dict]:
    """Trade the account through every bar it hasn't seen yet and return the fills.

    For each new day: fill the pending targets at the open, mark the account at the close, and take that
    close's row of weights as the targets for the next open. A new account makes its first decision at the
    latest close, or at the first close on or after `since`, replaying the days in between.
    """
    dates = [str(d) for d in panel.dates]
    if account.pending:
        if account.pending["decided"] > dates[-1]:
            raise ValueError(
                f"the account has targets from {account.pending['decided']}, after the last bar ({dates[-1]}): is the data stale?"
            )
        first = next((t for t, d in enumerate(dates) if d > account.pending["decided"]), len(dates) - 1)
    elif since:
        first = next((t for t, d in enumerate(dates) if d >= since), None)
        if first is None:
            raise ValueError(f"no bars on or after {since}")
    else:
        first = len(dates) - 1
    account.opened = account.opened or dates[first]
    if account.capital is None:  # an account saved before capital was recorded
        account.capital = account.history[0]["equity"] if account.history else account.cash
    fills = []
    for t in range(first, len(dates)):
        if account.pending and account.pending["decided"] < dates[t]:
            fills += account.rebalance(account.pending["targets"], dict(zip(panel.symbols, map(float, panel.open[t]))), dates[t], cost_bps)
        account.pending = {"decided": dates[t], "targets": targets(panel.symbols, weights[t])}
        account.marks[dates[t]] = round(account.value(dict(zip(panel.symbols, map(float, panel.close[t])))), 2)
    return fills
