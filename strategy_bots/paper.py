"""Paper account for the daily bot: fills at the given prices plus costs, saved as JSON.

It rebalances at most once per date, holds fractional shares, and never contacts a broker. Targets decided
at a close wait in `pending` until the next open, the same timing as the backtest.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class PaperAccount:
    cash: float
    positions: dict[str, float] = field(default_factory=dict)
    last_rebalance: str | None = None
    history: list[dict] = field(default_factory=list)
    pending: dict | None = None  # {"decided": date, "targets": {symbol: weight}}, filled at the next open

    @classmethod
    def open(cls, path: Path, capital: float) -> PaperAccount:
        path = Path(path)
        if path.exists():
            return cls(**json.loads(path.read_text()))
        return cls(cash=capital)

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
