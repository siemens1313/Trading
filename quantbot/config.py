from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class GeneralCfg:
    mode: str = "paper"
    data_provider: str = "csv"
    data_dir: str = "data"
    state_dir: str = "state"
    reports_dir: str = "reports"


@dataclass
class UniverseCfg:
    symbols: list[str] = field(default_factory=lambda: ["SPY", "TLT", "GLD"])
    benchmark: str = "SPY"


@dataclass
class StrategyCfg:
    momentum_lookbacks: list[int] = field(default_factory=lambda: [63, 126, 252])
    skip_recent: int = 0
    trend_window: int = 200
    vol_window: int = 63
    top_n: int = 3
    target_vol: float = 0.12
    max_weight: float = 0.40
    rebalance: str = "monthly"


@dataclass
class ExecutionCfg:
    slippage_bps: float = 5.0
    commission_per_order: float = 0.0
    fractional: bool = True
    min_trade_weight: float = 0.01
    min_order_notional: float = 5.0


@dataclass
class RiskCfg:
    starting_cash: float = 100_000.0
    max_capital: float = 1_000.0
    kill_switch_drawdown: float = 0.30
    max_order_notional: float = 50_000.0
    max_price_deviation: float = 0.10


@dataclass
class Config:
    general: GeneralCfg = field(default_factory=GeneralCfg)
    universe: UniverseCfg = field(default_factory=UniverseCfg)
    strategy: StrategyCfg = field(default_factory=StrategyCfg)
    execution: ExecutionCfg = field(default_factory=ExecutionCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)
    root: Path = Path(".")

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def validate(self) -> None:
        s, g = self.strategy, self.general
        if g.mode not in ("paper", "live"):
            raise ValueError(f"general.mode must be 'paper' or 'live', got {g.mode!r}")
        if g.data_provider not in ("csv", "yahoo", "robinhood"):
            raise ValueError(f"unknown data_provider {g.data_provider!r}")
        if s.rebalance not in ("daily", "weekly", "monthly"):
            raise ValueError(f"strategy.rebalance must be daily/weekly/monthly, got {s.rebalance!r}")
        if not (0 < s.max_weight <= 1):
            raise ValueError("strategy.max_weight must be in (0, 1]")
        if s.top_n < 1 or not s.momentum_lookbacks:
            raise ValueError("strategy.top_n must be >= 1 and momentum_lookbacks non-empty")
        if s.top_n * s.max_weight < 1 - 1e-9:
            # Not an error, but worth knowing: the portfolio can never be fully invested.
            pass
        if self.universe.benchmark not in self.universe.symbols:
            raise ValueError("universe.benchmark must be one of universe.symbols")


def _build(cls, data: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown keys in [{cls.__name__}]: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path = "config.toml") -> Config:
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    cfg = Config(
        general=_build(GeneralCfg, raw.get("general", {})),
        universe=_build(UniverseCfg, raw.get("universe", {})),
        strategy=_build(StrategyCfg, raw.get("strategy", {})),
        execution=_build(ExecutionCfg, raw.get("execution", {})),
        risk=_build(RiskCfg, raw.get("risk", {})),
        root=path.resolve().parent,
    )
    cfg.validate()
    return cfg
