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
    max_quote_age_minutes: float = 30.0


@dataclass
class UniverseCfg:
    symbols: list[str] = field(default_factory=lambda: ["SPY", "TLT", "GLD"])
    benchmark: str = "SPY"


@dataclass
class AlphaCfg:
    enabled: list[str] = field(default_factory=lambda: ["tsmom", "xsmom", "strev", "lowvol"])
    horizon: int = 5
    ic_window: int = 252
    ic_min_obs: int = 126
    prior_weight: float = 0.3
    ic_haircut: float = 0.5
    params: dict = field(default_factory=dict)


@dataclass
class RiskModelCfg:
    vol_halflife: int = 21
    corr_window: int = 126


@dataclass
class PortfolioCfg:
    long_only: bool = True
    max_weight: float = 0.20
    max_gross: float = 1.0
    target_vol: float = 0.12
    risk_aversion: float = 10.0
    calibrate_risk: bool = True
    tcost_aversion: float = 1.0
    prior_premium: float = 0.04
    beta_window: int = 252
    rebalance: str = "daily"


@dataclass
class CostCfg:
    half_spread_bps: float = 1.0
    slippage_bps: float = 2.0
    impact_coef: float = 0.1

    @property
    def linear_bps(self) -> float:
        return self.half_spread_bps + self.slippage_bps


@dataclass
class ExecutionCfg:
    fractional: bool = True
    min_trade_weight: float = 0.005
    min_order_notional: float = 5.0
    limit_buffer_bps: float = 10.0


@dataclass
class RiskCfg:
    starting_cash: float = 100_000.0
    max_capital: float = 1_000.0
    kill_switch_drawdown: float = 0.25
    daily_loss_limit: float = 0.03
    max_order_notional: float = 50_000.0
    max_price_deviation: float = 0.10


@dataclass
class Config:
    general: GeneralCfg = field(default_factory=GeneralCfg)
    universe: UniverseCfg = field(default_factory=UniverseCfg)
    alphas: AlphaCfg = field(default_factory=AlphaCfg)
    risk_model: RiskModelCfg = field(default_factory=RiskModelCfg)
    portfolio: PortfolioCfg = field(default_factory=PortfolioCfg)
    costs: CostCfg = field(default_factory=CostCfg)
    execution: ExecutionCfg = field(default_factory=ExecutionCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)
    root: Path = Path(".")

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def validate(self) -> None:
        from .alphas import ALPHAS
        g, p, a = self.general, self.portfolio, self.alphas
        if g.mode not in ("paper", "live"):
            raise ValueError(f"general.mode must be 'paper' or 'live', got {g.mode!r}")
        if g.data_provider not in ("csv", "connector", "yahoo", "robinhood"):
            raise ValueError(f"unknown data_provider {g.data_provider!r}")
        if p.rebalance not in ("daily", "weekly"):
            raise ValueError("portfolio.rebalance must be 'daily' or 'weekly'")
        if g.mode == "live" and not p.long_only:
            raise ValueError("live trading on Robinhood requires portfolio.long_only = true")
        if not 0 < p.max_weight <= 1 or p.max_gross <= 0 or p.risk_aversion <= 0:
            raise ValueError("portfolio: need 0 < max_weight <= 1, max_gross > 0, risk_aversion > 0")
        bad = [k for k in a.enabled if k not in ALPHAS]
        if bad or not a.enabled:
            raise ValueError(f"alphas.enabled has unknown/empty entries {bad}; known: {sorted(ALPHAS)}")
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
    sections = {"general": GeneralCfg, "universe": UniverseCfg, "alphas": AlphaCfg, "risk_model": RiskModelCfg,
                "portfolio": PortfolioCfg, "costs": CostCfg, "execution": ExecutionCfg, "risk": RiskCfg}
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    cfg = Config(**{k: _build(cls, raw.get(k, {})) for k, cls in sections.items()}, root=path.resolve().parent)
    cfg.validate()
    return cfg
