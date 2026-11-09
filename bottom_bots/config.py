"""Load the packaged thresholds, optionally overlaid with a user TOML file."""

from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path


class ConfigError(ValueError):
    pass


# Tables whose keys are user-defined (symbols), so an override replaces them wholesale.
_FREEFORM = {"instruments"}


def load(path: Path | None = None) -> dict:
    with resources.files("bottom_bots").joinpath("thresholds.toml").open("rb") as f:
        cfg = tomllib.load(f)
    if path is not None:
        with open(path, "rb") as f:
            _merge(cfg, tomllib.load(f), prefix="")
    validate(cfg)
    return cfg


def _merge(base: dict, override: dict, prefix: str) -> None:
    for key, value in override.items():
        where = prefix + key
        if key not in base:
            raise ConfigError(f"unknown setting '{where}'")
        if isinstance(base[key], dict) and key not in _FREEFORM:
            if not isinstance(value, dict):
                raise ConfigError(f"'{where}' must be a table")
            _merge(base[key], value, where + ".")
        else:
            base[key] = value


def validate(cfg: dict) -> None:
    tranches = cfg["plan"]["tranches"]
    if len(tranches) != 6:
        raise ConfigError("plan.tranches needs 6 values, one per score 0..5")
    if any(not 0.0 <= t <= 1.0 for t in tranches):
        raise ConfigError("plan.tranches values must be between 0 and 1")
    if any(b < a for a, b in zip(tranches, tranches[1:])):
        raise ConfigError("plan.tranches must not decrease as the score rises")
    weights = cfg["plan"]["instruments"]
    if weights and abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ConfigError("plan.instruments weights must sum to 1")
    if cfg["flows_sentiment"]["require"] not in ("all", "any"):
        raise ConfigError("flows_sentiment.require must be 'all' or 'any'")
    agg = cfg["aggregator"]
    if agg["persistence_runs"] < 1:
        raise ConfigError("aggregator.persistence_runs must be at least 1")
    if not 1 <= agg["oversold_min"] <= agg["capitulation_min"] <= 5:
        raise ConfigError("need 1 <= aggregator.oversold_min <= aggregator.capitulation_min <= 5")
