"""The six bots: five sensors, one per checklist item, and the judge that scores them."""

from __future__ import annotations

from datetime import date

from ..models import BotResult
from ..store import SeriesStore
from .aggregator import AggregatorBot
from .flows_sentiment import FlowsSentimentBot
from .gamma import GammaBot
from .put_call import PutCallBot
from .short_interest import ShortInterestBot
from .volatility import VolatilityBot

SENSOR_BOTS = (VolatilityBot, PutCallBot, GammaBot, ShortInterestBot, FlowsSentimentBot)

__all__ = ["AggregatorBot", "SENSOR_BOTS", "run_sensors"]


def run_sensors(cfg: dict, store: SeriesStore, as_of: date, only: set[str] | None = None) -> list[BotResult]:
    return [bot(cfg, store, as_of).evaluate() for bot in SENSOR_BOTS if only is None or bot.key in only]
