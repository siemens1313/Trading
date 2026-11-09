"""The daily bot: ten strategies that decide what to hold from daily bars."""

from .adaptive import Adaptive
from .base import DailyStrategy
from .genetic import GeneticAlgorithm
from .ml import MachineLearning
from .momentum import Momentum
from .pairs import PairsTrading
from .patterns import PatternRecognition
from .rl import DeepRL
from .seasonality import Seasonality
from .sentiment import Sentiment
from .volatility import VolatilityBreakout

STRATEGIES: dict[str, type[DailyStrategy]] = {
    cls.name: cls
    for cls in sorted(
        (PairsTrading, Momentum, Seasonality, VolatilityBreakout, MachineLearning, PatternRecognition, Sentiment, DeepRL, Adaptive, GeneticAlgorithm),
        key=lambda c: c.number,
    )
}

__all__ = ["DailyStrategy", "STRATEGIES"]
