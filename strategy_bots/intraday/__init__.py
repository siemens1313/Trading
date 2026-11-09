"""The intraday bot: five strategies for trading inside the day.

Execution (TWAP, VWAP, implementation shortfall) turns the daily bot's orders into child orders; scalping,
market making and smart order routing are intraday strategies of their own, run here as simulations.
"""

# name: (row in the strategy table, title, command that runs it)
INTRADAY = {
    "scalping": (2, "Scalping (bar order-flow imbalance, PDT-guarded)", "scalp"),
    "sor": (3, "Smart order routing (adaptive venue fill rates)", "route"),
    "market_making": (4, "Market making (Avellaneda-Stoikov, optional stat-arb skew)", "make-market"),
    "twap": (6, "Time-weighted average price execution", "execute --algo twap"),
    "vwap": (7, "Volume-weighted average price execution, plus implementation shortfall", "execute --algo vwap|is"),
}
