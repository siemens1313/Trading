"""Two bots for the fifteen strategies in the strategy table.

The daily bot decides what to hold from daily bars (strategies 1, 5, 8, 9, 10, 11, 12, 13, 14, 15).
The intraday bot decides how to trade inside the day (strategies 2, 3, 4, 6, 7).
Both run on paper: nothing here sends an order to a broker.
"""

__version__ = "0.1.0"
