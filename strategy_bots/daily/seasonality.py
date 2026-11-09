"""Strategy 8: seasonality.

Holds `symbols` over the turn of the month (the last `days_before` and first `days_after` trading days),
skipping calendar months that have been weak in past years (z-score of that month's average return
against all months below weak_z, once there are min_years of history). Calendar positions come from the
weekday calendar, so a decision never peeks at tomorrow's date in the data. Calendar spreads in futures
and options, the table's other example, need instruments this stock bot doesn't trade.
"""

from __future__ import annotations

import numpy as np

from .base import DailyStrategy


class Seasonality(DailyStrategy):
    name, number, title = "seasonality", 8, "Seasonality (turn of the month + weak-month filter)"
    defaults = {"symbols": ["SPY"], "days_before": 1, "days_after": 3, "month_filter": True, "min_years": 3, "weak_z": -1.0}

    def weights(self, panel):
        p = self.p
        names = [s for s in p["symbols"] if s in panel.symbols] or list(panel.symbols)
        cols = [panel.col(s) for s in names]
        T, N = panel.close.shape
        W = np.zeros((T, N))
        c = panel.close[:, cols]
        daily = np.r_[0.0, np.log(c[1:] / c[:-1]).mean(axis=1)]
        months = panel.dates.astype("datetime64[M]")
        monthly: dict[np.datetime64, float] = {}
        for t in range(T):
            monthly[months[t]] = monthly.get(months[t], 0.0) + daily[t]
            hold_day = np.busday_offset(panel.dates[t], 1, roll="forward")
            hold_month = hold_day.astype("datetime64[M]")
            from_start = np.busday_count(hold_month.astype("datetime64[D]"), hold_day)
            to_end = np.busday_count(hold_day, (hold_month + 1).astype("datetime64[D]")) - 1
            if from_start >= p["days_after"] and to_end >= p["days_before"]:
                continue
            if p["month_filter"] and self._weak(monthly, hold_month):
                continue
            W[t, cols] = 1 / len(cols)
        return W

    def _weak(self, monthly: dict, month: np.datetime64) -> bool:
        """Has this calendar month been reliably weak in completed past years?"""
        past = {m: r for m, r in monthly.items() if m < month}
        same = [r for m, r in past.items() if (m.astype(int) % 12) == (month.astype(int) % 12)]
        if len(same) < self.p["min_years"] or len(past) < 24:
            return False
        everything = np.array(list(past.values()))
        se = everything.std(ddof=1) / np.sqrt(len(same))
        return se > 0 and (np.mean(same) - everything.mean()) / se < self.p["weak_z"]
