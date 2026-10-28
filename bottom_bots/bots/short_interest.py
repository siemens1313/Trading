"""Bot 4 — short interest: is there fuel, and has price stopped making new lows? (article §4)"""

from __future__ import annotations

from ..logic import all_of, any_of, status_of
from ..models import BotResult
from ..series import sessions_since_low, sma
from .base import SensorBot


class ShortInterestBot(SensorBot):
    key = "short_interest"
    item = 4
    title = "Short-interest bot"
    rule = "Short interest heavy (≥10% of float or ≥5 days to cover) while price has stopped making new lows"

    def evaluate(self) -> BotResult:
        c = self.cfg["short_interest"]
        symbol = c["symbol"].upper()
        si = self.fresh("si_pct_float")
        dtc = self.fresh("days_to_cover")
        fee = self.fresh("borrow_fee")
        closes = [p.value for p in self.fresh_history(f"close:{symbol}")][-c["low_lookback"]:]

        heavy = any_of(
            None if si is None else si.value >= c["pct_float_elevated"],
            None if dtc is None else dtc.value >= c["days_to_cover_long"],
        )
        since_low = sessions_since_low(closes) if len(closes) > c["no_new_low_sessions"] else None
        stopped = None if since_low is None else since_low >= c["no_new_low_sessions"]
        verdict = all_of(heavy, stopped)

        parts = []
        if heavy is None:
            parts.append("no short-interest reading")
        else:
            fuel = []
            if si is not None:
                fuel.append(f"{si.value:.1f}% of float")
            if dtc is not None:
                fuel.append(f"{dtc.value:.1f} days to cover")
            parts.append(f"short interest {'heavy' if heavy else 'light'} ({', '.join(fuel)})")
        if stopped is None:
            parts.append(f"not enough fresh {symbol} closes to judge new lows")
        elif stopped:
            parts.append(f"{symbol}'s {len(closes)}-session low is {since_low} sessions old: no new low")
        else:
            parts.append(f"{symbol} made a new {len(closes)}-session low {since_low} session(s) ago")
        summary = "; ".join(parts)
        summary = summary[0].upper() + summary[1:]

        flags: list[str] = []
        if si is not None and si.value >= c["pct_float_crowded"]:
            flags.append(f"Short interest {si.value:.1f}% of float ≥ {c['pct_float_crowded']:g}%: crowded")
        average = sma(closes, c["ignition_sma"])
        if average is not None and closes[-1] > average:
            flags.append(f"Ignition: {symbol} closed above its {c['ignition_sma']}-session average ({closes[-1]:.2f} > {average:.2f})")

        notes = [
            "Heavy shorts are only fuel; price starting to rise is the ignition",
            "Short interest is published twice a month and runs more than a week late",
            "Short Sale Volume ≠ Short Interest — daily short-sale volume is not a net short position",
        ]
        if si is not None and symbol in self.cfg["flows_sentiment"]["etfs"]:
            notes.append(f"The 10%/20%-of-float thresholds are written for stocks; ETF short interest ({symbol}) reads differently")
        readings = [
            self.reading("SI", si, ".1f", "% float"),
            self.reading("DTC", dtc, ".1f", "d"),
            self.reading("Borrow", fee, ".2f", "%"),
        ]
        return self.result(status_of(verdict), summary, readings, flags, notes)
