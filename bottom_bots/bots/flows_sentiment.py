"""Bot 5 — flows and sentiment: is the money coming back? (article §5)"""

from __future__ import annotations

from datetime import timedelta

from ..logic import all_of, any_of, status_of
from ..models import BotResult
from .base import SensorBot


class FlowsSentimentBot(SensorBot):
    key = "flows_sentiment"
    item = 5
    title = "Money-back bot (flows & sentiment)"
    rule = "ETF flows turned from outflows to consecutive inflows, and AAII bearish topped 50% and is falling"

    def evaluate(self) -> BotResult:
        c = self.cfg["flows_sentiment"]
        etfs = "+".join(c["etfs"])
        flows_turned, flows_text = self._flows(c, etfs)
        aaii_turned, aaii_text = self._aaii(c)
        combine = all_of if c["require"] == "all" else any_of
        verdict = combine(flows_turned, aaii_turned)
        summary = f"{flows_text}; {aaii_text}"

        flags: list[str] = []
        extremes: list[str] = []
        fear_greed = self.fresh("fear_greed")
        naaim = self.fresh("naaim")
        breadth = self.fresh("breadth_200dma")
        aaii = self.fresh("aaii_bearish")
        if fear_greed is not None and fear_greed.value < c["fear_greed_extreme"]:
            rare = " — below 10 is rare" if fear_greed.value < c["fear_greed_rare"] else ""
            extremes.append(f"Fear & Greed {fear_greed.value:.0f} (< {c['fear_greed_extreme']:g}: extreme fear{rare})")
        if aaii is not None and aaii.value > c["aaii_extreme"]:
            extremes.append(f"AAII bearish {aaii.value:.1f}% (> {c['aaii_extreme']:g}%)")
        if breadth is not None:
            if breadth.value < c["breadth_washout"]:
                extremes.append(f"Only {breadth.value:.0f}% of S&P 500 above the 200-day: broad washout")
            else:
                flags.append(f"{breadth.value:.0f}% of S&P 500 members still above the 200-day: no broad washout yet")
        if naaim is not None and naaim.value >= c["naaim_high"]:
            flags.append(f"NAAIM {naaim.value:.1f}: professional managers are still heavily long, not capitulating")

        notes = ["The question is not one day's bounce but whether money keeps moving back for weeks (article §5)"]
        readings = [
            self.reading(f"{etfs} flow", self.fresh("etf_flows_weekly"), "+,.0f", " $mn/wk"),
            self.reading("AAII bear", aaii, ".1f", "%"),
            self.reading("F&G", fear_greed, ".0f"),
            self.reading("NAAIM", naaim, ".1f"),
            self.reading(">200DMA", breadth, ".0f", "%"),
        ]
        return self.result(status_of(verdict), summary, readings, flags, notes, extremes)

    def _flows(self, c: dict, etfs: str) -> tuple[bool | None, str]:
        k, m = c["inflow_weeks"], c["prior_outflow_weeks"]
        flows = [p.value for p in self.fresh_history("etf_flows_weekly")]
        if len(flows) < k + 1:
            return None, f"{etfs} flows: need {k + 1}+ weekly readings, have {len(flows)}"
        recent, prior = flows[-k:], flows[-(k + m):-k]
        if not all(v > 0 for v in recent):
            return False, f"{etfs} flows: not {k} straight weeks of inflows"
        if sum(prior) >= 0:
            return False, f"{etfs} flows: {k} weeks of inflows, but no outflow phase before them to turn from"
        return True, f"{etfs} flows turned: {k} straight weeks of inflows after net outflows"

    def _aaii(self, c: dict) -> tuple[bool | None, str]:
        latest = self.fresh("aaii_bearish")
        if latest is None:
            return None, "no fresh AAII reading"
        start = self.as_of - timedelta(weeks=c["aaii_peak_lookback_weeks"])
        window = [p.value for p in self.store.series("aaii_bearish", until=self.as_of) if p.day >= start]
        peak = max(window)
        if peak <= c["aaii_extreme"]:
            return False, f"AAII bearish {latest.value:.1f}% (peak {peak:.1f}%) never topped {c['aaii_extreme']:g}%"
        if len(window) < 2:
            return None, f"AAII bearish {latest.value:.1f}%: need a prior week to see it falling"
        if window[-1] < window[-2] and window[-1] < peak:
            return True, f"AAII bearish peaked at {peak:.1f}% and is falling ({window[-2]:.1f}% → {window[-1]:.1f}%)"
        return False, f"AAII bearish {latest.value:.1f}% topped {c['aaii_extreme']:g}% but isn't falling yet"
