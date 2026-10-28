"""Bot 1 — VIX term structure: has the panic spread to the whole index? (article §1)"""

from __future__ import annotations

from ..logic import any_of, status_of
from ..models import BotResult, Reading
from .base import SensorBot

# Cboe study quoted in the article: average S&P 500 return over the following year
# after VIX closes above each level.
CBOE_FORWARD_RETURNS = ((40.0, 33), (35.0, 28), (30.0, 23))


class VolatilityBot(SensorBot):
    key = "volatility"
    item = 1
    title = "VIX term-structure bot"
    rule = "VIX > 30, or VIX/VIX3M > 1 (term structure inverted)"

    def evaluate(self) -> BotResult:
        c = self.cfg["volatility"]
        vix = self.fresh("vix")
        vix3m = self.fresh("vix3m")
        vxn = self.fresh("vxn")
        pair = self.aligned("vix", "vix3m")
        ratio = pair[1] / pair[2] if pair and pair[2] else None

        stressed = None if vix is None else vix.value > c["vix_stressed"]
        inverted = None if ratio is None else ratio > c["term_ratio_inverted"]
        verdict = any_of(stressed, inverted)

        if verdict:
            parts = []
            if stressed:
                parts.append(f"VIX {vix.value:.2f} > {c['vix_stressed']:g}")
            if inverted:
                parts.append(f"VIX/VIX3M {ratio:.2f} > {c['term_ratio_inverted']:g}: term structure inverted")
            summary = "; ".join(parts)
        elif verdict is False:
            summary = (
                f"VIX {vix.value:.2f} ≤ {c['vix_stressed']:g} and VIX/VIX3M {ratio:.2f} ≤ "
                f"{c['term_ratio_inverted']:g}: no inversion"
            )
        elif vix is None:
            summary = "No fresh VIX reading"
        else:
            summary = f"VIX {vix.value:.2f} ≤ {c['vix_stressed']:g}, but no same-day VIX3M to check for an inversion"

        flags: list[str] = []
        notes: list[str] = []
        if vix is not None:
            if vix.value > c["vix_extreme"]:
                flags.append(f"VIX above {c['vix_extreme']:g}: the article's extreme zone")
            for level, avg in CBOE_FORWARD_RETURNS:
                if vix.value > level:
                    flags.append(
                        f"Cboe: after VIX closes above {level:g}, the S&P 500 averaged about +{avg}% over the "
                        "next year — yet in 2008 it kept falling after the first close above 40, and in 2020 "
                        "VIX reached 80"
                    )
                    break
        if vix is not None and vxn is not None:
            spread = vxn.value - vix.value
            if spread > c["vxn_spread_alert"]:
                flags.append(f"VXN − VIX = {spread:.1f} pts: fear is concentrated in the Nasdaq-100, not the whole index")
                if vix.value < c["vix_stressed"]:
                    notes.append("A calm VIX does not by itself mean the market is safe (article §1)")
            self._check_divergence(c, notes)

        readings = [
            self.reading("VIX", vix),
            self.reading("VIX3M", vix3m),
            Reading("VIX/VIX3M", ratio, pair[0] if pair else None),
            self.reading("VXN", vxn),
        ]
        return self.result(status_of(verdict), summary, readings, flags, notes)

    def _check_divergence(self, c: dict, notes: list[str]) -> None:
        # "如果VXN开始下降，但 $SOXX 仍然持续创新低，不能直接当成底部确认"
        n, lookback, symbol = c["vxn_trend_sessions"], c["semis_low_lookback"], c["semis_symbol"]
        vxn = self.fresh_history("vxn")
        semis = self.fresh_history(f"close:{symbol}")
        if len(vxn) <= n or len(semis) <= lookback:
            notes.append(
                f"VXN/{symbol} divergence check skipped: needs {n + 1} sessions of VXN and "
                f"{lookback + 1} of {symbol} closes"
            )
            return
        vxn_falling = vxn[-1].value < vxn[-1 - n].value
        semis_new_low = semis[-1].value <= min(p.value for p in semis[-1 - lookback:-1])
        if vxn_falling and semis_new_low:
            self.warn(
                f"VXN is falling ({vxn[-1 - n].value:.2f} → {vxn[-1].value:.2f} over {n} sessions) while "
                f"{symbol} closed at a new {lookback}-session low — not a bottom confirmation (article §1)"
            )
