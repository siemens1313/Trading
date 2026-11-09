"""Bot 3 — dealer gamma: will market makers amplify the decline? (article §3)"""

from __future__ import annotations

from ..logic import all_of, status_of
from ..models import BotResult
from .base import SensorBot


class GammaBot(SensorBot):
    key = "gamma"
    item = 3
    title = "Dealer-gamma bot"
    rule = "Net gamma flipped from negative to positive, and the index is back above the flip level"

    def evaluate(self) -> BotResult:
        c = self.cfg["gamma"]
        underlying = c["underlying"].upper()
        gex = self.fresh("gex")
        dix = self.fresh("dix")
        flip = self.fresh("gamma_flip")
        price = self.fresh(f"close:{underlying}")

        flags: list[str] = []
        notes = [
            "Gamma answers one question — will the decline accelerate — not where the bottom is",
            "Public open interest can't prove dealers' real positioning; every GEX figure is a model estimate",
        ]
        if dix is not None and dix.value >= c["dix_strong"]:
            flags.append(f"DIX {dix.value:.1f}% ≥ {c['dix_strong']:g}%: dark-pool buying strong (SqueezeMetrics' own threshold)")
        readings = [
            self.reading("GEX", gex, "+.2f", " $bn/1%"),
            self.reading("Flip", flip, ".0f"),
            self.reading(underlying, price, ".0f"),
            self.reading("DIX", dix, ".1f", "%"),
        ]
        if gex is None:
            return self.result(status_of(None), "No fresh net-gamma (GEX) reading", readings, flags, notes)

        prior = [p for p in self.store.series("gex", until=self.as_of) if p.day < gex.day]
        prior = prior[-c["negative_lookback"]:]
        positive = gex.value > 0
        was_negative = any(p.value < 0 for p in prior) if prior else None
        if flip is not None and price is not None:
            above_flip = price.value >= flip.value
        else:
            # Without a published flip level, positive net gamma already means the index sits above it.
            above_flip = positive
            notes.append("No flip level entered: 'above the flip' is taken from the sign of net gamma")
        verdict = all_of(positive, was_negative, above_flip)

        if not positive:
            summary = (
                f"Net gamma {gex.value:+.2f} $bn/1% is negative: dealers sell into declines and buy into "
                "rallies, so moves get amplified"
            )
        elif was_negative is None:
            summary = "Net gamma is positive, but there are no earlier readings to confirm a flip from negative"
        elif not was_negative:
            summary = f"Net gamma positive through the last {len(prior)} readings: no flip from negative"
        elif not above_flip:
            summary = f"Gamma flipped positive but {underlying} {price.value:.0f} is still below the flip level {flip.value:.0f}"
        else:
            summary = "Net gamma flipped from negative to positive"
            if flip is not None and price is not None:
                summary += f"; {underlying} {price.value:.0f} ≥ flip {flip.value:.0f}"
        return self.result(status_of(verdict), summary, readings, flags, notes)
