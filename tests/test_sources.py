"""Parser tests against fixture payloads shaped like each public endpoint's response.

The fixtures follow the published formats; they could not be checked against the live
hosts from the build sandbox, so a format change shows up at run time as a data warning.
"""

import json
import unittest
from datetime import date, datetime, timezone

from bottom_bots import config, sources
from bottom_bots.sources import cboe, cnn, prices, squeeze
from bottom_bots.sources.http import FetchError
from bottom_bots.store import SeriesStore

CBOE_CSV = "DATE,OPEN,HIGH,LOW,CLOSE\n10/22/2026,20.10,21.00,19.20,19.80\n10/23/2026,19.90,20.40,19.00,19.45\n"

PUT_CALL = {
    "ratios": [
        {"name": "TOTAL PUT/CALL RATIO", "value": "0.93"},
        {"name": "INDEX PUT/CALL RATIO", "value": "1.10"},
        {"name": "EQUITY PUT/CALL RATIO", "value": "0.67"},
        {"name": "CBOE VOLATILITY INDEX (VIX) PUT/CALL RATIO", "value": "0.41"},
        {"name": "SPX + SPXW PUT/CALL RATIO", "value": "1.31"},
    ]
}


def _ms(*args) -> int:
    return int(datetime(*args, tzinfo=timezone.utc).timestamp() * 1000)


class CboeTest(unittest.TestCase):
    def test_index_history(self):
        self.assertEqual(cboe.parse_index_history(CBOE_CSV, "VIX")[-1], (date(2026, 10, 23), 19.45))

    def test_two_column_history_uses_the_symbol_column(self):
        rows = cboe.parse_index_history("DATE,VIX3M\n2026-10-23,20.48\n", "VIX3M")
        self.assertEqual(rows, [(date(2026, 10, 23), 20.48)])

    def test_history_without_a_header_is_an_error(self):
        with self.assertRaises(ValueError):
            cboe.parse_index_history("<html>blocked</html>", "VIX")

    def test_put_call_ratios_ignore_product_ratios(self):
        self.assertEqual(cboe.parse_put_call(PUT_CALL), {"pc_total": 0.93, "pc_index": 1.10, "pc_equity": 0.67})

    def test_put_call_ratios_are_found_when_nested(self):
        self.assertEqual(cboe.parse_put_call({"data": [{"stats": PUT_CALL}]})["pc_equity"], 0.67)

    def test_put_call_walks_back_over_a_failed_day_and_the_weekend(self):
        requested = []

        def fetch(url):
            requested.append(url)
            if "2026-10-23" in url:
                return json.dumps(PUT_CALL).encode()
            raise FetchError("HTTP 403", 403)  # Cboe answers 403 for days it hasn't published

        points = cboe.fetch_put_call(date(2026, 10, 26), fetch)
        self.assertEqual({p[1] for p in points}, {date(2026, 10, 23)})
        self.assertEqual(len(requested), 2)  # Monday failed, weekend skipped, Friday found


    def test_put_call_stops_at_once_when_the_host_is_unreachable(self):
        requested = []

        def fetch(url):
            requested.append(url)
            raise FetchError("connection refused")  # no HTTP status: network-level failure

        with self.assertRaises(FetchError):
            cboe.fetch_put_call(date(2026, 10, 23), fetch)
        self.assertEqual(len(requested), 1)


class OtherSourcesTest(unittest.TestCase):
    def test_squeezemetrics_units(self):
        text = "date,price,dix,gex\n2026-10-22,6612.5,0.431,-1234567890.5\n"
        got = {m: v for m, _, v, _ in squeeze.parse_dix_csv(text)}
        self.assertAlmostEqual(got["dix"], 43.1)
        self.assertAlmostEqual(got["gex"], -1.2345678905)
        self.assertEqual(got["close:SPX"], 6612.5)

    def test_yahoo_chart_uses_the_exchange_date_and_skips_gaps(self):
        stamp = int(datetime(2026, 10, 22, 13, 30, tzinfo=timezone.utc).timestamp())
        payload = {
            "chart": {
                "result": [
                    {
                        "meta": {"gmtoffset": -14400},
                        "timestamp": [stamp, stamp + 86400],
                        "indicators": {"quote": [{"close": [250.5, None]}]},
                    }
                ]
            }
        }
        self.assertEqual(prices.parse_yahoo_chart(payload), [(date(2026, 10, 22), 250.5)])

    def test_stooq_csv(self):
        text = "Date,Open,High,Low,Close,Volume\n2026-10-22,250,255,248,251.2,1000\n"
        self.assertEqual(prices.parse_stooq_csv(text), [(date(2026, 10, 22), 251.2)])
        with self.assertRaises(ValueError):
            prices.parse_stooq_csv("No data")

    def test_stooq_html_page_is_reported_plainly(self):
        with self.assertRaisesRegex(ValueError, "HTML page"):
            prices.parse_stooq_csv('<!DOCTYPE html><html><head><meta charset="utf-8">')

    def test_cboe_chart_history(self):
        payload = {"data": [{"date": "2026-10-26", "close": "491.83"}, {"date": "2026-10-27", "close": 511.17}, {"x": 1}]}
        self.assertEqual(
            prices.parse_cboe_history(payload), [(date(2026, 10, 26), 491.83), (date(2026, 10, 27), 511.17)]
        )
        with self.assertRaises(ValueError):
            prices.parse_cboe_history({"error": "not found"})

    def test_prices_try_cboe_first(self):
        requested = []

        def fetch(url):
            requested.append(url)
            return json.dumps({"data": [{"date": "2026-10-27", "close": 7723.0}]}).encode()

        self.assertEqual(prices.fetch_closes("SPX", fetch), [("close:SPX", date(2026, 10, 27), 7723.0, "cboe")])
        self.assertTrue(requested[0].endswith("/charts/historical/_SPX.json"))

    def test_prices_fall_back_to_stooq(self):
        def fetch(url):
            if "cboe" in url or "yahoo" in url:
                raise FetchError("HTTP 429", 429)
            return b"Date,Open,High,Low,Close,Volume\n2026-10-22,250,255,248,251.2,1000\n"

        self.assertEqual(prices.fetch_closes("SOXX", fetch), [("close:SOXX", date(2026, 10, 22), 251.2, "stooq")])

    def test_fear_and_greed(self):
        payload = {
            "fear_and_greed": {"score": 21.4, "timestamp": "2026-10-23T20:00:00+00:00"},
            "fear_and_greed_historical": {"data": [{"x": _ms(2026, 10, 22, 23, 59), "y": 24.0}]},
        }
        self.assertEqual(cnn.parse_fear_greed(payload), [(date(2026, 10, 22), 24.0), (date(2026, 10, 23), 21.4)])


class RefreshTest(unittest.TestCase):
    def test_every_failure_becomes_a_warning(self):
        def blocked(url):
            raise FetchError("HTTP 403")

        store = SeriesStore()
        warnings = sources.refresh(store, config.load(), date(2026, 10, 23), blocked)
        self.assertEqual(len(warnings), 8)  # VIX, VIX3M, VXN, put/call, SqueezeMetrics, CNN, SOXX, SPX
        self.assertEqual(store.metrics(), [])

    def test_partial_success_keeps_what_arrived(self):
        def fetch(url):
            if url.endswith("/VIX_History.csv"):
                return CBOE_CSV.encode()
            raise FetchError("HTTP 403")

        store = SeriesStore()
        sources.refresh(store, config.load(), date(2026, 10, 23), fetch)
        self.assertEqual(store.latest("vix").value, 19.45)


if __name__ == "__main__":
    unittest.main()
