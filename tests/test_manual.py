import tempfile
import unittest
from datetime import date
from pathlib import Path

from bottom_bots import manual
from bottom_bots.store import SeriesStore


def load(text: str) -> SeriesStore:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "manual.toml"
        path.write_text(text)
        store = SeriesStore()
        manual.load_into(store, path)
        return store


class ManualInputTest(unittest.TestCase):
    def test_rows_load_with_their_source(self):
        store = load('[[aaii_bearish]]\ndate = 2026-10-22\nvalue = 45.8\nsource = "AAII"\n')
        point = store.latest("aaii_bearish")
        self.assertEqual((point.day, point.value, point.source), (date(2026, 10, 22), 45.8, "manual:AAII"))

    def test_prices_go_under_a_symbol(self):
        store = load("[[close.soxx]]\ndate = 2026-10-22\nvalue = 251.2\n")
        self.assertEqual(store.latest("close:SOXX").value, 251.2)

    def test_unknown_metric_is_rejected(self):
        with self.assertRaisesRegex(manual.ManualInputError, "unknown metric 'aaii'"):
            load("[[aaii]]\ndate = 2026-10-22\nvalue = 45.8\n")

    def test_short_sale_volume_is_rejected(self):
        with self.assertRaisesRegex(manual.ManualInputError, "Short Interest"):
            load("[[short_sale_volume]]\ndate = 2026-10-22\nvalue = 0.52\n")

    def test_value_must_be_numeric(self):
        with self.assertRaisesRegex(manual.ManualInputError, "numeric"):
            load('[[vix]]\ndate = 2026-10-22\nvalue = "19"\n')

    def test_date_must_be_a_toml_date(self):
        with self.assertRaisesRegex(manual.ManualInputError, "unquoted"):
            load('[[vix]]\ndate = "2026-10-22"\nvalue = 19.0\n')


class StorePriorityTest(unittest.TestCase):
    def test_manual_beats_fetched_for_the_same_day(self):
        store = SeriesStore()
        day = date(2026, 10, 23)
        store.add("vix", day, 19.0, "cboe")
        store.add("vix", day, 19.45, "manual")
        store.add("vix", day, 19.1, "cboe")
        self.assertEqual(store.latest("vix").value, 19.45)

    def test_cache_keeps_fetched_data_only(self):
        store = SeriesStore()
        store.add("vix", date(2026, 10, 22), 19.8, "cboe")
        store.add("aaii_bearish", date(2026, 10, 22), 45.8, "manual:AAII")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "series.json"
            store.save(path)
            reloaded = SeriesStore.load(path)
        self.assertEqual(reloaded.metrics(), ["vix"])


if __name__ == "__main__":
    unittest.main()
