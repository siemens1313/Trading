import tempfile
import unittest
from pathlib import Path

from bottom_bots import config


def load_with(text: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "override.toml"
        path.write_text(text)
        return config.load(path)


class ConfigTest(unittest.TestCase):
    def test_defaults_are_valid(self):
        cfg = config.load()
        self.assertEqual(cfg["volatility"]["vix_stressed"], 30.0)
        self.assertEqual(cfg["aggregator"]["capitulation_min"], 4)

    def test_override_merges_one_key(self):
        cfg = load_with("[put_call]\nequity_capitulation = 0.9\n")
        self.assertEqual(cfg["put_call"]["equity_capitulation"], 0.9)
        self.assertEqual(cfg["put_call"]["equity_euphoria"], 0.5)

    def test_unknown_setting_is_rejected(self):
        with self.assertRaisesRegex(config.ConfigError, "put_call.equity_capitulaton"):
            load_with("[put_call]\nequity_capitulaton = 0.9\n")

    def test_decreasing_tranches_are_rejected(self):
        with self.assertRaises(config.ConfigError):
            load_with("[plan]\ntranches = [0.0, 0.0, 0.5, 0.25, 0.5, 0.75]\n")

    def test_instruments_are_replaced_wholesale(self):
        self.assertEqual(load_with("[plan]\ninstruments = { SPY = 1.0 }\n")["plan"]["instruments"], {"SPY": 1.0})

    def test_weights_must_sum_to_one(self):
        with self.assertRaises(config.ConfigError):
            load_with("[plan]\ninstruments = { SPY = 0.5 }\n")


if __name__ == "__main__":
    unittest.main()
