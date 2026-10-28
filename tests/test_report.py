import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from bottom_bots.cli import main
from bottom_bots.report import markdown_to_html, markdown_to_text

EXAMPLE = Path(__file__).resolve().parent.parent / "inputs" / "manual.example.toml"


def example_report() -> str:
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(io.StringIO()):
        main(["run", "--offline", "--manual", str(EXAMPLE), "--as-of", "2026-10-23", "--no-save"])
    return out.getvalue()


class EmailRenderTest(unittest.TestCase):
    def test_html_keeps_the_table_and_drops_the_markdown(self):
        page = markdown_to_html(example_report())
        self.assertIn(">Bottom checklist — 2026-10-23</h2>", page)
        self.assertEqual(page.count("<tr>"), 6)  # header + five bots
        self.assertIn("<b>NOT MET</b>", page)
        self.assertIn("<i>A calm VIX does not by itself mean the market is safe (article §1)</i>", page)
        self.assertNotIn("**", page)

    def test_html_escapes_and_leaves_url_underscores_alone(self):
        page = markdown_to_html(
            "- Cboe VIX: unavailable (https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv)\n"
            "Why: Equity P/C 0.59 < 0.80\n"
        )
        self.assertNotIn("<i>", page)
        self.assertIn("us_indices/daily_prices/VIX_History.csv", page)
        self.assertIn("0.59 &lt; 0.80", page)

    def test_nested_bullets_are_indented(self):
        page = markdown_to_html("- New tranche: **+25%**\n  - proposed: BUY QQQ $1,250.00\n")
        self.assertIn('<li style="margin-left:18px">proposed: BUY QQQ $1,250.00</li>', page)

    def test_text_has_no_markdown(self):
        text = markdown_to_text(example_report())
        self.assertTrue(text.startswith("Bottom checklist — 2026-10-23\n"))
        self.assertIn("1 · VIX term-structure bot · NOT MET · VIX 19.45", text)
        self.assertNotIn("**", text)
        self.assertNotIn("|", text)


if __name__ == "__main__":
    unittest.main()
