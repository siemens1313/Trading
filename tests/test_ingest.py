import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from quantbot.ingest import IngestError, ingest_bars, ingest_quotes


def _bars_file(tmp_path, symbols, n=320, interval="day"):
    days = pd.bdate_range("2025-01-01", periods=n)
    res = [{"symbol": s, "interval": interval, "bars": [
        {"begins_at": f"{d:%Y-%m-%d}T00:00:00Z", "open_price": "10.0", "high_price": "11.0", "low_price": "9.5",
         "close_price": f"{10 + i * 0.01:.6f}", "volume": 1000, "session": "reg"} for i, d in enumerate(days)]}
        for s in symbols]
    p = tmp_path / "bars.json"
    p.write_text(json.dumps({"data": {"results": res}, "guide": "x"}))
    return p, days[-1], 10 + (n - 1) * 0.01


def test_ingest_bars_roundtrip_and_checks(tmp_path):
    p, last, close = _bars_file(tmp_path, ["AAA", "BBB"])
    out = ingest_bars([p], tmp_path / "data", ["AAA", "BBB"])
    assert out["AAA"] == (str(last.date()), 320)
    df = pd.read_csv(tmp_path / "data" / "AAA.csv", index_col="date")
    assert df["close"].iloc[-1] == pytest.approx(close)
    with pytest.raises(IngestError, match="missing"):
        ingest_bars([p], tmp_path / "data", ["AAA", "CCC"])
    p2, *_ = _bars_file(tmp_path, ["AAA"], interval="5minute")
    with pytest.raises(IngestError, match="interval"):
        ingest_bars([p2], tmp_path / "data", ["AAA"])


def test_ingest_quotes_validates_transcription(tmp_path):
    p, last, close = _bars_file(tmp_path, ["AAA"])
    ingest_bars([p], tmp_path / "data", ["AAA"])
    now = datetime(2030, 1, 2, 15, 0, tzinfo=timezone.utc)
    good = f"AAA {close*1.01:.4f} {close*1.0102:.4f} {close*1.0101:.4f} 2030-01-02T14:55:00Z {close:.6f} {last:%Y-%m-%d}\n"
    q = tmp_path / "q.txt"
    q.write_text(good)
    doc = ingest_quotes(q, tmp_path / "data", ["AAA"], now=now)
    assert doc["quotes"]["AAA"]["bid"] < doc["quotes"]["AAA"]["ask"]
    for bad, why in [
        (good.replace(f"{close:.6f}", f"{close*1.02:.6f}"), "transcription"),        # typo in prev close
        (good.replace("14:55:00Z", "13:00:00Z"), "min old"),                          # stale
        (f"AAA {close*1.02:.4f} {close*1.01:.4f} {close:.4f} 2030-01-02T14:55:00Z {close:.6f} {last:%Y-%m-%d}\n", "bad book"),
        ("", "missing quotes"),
    ]:
        q.write_text(bad)
        with pytest.raises(IngestError, match=why):
            ingest_quotes(q, tmp_path / "data", ["AAA"], now=now)


def test_quote_passes_when_either_close_field_matches_the_bar(tmp_path):
    """Robinhood's official close can differ from its bar close by cents (QQQ, 2026-11-10: 674.82 vs 674.76)."""
    p, last, close = _bars_file(tmp_path, ["AAA"])
    ingest_bars([p], tmp_path / "data", ["AAA"])
    now = datetime(2030, 1, 2, 15, 0, tzinfo=timezone.utc)
    book = f"AAA {close*1.01:.4f} {close*1.0102:.4f} {close*1.0101:.4f} 2030-01-02T14:55:00Z"
    official = close + 0.06                        # official close disagrees with the bar
    q = tmp_path / "q.txt"
    q.write_text(f"{book} {official:.6f} {last:%Y-%m-%d} {close:.6f}\n")   # adjusted close matches the bar
    assert ingest_quotes(q, tmp_path / "data", ["AAA"], now=now)["quotes"]["AAA"]["adj_prev_close"] == pytest.approx(close)
    q.write_text(f"{book} {official:.6f} {last:%Y-%m-%d}\n")                # 7 fields: no second chance
    with pytest.raises(IngestError, match="transcription"):
        ingest_quotes(q, tmp_path / "data", ["AAA"], now=now)
    q.write_text(f"{book} {official:.6f} {last:%Y-%m-%d} {close + 0.01:.6f}\n")  # both wrong -> rejected
    with pytest.raises(IngestError, match="transcription"):
        ingest_quotes(q, tmp_path / "data", ["AAA"], now=now)
