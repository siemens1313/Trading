# AUTOPILOT: one paper-trading cycle run by Claude Code

This runbook is for a Claude Code session started by a scheduled task. The cloud sandbox can't reach market-data
sites, so Claude fetches data through the **Robinhood connector**, the bot validates it, trades the **paper**
account, and the new state is committed back to git. Follow the steps in order. If any step fails, stop:
**no trade is better than a trade on bad data.**

## Hard rules
1. **Paper only.** Never call any Robinhood tool that places, reviews, cancels or changes orders, or exercises
   options (`place_*`, `review_*`, `cancel_*`, `exercise_*`). The only Robinhood tools allowed are
   `get_equity_historicals` and `get_equity_quotes`. If `config.toml` says `mode = "live"`, stop and report.
2. Never retype bar data. Bars come only from tool-result files saved by Claude Code.
3. Don't change strategy code, `config.toml` or tests during an autopilot run. Report problems instead.
4. Only commit `data/` and `state/`.

## Steps
0. `cd` into the repo. Install deps only if `python -c "import pandas, numpy"` fails: `pip install -r requirements.txt`.
   Check `python -m quantbot clock`. If it exits 1 (market closed), stop and report "market closed".
1. **Bars.** Call `get_equity_historicals` three times, with `interval: "day"`, `adjustment_type: "split"`,
   `start_time: "2019-01-01T00:00:00Z"`, `end_time: "<today>T00:00:00Z"` (today in UTC), and these symbols:
   - `["SPY","QQQ","IWM","EFA","EEM","TLT","IEF","GLD","DBC","VNQ"]`
   - `["XLK","XLF","XLE","XLV","XLY","XLP","XLI","XLB","XLU","XLRE"]`
   - `["XLC"]`

   Each result is too large to show and gets saved to a file; the tool result gives the path. Then run
   `python -m quantbot ingest-bars <path1> <path2> <path3>`. If a result came back inline instead of as a file, stop.
2. **Quotes.** Call `get_equity_quotes` twice: the first 11 symbols (SPY…XLK) and the last 10 (XLF…XLC).
   Write `/tmp/quotes.txt` with one line per symbol, copying values exactly from the results:
   `SYMBOL bid_price ask_price last_trade_price venue_last_trade_time close.price close.date adjusted_previous_close`.
   Shorten the time to seconds with a `Z` (e.g. `2026-11-06T18:35:37Z`). If `close` is missing, use
   `quote.previous_close` and `quote.previous_close_date`. The last field is `quote.adjusted_previous_close`:
   Robinhood's official close and its daily bar sometimes differ by a few cents, and the check accepts
   an exact match to either. Then run `python -m quantbot ingest-quotes /tmp/quotes.txt`.
   The bot cross-checks every line against the bar data. If it rejects the file, fetch quotes again and retry once.
   If it fails again, stop without trading. Don't run step 3 after a rejected quote file.
3. **Trade.** `python -m quantbot run`, which uses `data_provider = "connector"` and paper mode, then `python -m quantbot status`.
4. **Persist.** `git add data state && git commit -m "paper cycle <date>: <action>" && git push`
   (retry the push up to 4 times with backoff if the network fails).
5. **Report** in 3–6 lines: equity and day P&L, action (rebalance / holding / limit hit), number of fills,
   any rejected orders or data problems, and the commit hash.
