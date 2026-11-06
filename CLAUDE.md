# quantbot

Desk-style systematic ETF trading bot (see README.md). Paper trading by default.

- Scheduled autopilot runs: follow **AUTOPILOT.md** exactly. Paper only: never use Robinhood order tools.
- Tests: `python -m pytest -q`. Backtest: `python -m quantbot backtest --report`. Research: `python -m quantbot research`.
- `state/` holds the paper account, engine state, equity log and decision journal. It's committed so runs persist.
