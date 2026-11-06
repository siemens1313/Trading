# quantbot

A small, tested quantitative trading bot. **Paper trading first**, with an optional and locked-by-default Robinhood live adapter.

> Robinhood has no paper-trading account and no official stock-trading API. So paper mode here runs a
> **local simulated account** (`state/paper_account.json`) that fills at real market prices plus slippage.
> Live mode uses the unofficial [`robin_stocks`](https://github.com/jmfernandes/robin_stocks) library.
> Automated trading through it is not supported by Robinhood and may break or get your account restricted. Use small sizes.

## Strategy: dual momentum + trend + volatility targeting

Universe: 10 liquid ETFs (`SPY QQQ IWM EFA EEM TLT IEF GLD DBC VNQ`). Once a month:

1. **Momentum score** = average of 3, 6 and 12-month total return.
2. **Eligible** only if momentum > 0 *and* price > 200-day moving average.
3. Hold the **top 3** eligible ETFs, weighted by **inverse volatility**.
4. Scale the book so estimated portfolio volatility is ≤ **12%/yr**. No asset is above 40%. The rest stays in **cash**.
   When nothing qualifies, the bot is 100% cash.

All parameters are in `config.toml`.

## Backtest (real Robinhood daily data, 2020-01-02 → 2026-11-05)

| | Strategy | SPY buy & hold |
|---|---:|---:|
| CAGR | +9.1% | +12.6% |
| Volatility | 12.8% | 20.1% |
| Sharpe (rf=0) | 0.74 | 0.69 |
| Max drawdown | **−16.4%** | −34.1% |
| Calmar | 0.55 | 0.37 |

What this means: the strategy gives up return in strong bull markets and roughly halves the drawdown. 2020–2022: strategy 6.5% CAGR
vs SPY 5.6% (−16% vs −34% max DD). 2023–2026: strategy 11.0% vs SPY 18.6%. Across top_n ∈ {2,3,4} × target_vol ∈ {10,12,15%}
× monthly/weekly, Sharpe ranged 0.50–0.92 and max DD never exceeded −18.5%. Monthly beat weekly every time.

Caveats: the bundled CSVs are split-adjusted but **not dividend-adjusted** (understates both, bonds most). The backtest
models 5 bps slippage and no taxes. Past performance is not predictive; this is not financial advice.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                     # 17 tests: no-lookahead, accounting, risk, engine
python -m quantbot backtest             # writes reports/backtest_*.csv
python -m quantbot backtest --start 2023-01-01
python -m quantbot run --dry-run -v     # show signals + orders, change nothing
python -m quantbot run                  # one paper trading cycle
python -m quantbot status               # account, positions, P&L, signals
```

### Paper trading with live prices

`data_provider = "csv"` replays the bundled data (useful offline). For real paper trading, switch to live quotes:

```bash
pip install yfinance
python -m quantbot update-data --source yahoo
```
then set `data_provider = "yahoo"` in `config.toml` (or `"robinhood"`, see below), and schedule it:

```bash
# crontab -e   (10:00 ET on weekdays; the bot only rebalances on the first run of each month)
0 10 * * 1-5 cd /path/to/quantbot && python -m quantbot run >> state/cron.log 2>&1
```
or keep it running: `python -m quantbot loop --every 30`.

Run it for at least a few weeks to a couple of months. Then compare `state/paper_equity.csv` against the backtest before you think about live.

## Going live on Robinhood (only after paper trading)

1. `pip install robin_stocks pyotp`
2. Set credentials as environment variables (never in the config):
   `RH_USERNAME`, `RH_PASSWORD`, and `RH_TOTP_SECRET` (authenticator secret) or `RH_MFA_CODE`.
3. In `config.toml`: `mode = "live"`, `data_provider = "robinhood"`, and set `risk.max_capital` to a small amount (default **$1,000**).
4. Unlock: `export QUANTBOT_ALLOW_LIVE=yes`
5. `python -m quantbot run --dry-run -v`, then `python -m quantbot run`. You type `send` to confirm each batch of orders (`--yes` skips this for cron).

The live broker only trades symbols in the universe and never manages more than `max_capital`. Your other holdings are left alone.
Orders are market orders (fractional by quantity, good-for-day) during regular hours. **The live adapter has not been tested against a real account.**
Watch the first runs in the Robinhood app.

## Safety rails

- **Kill switch**: equity down 30% from peak → flatten everything and halt. `python -m quantbot resume` to clear it.
- Rejects orders whose quote is >10% from the last close (bad data) or above `max_order_notional`.
- Never shorts, never uses leverage, never spends cash it doesn't have.
- Signals use completed daily bars only (a test shocks future prices and checks that the past doesn't change).
- Idempotent: running every 15 minutes still rebalances once per period.

## Layout

```
config.toml            all parameters
data/*.csv             daily bars (from Robinhood market data)
quantbot/strategy.py   signal -> target weights
quantbot/execution.py  target weights -> orders (shared by backtest and live)
quantbot/broker.py     SimulatedBroker, PaperBroker, RobinhoodBroker
quantbot/risk.py       pre-trade checks, kill switch
quantbot/backtest.py   event-driven backtest + metrics
quantbot/engine.py     one paper/live cycle with persistent state
quantbot/cli.py        command line
tests/                 pytest suite
```

To try your own idea, write a function with the same signature as `compute_targets(closes, cfg) -> weights` and backtest it.
