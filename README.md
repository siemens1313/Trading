# quantbot v2: a desk-style systematic trading bot

Paper trading first, with an optional and locked-by-default Robinhood live adapter.

## What "trade like Jane Street" can and can't mean here

Jane Street's core edge is **market making**: quoting both sides with low-latency infrastructure,
pricing ETFs and options better than everyone else, and buying retail order flow, including from Robinhood.
A Robinhood bot can't do any of that. It has no direct market access, no co-location and no exchange rebates,
and its orders are filled by those same market makers.

What we can copy is how a professional quant desk works:

| Desk discipline | Implemented as |
|---|---|
| Many small, independent edges | `alphas.py`: time-series momentum, cross-sectional momentum, short-term residual reversal (stat arb), low-vol |
| Trust only measured edge | Each alpha's weight comes from its *realized* information coefficient (IC), lagged until the forward return is actually known. A signal with negative IC gets zero weight, and the bot never flips it to make it look good |
| Haircut your backtest | Forecast size = Grinold `IC × σ × score`, with the IC cut in half (`ic_haircut`) |
| Separate alpha from beta | The alpha is relative value (demeaned). Long-only books add a CAPM prior: beta × 4%/yr |
| Don't let noise drive sizing | Ledoit-Wolf-shrunk correlations × EWMA vols (`riskmodel.py`) |
| Trade only when edge beats cost | Cost-aware mean-variance optimizer (`optimizer.py`, FISTA, no dependencies). An L1 cost term creates a no-trade band, and cost is amortized over the signal's *measured* persistence |
| Live within a risk budget | Risk aversion is calibrated to a 12% volatility target, with gross and per-name caps |
| Realistic fills | Half-spread + slippage + square-root market impact (`costs.py`). Paper fills cross the actual bid/ask. Live sends marketable **limit** orders and cancels its own stale ones |
| Hard risk limits | Kill switch (−25% from peak), 3% daily loss limit, bad-quote filter, per-order notional cap, `max_capital` cap for live |
| Post-trade review | `state/journal.jsonl` logs every decision: forecasts, alpha weights, ICs, targets, orders, fills |
| Be honest about luck | Block-bootstrap Sharpe confidence interval, probabilistic and **deflated** Sharpe, in-sample/out-of-sample protocol, and a check of whether the alpha overlay adds value net of costs |

## Results (real Robinhood daily data, 21 ETFs, 2019-08 → 2026-11)

| | Strategy | SPY buy & hold |
|---|---:|---:|
| CAGR | +8.2% | +14.0% |
| Volatility | 12.0% | 19.7% |
| Sharpe (95% CI) | 0.72 (0.01 – 1.54) | 0.77 (0.08 – 1.56) |
| Max drawdown | −23.4% | −34.1% |
| Beta to SPY | 0.51 | 1.00 |
| Turnover / cost drag | 24x/yr, 0.74%/yr | — |

- **Out of sample:** settings were chosen on 2019–2022 only (12 variants). The chosen setup made Sharpe 0.33 in-sample and **1.09 out-of-sample (2023–2026)**. Every variant landed at 0.98–1.10 out-of-sample, so the result isn't sensitive to settings.
- **Deflated Sharpe:** 89% probability the edge is real after counting about 15 tried configurations.
- **The uncomfortable part, which `python -m quantbot research` prints for you:** the risk-managed beta book *alone* (alpha forecasts zeroed) also makes Sharpe 0.72 at 7.8x turnover and −19% max drawdown. The alpha overlay adds **+0.04%/yr, t-stat 0.03: no proven net value**.
  Time-series momentum does have real predictive power (5-day IC +0.05, t = 2.6). But across 21 highly correlated ETFs, long-only and after costs, it doesn't turn into profit.
  A desk wouldn't trade an overlay that can't prove itself, so:
  - paper-trade it to gather live evidence;
  - for live money, set `alphas.ic_haircut = 0` (beta book only) unless `research` shows the overlay's t-stat above about 2.
- **Dividends are missing:** the bundled data is split-adjusted, not dividend-adjusted, which understates bonds and REITs. Taxes are ignored. Not financial advice.

## Commands

```bash
pip install -r requirements.txt
python -m pytest -q                           # 30 tests (incl. no-lookahead, optimizer vs scipy, planted-alpha combiner test)
python -m quantbot research                   # per-alpha IC/decay/persistence, deflated Sharpe, overlay attribution
python -m quantbot backtest --report          # metrics + reports/tearsheet.html
python -m quantbot backtest --start 2023-01-01
python -m quantbot run --dry-run -v           # full forecast table and orders, nothing sent
python -m quantbot run                        # one paper cycle (rebalances at most once per day)
python -m quantbot status                     # P&L, exposures, beta, alpha ICs/weights, per-symbol book
```

### Paper trading with live data

```bash
pip install yfinance && python -m quantbot update-data --source yahoo
```
Set `data_provider = "yahoo"`, or `"robinhood"` for real bid/ask, then run once each weekday morning:
```bash
# crontab -e   (09:45 ET Mon-Fri)
45 9 * * 1-5 cd /path/to/quantbot && python -m quantbot run >> state/cron.log 2>&1
```
Compare `state/paper_equity.csv` and `state/journal.jsonl` with the backtest for several weeks before thinking about live.

### Long/short (paper only)

Set `portfolio.long_only = false` to paper-trade a market-neutral stat-arb book: no beta prior, shorts allowed.
Robinhood's API doesn't support shorting through this bot, so live mode refuses long/short configs.

## Going live on Robinhood (after paper trading)

1. `pip install robin_stocks pyotp`
2. Set env vars `RH_USERNAME`, `RH_PASSWORD`, and `RH_TOTP_SECRET` (or `RH_MFA_CODE`). Never put them in the config.
3. In `config.toml`: `mode = "live"`, `data_provider = "robinhood"`, and a small `risk.max_capital` (default $1,000).
4. `export QUANTBOT_ALLOW_LIVE=yes`, then `python -m quantbot run --dry-run -v`, then `python -m quantbot run`. You type `send` to confirm.

The live broker only trades universe symbols, never manages more than `max_capital`, sends marketable limit orders
(good for day), and only cancels orders it placed itself. It uses the unofficial `robin_stocks` library. Robinhood doesn't
support automated trading through it, and it may break or get an account restricted. **The live adapter has not been tested
against a real account.** Watch the first runs in the Robinhood app.

## Adding your own alpha

Write `def myalpha(C: DataFrame, bench: str, **params) -> DataFrame` (scores per date and symbol, using data up to each date only),
register it in `ALPHAS` in `quantbot/alphas.py`, add it to `alphas.enabled`, and run `research`. The combiner will weight it by its
realized IC automatically. The tests show the pattern: a planted predictive alpha gets weighted and a planted anti-predictive one gets zero.

## Layout

```
config.toml               all parameters, commented
data/*.csv                21 ETFs, daily bars from Robinhood market data
quantbot/alphas.py        signals, realized-IC measurement, evidence-weighted combination
quantbot/riskmodel.py     EWMA vol x Ledoit-Wolf correlation
quantbot/optimizer.py     cost-aware mean-variance (FISTA), risk-budget calibration
quantbot/portfolio.py     forecast + prior + risk model + optimizer -> target weights
quantbot/costs.py         spread + slippage + sqrt impact
quantbot/execution.py     weights -> orders (shared by backtest and live)
quantbot/broker.py        SimulatedBroker, PaperBroker, RobinhoodBroker (limit orders)
quantbot/risk.py          pre-trade checks, kill switch
quantbot/backtest.py      event-driven backtest
quantbot/stats.py         bootstrap CI, probabilistic/deflated Sharpe
quantbot/research.py      alpha diagnostics, overlay attribution
quantbot/report.py        HTML tearsheet
quantbot/engine.py        one paper/live cycle, state, journal
quantbot/cli.py           command line
```
