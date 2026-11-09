# Trading

## Bottom-confirmation bots

Six bots that turn the five-signal "has the selloff really bottomed?" checklist into a
daily, scored report:

> VIX是否冲上30以上，或者VIX/VIX3M大于1；Equity Put/Call是否升到0.8以上；Gamma是否从负转正，指数是否收复翻转位置；空头仓位是否高企，同时价格停止创新低；ETF资金是否从流出转为连续流入，AAII看空是否超过50%后开始回落。
>
> 满足一两条，只能算超跌；四五条同时出现，才接近历史上那种投降式底部。

### Why six

The checklist has exactly five items and its verdict is a count out of five, so there is
one **sensor bot per item** (the score stays out of 5 and each bot fails independently
when its data source does) plus one **judge** that counts them and runs the staged plan.
Splitting further — say AAII and ETF flows into separate bots — would turn the article's
/5 score into a /6 one.

| # | Bot | MET when | Data |
|---|-----|----------|------|
| 1 | VIX term-structure | VIX > 30, **or** VIX/VIX3M > 1 | Cboe (auto) |
| 2 | Put/Call | Cboe **Equity** put/call ≥ 0.80 | Cboe (auto) |
| 3 | Dealer gamma | net gamma flipped negative → positive within 20 readings, **and** index ≥ the flip level | SqueezeMetrics GEX (auto); flip level manual |
| 4 | Short interest | (SI ≥ 10% of float **or** days-to-cover ≥ 5) **and** no new 60-session low in the last 5 sessions | SI manual; closes auto |
| 5 | Money back (flows & sentiment) | SOXX+SMH: 3 straight weeks of inflows after 4 net-outflow weeks, **and** AAII bearish topped 50% within 8 weeks and fell this week | manual |
| 6 | Capitulation judge | 0 = no signal · 1–2 = **oversold only** · 3 = building · 4–5 = **capitulation-like** | — |

Each sensor answers **MET**, **NOT MET** or **UNKNOWN**. Missing or stale data is UNKNOWN
and can never add to the score; the report shows how high the score could go once that
data arrives. Besides its verdict, each bot reports the article's context signals without
scoring them: VIX > 40 and the Cboe forward-return study (with its 2008/2020 caveat), the
VXN − VIX gap, "VXN falling while SOXX makes new lows", the put/call one-year percentile
and the < 0.5 euphoria zone, DIX ≥ 45%, crowded shorts (≥ 20%) and "ignition" (price
back above its 20-day average), Fear & Greed < 25, NAAIM, and the share of S&P 500
members above their 200-day average.

### Quick start

Python 3.11+, standard library only.

```sh
python -m bottom_bots list                     # the six bots, their rules and data
python -m bottom_bots run                      # fetch, score, print, save to state/
python -m bottom_bots run --bot put_call       # one sensor only (nothing saved)
python -m bottom_bots run --json               # machine-readable
python -m bottom_bots backfill --days 370      # a year of put/call history for the percentile

# The article's own 10/23/2026 readings — reproduces its "only sentiment is extreme" verdict:
python -m bottom_bots run --offline --manual inputs/manual.example.toml --as-of 2026-10-23 --no-save
```

### Daily routine

Run once a day **after the close** (Cboe publishes put/call ratios after the session;
intraday prices aren't closes):

1. Add any new manual readings to `inputs/manual.toml` (start from a copy of
   `inputs/manual.example.toml`) — one dated row per reading, e.g.
   ```toml
   [[aaii_bearish]]          # AAII, Thursdays
   date = 2026-10-29
   value = 47.2
   source = "AAII"
   ```
   `python -m bottom_bots list` names every metric; units are in `bottom_bots/manual.py`.
   `short_sale_volume` is rejected on purpose: short-sale volume ≠ short interest.
2. `python -m bottom_bots run` — the report prints and is saved to `state/reports/<date>.md`.

### Daily schedule (GitHub Actions + email)

`.github/workflows/daily.yml` runs the bots every weekday at **11:50 Chicago time** and
saves the day's result to the `bottom-bots-state` branch; a scheduled task in Claude
picks it up at 12:05 and emails it to you from your Gmail.

- It scores the **last completed session** (`--as-of last-session`): at noon today's
  closes don't exist yet, so Wednesday's report covers Tuesday's close.
- GitHub cron is UTC, so the workflow schedules 11:50 in both Central daylight and
  standard time and `scripts/daily.sh` keeps the entry that matches the day.
- State (the data cache, the run history and every report) lives on
  `bottom-bots-state`, so the persistence rule and the tranche ratchet carry over. Each
  run also writes `latest.md`, `latest.html`, `latest.txt` and `latest.json` there; the
  email sends `latest.html` and `latest.txt` exactly as rendered.
- A day with no new session (a market holiday, or a failed VIX download) gets a short
  note with the data warnings instead of a repeat report.
- Manual inputs are read from `inputs/manual.toml` on `main`; commit updates there.
- Run it on demand from the Actions tab: **daily checklist → Run workflow**.
- On pull requests the same job runs against live data without saving.
- The repository is public, so `bottom-bots-state`, with every report, is public too.

GitHub runs schedules only from the default branch, and can start one a few minutes
late; if the 12:05 email finds no result from today, it says so instead.

To run it on your own machine instead — cron uses the machine's time zone, so pick an
hour after the US close there; `TZ` makes the run date the New York trading day:

```cron
30 18 * * 1-5  cd /path/to/Trading && TZ=America/New_York python3 -m bottom_bots run >> state/cron.log 2>&1
```

### The plan is a dry run

The judge turns the score into a target share of a cash reserve you set aside, and lists
the orders that would get there. **It never sends an order.**

- **Persistence** — a score counts only after it has held for `persistence_runs`
  consecutive daily runs (default 2), so a first run never deploys anything and one wild
  day can't trigger a tranche.
- **Tranches** — cumulative target by persistent score: 0–2 → 0%, 3 → 25%, 4 → 50%,
  5 → 75%. Never 100%: "我宁愿错过最低点，也不愿在真正便宜的价格到来之前，就提前打光所有子弹".
- **Money-back gate** — without item 5 the target stops at the score-3 tranche:
  "底部猜不出来，钱回来了才算数".
- **Ratchet** — the target only rises. When the score falls the plan pauses; it never sells.

The tranche percentages, the instruments (QQQ/RSP, from your watchlist) and every
`[default]` threshold are my choices, not the article's — they live in
`bottom_bots/thresholds.toml` next to the article line each quoted number came from.
Override any of them with a small TOML file:

```toml
# my.toml — python -m bottom_bots run --config my.toml
[plan]
reserve_cash = 5000.0
instruments = { QQQ = 0.6, SOXX = 0.4 }

[aggregator]
persistence_runs = 3
```

Unknown keys are rejected, so a typo can't silently fall back to a default.

### Data

| Fetched automatically | Manual (`inputs/manual.toml`) — paywalled or no stable free API |
|---|---|
| VIX, VIX3M, VXN — Cboe index history CSVs | short interest % float, days to cover, borrow fee — FINRA / Nasdaq / Fintel / ORTEX / IBorrowDesk |
| Total / Index / Equity put/call — Cboe daily statistics | SOXX + SMH weekly net flows — ETF.com |
| GEX, DIX, SPX close — SqueezeMetrics `DIX.csv` | AAII bearish %, NAAIM exposure |
| Fear & Greed — CNN | % of S&P 500 above the 200-day — Barchart / StockCharts |
| Daily closes — Cboe chart history, then Yahoo, then Stooq | gamma flip level — SpotGamma / your model |

- A manual number beats a fetched one for the same day.
- A reading older than its `[staleness_days]` limit counts as missing: 5 days for daily
  data, 10 for weekly, 30 for short interest (published twice a month, a week late).
- Fetch failures become warnings in the report's **Data** section; they never crash a run.
- `state/series.json` caches fetched data only. Delete it to rebuild from scratch.

### Limits

- **Live sources, as checked from GitHub's runners on 10/28/2026:** Cboe (VIX, VIX3M, VXN,
  put/call, daily closes) and SqueezeMetrics answer. CNN refuses automated requests
  (HTTP 418), so enter `fear_greed` by hand if you want it; Yahoo rate-limits GitHub's
  runners and Stooq serves an HTML page, so both are only fallbacks. A source that stops
  answering shows up under **Data**, and its items fall back to manual inputs or UNKNOWN.
- Every GEX figure is a model estimate; public open interest can't prove dealers' real
  positioning. Stick to one source over time.
- The 10% / 20%-of-float thresholds are written for single stocks; ETF short interest
  (SOXX by default) reads differently. Point `[short_interest] symbol` at the names you
  actually watch.
- In the article's words: a checklist, not a crystal ball — "期权不是水晶球". Not investment advice.

## Strategy bots

Two bots that run fifteen common strategies on US stocks. `python -m strategy_bots list` shows them
all. Neither places a real order: the daily bot keeps a paper account in a JSON file, and the intraday
bot is a simulator.

| Bot | Strategies (numbers from the strategy table) | Works on |
|---|---|---|
| **Daily bot**: decides at each close what to hold from the next open | 1 pairs, 5 momentum, 8 seasonality, 9 volatility breakout, 10 machine learning, 11 chart patterns, 12 sentiment, 13 deep reinforcement learning, 14 adaptive, 15 genetic algorithm | daily bars |
| **Intraday bot**: works inside the trading day | 2 scalping, 3 smart order routing, 4 market making, 6 TWAP, 7 VWAP and implementation shortfall | 5-minute bars |

Why two: the split is by clock. The ten daily strategies answer the same question, "what should I hold
tomorrow?", so they share one backtest, one paper account and one no-look-ahead rule. The five
intraday ones work within a session. TWAP and VWAP carry out orders rather than pick them, and
scalping, routing and market making can't be run from a retail account at all (see the table below).

### Quick start

```sh
pip install numpy
python -m strategy_bots ingest <saved get_equity_historicals results>  # Robinhood connector -> data/daily, data/intraday
python -m strategy_bots backtest                      # all ten daily strategies vs equal weight and SPY
python -m strategy_bots backtest --strategy momentum --param lookback=63 --start 2023-01-03
python -m strategy_bots signal --strategy genetic    # targets for the next open, and why
python -m strategy_bots paper --since 2026-11-06     # advance all ten paper accounts (state/strategy_bots/)
python -m strategy_bots execute --bars data/intraday/SPY_5minute.csv --qty 20000 --minutes 60
python -m strategy_bots route
python -m strategy_bots scalp --bars data/intraday/SPY_5minute.csv --equity 10000
python -m strategy_bots make-market --bars data/intraday/AAPL_5minute.csv
```

`paper` with no `--strategy` advances all ten accounts at once and prints one table. Each account is a
JSON file in `state/strategy_bots/` (git-ignored). `--since DATE` opens new accounts at that close and
replays the days since. Because no strategy looks ahead, a replay ends exactly where daily runs would
have, so a lost state folder costs nothing: run the same command again.

Any CSV with `date,open,high,low,close,volume` in `data/daily/SYMBOL.csv` works too (the `quantbot`
branch's `data/*.csv` files are in that format). `data/` is git-ignored: the bars come from your broker's
connector and aren't republished in this public repo. `--param STRATEGY.KEY=VALUE` overrides any
default (a value is read as JSON when it parses), e.g. `--param 'pairs.pairs=[["GOOGL","META"]]'`.

### Weekday paper trading

A Claude scheduled task (set up in the owner's account, outside this repo) runs the daily bot at
8:13 pm New York time on weekdays. By then the UTC day has rolled over, which the Robinhood connector
needs before it returns the day's bar. The task:

1. fetches daily bars since 2019 for the 20 symbols with the connector's `get_equity_historicals`, its
   only Robinhood call;
2. ingests them;
3. runs `python -m strategy_bots paper --since 2026-11-06`;
4. reports the table.

It never places, changes or cancels an order.

### The daily bot

| # | Strategy | Rule |
|---|---|---|
| 1 | `pairs` | Every 21 days, tests KO/PEP, V/MA and XOM/CVX for cointegration over the last 250 days (Engle-Granger, 5%). A pair that passed trades when its spread is stretched 2σ: long the cheap leg, short the rich one, out at 0.5σ, a 4σ stop, 30 days or a failed retest. **Needs short selling.** |
| 5 | `momentum` | Monthly, up to 5 names with the best 6-month return (skipping the last month) that are above their 200-day average with RSI above 50; a name that loses its trend is sold early. |
| 8 | `seasonality` | Holds SPY over the turn of the month (the last trading day and the first three), skipping calendar months that have been weak in past years (z-score below −1). Calendar spreads need futures or options, which this bot doesn't trade. |
| 9 | `vol_breakout` | After a Bollinger squeeze (bandwidth in the bottom 20% of its last 120 days), buys a close above the upper band; 3-ATR trailing stop; positions sized to 20% volatility. |
| 10 | `ml` | A random forest, a neural net and a linear SVM, averaged, estimate the chance each stock is up 5 days later from 10 price features. Retrained every 126 days on the last 3 years; buys up to 5 names above 53%. |
| 11 | `patterns` | Zigzag swing pivots find inverse head and shoulders, double bottoms, ascending triangles and bull flags; buys the breakout with a stop under the pattern and a measured-move target; the bearish versions sell. |
| 12 | `sentiment` | Scores headlines with a finance word list (handles "fails to beat") and buys good news in an uptrend. Reads `inputs/headlines.csv` (`date,symbol,headline`); `headlines --feed URL --symbols ...` appends from any RSS or Atom feed. |
| 13 | `rl` | A deep Q-network (experience replay, target network) learns flat or long for each stock, deciding weekly; retrained yearly on the last 3 years. |
| 14 | `adaptive` | Kaufman's efficiency ratio picks each stock's regime: follow an adaptive moving average (KAMA) in a trend, buy 2-day-RSI dips in chop; volatility-targeted sizes. |
| 15 | `genetic` | Every 6 months, evolves a trend-plus-RSI rule (fast and slow average, RSI length and floor) on the last 3 years, scored on Sharpe ratio after costs and a turnover penalty. |

How the backtest keeps itself honest:

- A decision made at day t's close trades at day t+1's open and is held to the next open, so no
  strategy trades at a price it has already seen. Costs are 5 bps per unit of turnover (`--cost-bps`).
- The learners (10, 13, 15) retrain walk-forward, only on samples whose outcome was known by then.
- A test cuts the future off the data and checks that no strategy changes a decision it already made.
- Scoring starts after the longest warm-up (the RL agent needs three years), so every strategy is scored
  on the same days.
- Simplification: weights are held constant between decisions, which implies free daily rebalancing.
  The benchmarks are held the same way.

### Results on real data

Daily bars for 20 stocks and ETFs from the Robinhood connector (AAPL, MSFT, NVDA, GOOGL, AMZN, META,
AVGO, AMD, MU, INTC, MRVL, BKNG, KO, PEP, V, MA, XOM, CVX, SPY, QQQ), 1/2/2019 to 11/6/2026, scored
1/7/2022 to 11/5/2026 (4.8 years), default parameters:

| # | Strategy | CAGR | Volatility | Sharpe | Max drawdown | Turnover/yr | In the market |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | pairs | -0.1% | 0.5% | -0.25 | -1.5% | 0.4x | 0% |
| 5 | momentum | 24.1% | 28.5% | 0.90 | -26.9% | 10.3x | 80% |
| 8 | seasonality | 1.9% | 7.2% | 0.29 | -11.6% | 24.1x | 16% |
| 9 | vol_breakout | 11.3% | 13.9% | 0.84 | -15.8% | 21.5x | 66% |
| 10 | ml | -2.5% | 23.3% | 0.01 | -35.8% | 8.0x | 99% |
| 11 | patterns | 14.9% | 21.7% | 0.75 | -37.9% | 31.2x | 85% |
| 12 | sentiment | no headline history to test | | | | | |
| 13 | rl | 23.1% | 34.5% | 0.77 | -43.3% | 23.9x | 94% |
| 14 | adaptive | 9.3% | 14.2% | 0.69 | -16.2% | 62.0x | 75% |
| 15 | genetic | 12.0% | 12.7% | 0.95 | -20.4% | 8.4x | 55% |
| | equal weight, all 20 | 22.4% | 23.2% | 0.99 | -31.8% | | 100% |
| | buy and hold SPY | 9.8% | 17.5% | 0.62 | -26.0% | | 100% |

What it says, and what it doesn't:

- **Nothing beat holding all 20 names equally** on risk-adjusted return. Momentum earned a little more
  (24.1% a year against 22.4%) with a shallower worst drop (−27% against −32%), but it swung more, so
  its Sharpe ratio is lower. RL matched the return with a −43% drawdown, the worst in the table.
- **Lower risk is what some of them bought.** The volatility breakout, adaptive and genetic bots were
  in the market 55–75% of the time and lost 16–20% at worst, against 32% for equal weight and 26% for
  SPY, while earning about as much as SPY or more.
- **The universe flatters anything long.** These 20 names were picked in 2026 and include NVDA and
  AVGO, so equal weight, momentum and RL ride hindsight. Survivorship bias is real here.
- **Pairs and ML didn't work in this sample.** The three classic pairs were rarely cointegrated in
  2019–2026, so pairs barely traded. ML's 5-day forecasts added nothing: −2.5% a year while almost
  always invested.
- The defaults were written before this test and not tuned to it. Tune on one period and check on
  another, or the next table will look better than the future. One sample, one universe: not
  investment advice.

### The intraday bot

| # | Strategy | Rule | From a retail account? |
|---|---|---|---|
| 6 | TWAP | Equal slices over the window | Yes, by hand or with a broker algo |
| 7 | VWAP | Slices follow the usual volume curve, learned from earlier sessions; implementation shortfall (Almgren-Chriss) front-loads by `--urgency` | Yes |
| 2 | Scalping | Buying pressure in the last 3 bars (volume of up bars minus down bars) plus a break of the last bar's high; 15 bp target, 10 bp stop, 6 bars at most | Under $25,000 in a margin account, only 3 day trades per 5 business days (the pattern-day-trader rule), which the bot enforces |
| 3 | Smart order routing | Learns each venue's real fill rate and ranks venues by price plus the expected cost of a miss | No: brokers route retail orders themselves |
| 4 | Market making | Avellaneda-Stoikov quotes that lean against inventory; optional statistical-arbitrage lean from a correlated stock | No: needs exchange access, rebates and speed |

On SPY and AAPL 5-minute bars, 10/19 to 11/6/2026 (15 sessions):

- **Execution**: buying 20,000 SPY in the first hour of 11/6 cost +7.8 bps against the hour's VWAP with
  TWAP, +3.4 with VWAP and +0.1 with implementation shortfall. Against the 9:30 price all three paid
  55–63 bps, because SPY rallied that hour.
- **Routing** (simulated venues): the adaptive router filled 100% at 1.54 bps over the arrival mid. The
  static router filled 93% at 2.55 bps, because it kept sending 62% of its flow to a venue that shows the
  best price but fills only 20%.
- **Scalping** SPY: a $10,000 account got 9 trades (70 more signals were blocked by the PDT rule) for
  +11.4 bps in total. A $30,000 account got 55 trades for −27.5 bps: the edge doesn't cover the spread.
- **Market making**, 10-share lots: −$30 on AAPL, −$289 on SPY, and −$102 on AAPL with SPY as the
  stat-arb reference. The bars that trade through your bid are the ones that keep falling (adverse
  selection), and this bar-based fill model is still kinder than a real queue.

## Tests

```sh
pip install numpy   # the strategy bots need it; the bottom-confirmation bots are standard library only
python -m unittest discover -s tests -t .
```
