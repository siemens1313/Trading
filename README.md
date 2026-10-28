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

### Daily schedule (GitHub Actions)

`.github/workflows/daily.yml` runs the bots every weekday and posts the report as a
comment on the **Daily bottom checklist** issue around **noon Chicago time**,
@mentioning you so GitHub emails it (on by default for @mentions; see your GitHub
notification settings).

- It scores the **last completed session** (`--as-of last-session`): at noon today's
  closes don't exist yet, so Wednesday's report covers Tuesday's close.
- GitHub cron is UTC, so the workflow schedules 11:50 in both Central daylight and
  standard time and `scripts/daily.sh` keeps the entry that matches the day.
- State (the data cache, the run history and every report) lives on the
  `bottom-bots-state` branch, so the persistence rule and the tranche ratchet carry over.
- A day with no new session (a market holiday, or a failed VIX download) gets a short
  note with the data warnings instead of a repeat report.
- Manual inputs are read from `inputs/manual.toml` on `main`; commit updates there.
- Run it on demand from the Actions tab: **daily checklist → Run workflow**.
- On pull requests the same job runs against live data without saving or posting.

GitHub runs schedules only from the default branch, and can start one a few minutes late.

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
| Daily closes — Yahoo chart API, then Stooq | gamma flip level — SpotGamma / your model |

- A manual number beats a fetched one for the same day.
- A reading older than its `[staleness_days]` limit counts as missing: 5 days for daily
  data, 10 for weekly, 30 for short interest (published twice a month, a week late).
- Fetch failures become warnings in the report's **Data** section; they never crash a run.
- `state/series.json` caches fetched data only. Delete it to rebuild from scratch.

### Limits

- **The fetchers are untested against the live endpoints.** The build sandbox's network
  policy blocks every market-data host, so the parsers are tested against fixtures in each
  endpoint's published format (`tests/test_sources.py`). If a format has drifted, that
  source shows up under **Data** and its items fall back to manual inputs or UNKNOWN.
- Every GEX figure is a model estimate; public open interest can't prove dealers' real
  positioning. Stick to one source over time.
- The 10% / 20%-of-float thresholds are written for single stocks; ETF short interest
  (SOXX by default) reads differently. Point `[short_interest] symbol` at the names you
  actually watch.
- In the article's words: a checklist, not a crystal ball — "期权不是水晶球". Not investment advice.

### Tests

```sh
python -m unittest discover -s tests -t .
```
