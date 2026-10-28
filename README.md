# bottom-bots-state

Runtime state for the daily bottom checklist, written by `.github/workflows/daily.yml`
on `main`:

- `series.json` — the cache of fetched data
- `history.json` — past runs, behind the persistence rule and the tranche ratchet
- `reports/` — one report per session

Don't edit it by hand: deleting `history.json` resets persistence and the ratchet.
