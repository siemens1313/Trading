#!/usr/bin/env bash
# Daily checklist run for .github/workflows/daily.yml.
#
# Scores the last completed session against the state kept on the bottom-bots-state
# branch (checked out at ./state), writes the day's result to latest.{md,html,txt,json}
# there for the scheduled task that emails it, and pushes the new state.
# DRY_RUN=1 (pull requests) stops before the commit and push.
set -euo pipefail

state=state
out=${RUNNER_TEMP:-$(mktemp -d)}

# GitHub cron is UTC, so the workflow schedules 11:50 in both Central daylight and
# standard time; keep the entry that is 11:50 in Chicago today.
if [ "${GITHUB_EVENT_NAME:-}" = schedule ]; then
  case "$(TZ=America/Chicago date +%z)" in
    -0500) wanted="50 16 * * 1-5" ;;
    *) wanted="50 17 * * 1-5" ;;
  esac
  if [ "${SCHEDULE:-}" != "$wanted" ]; then
    echo "Schedule '${SCHEDULE:-}' isn't 11:50 in Chicago today; nothing to do."
    exit 0
  fi
fi

python -m bottom_bots run --as-of last-session --state-dir "$state" --json > "$out/run.json"
as_of=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["as_of"])' "$out/run.json")
if git -C "$state" ls-files --error-unmatch "reports/$as_of.md" > /dev/null 2>&1; then
  new=0  # already reported: a holiday, or the VIX download failed and the cache ends earlier
else
  new=1
fi

python - "$out/run.json" "$state" "$new" <<'PY'
import json, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from bottom_bots.report import markdown_to_html, markdown_to_text

run_path, state, new = sys.argv[1:]
run, state = json.load(open(run_path)), Path(state)
if new == "1":
    message = f"Bottom checklist for the {run['as_of']} session:\n\n" + (state / "reports" / f"{run['as_of']}.md").read_text()
else:
    warnings = run["data_warnings"]
    reason = "the Cboe VIX download failed" if any(w.startswith("Cboe VIX:") for w in warnings) else "probably a market holiday"
    message = f"No new trading session since {run['as_of']} ({reason}), so no new report today.\n\n"
    message += "\n".join(f"- {w}" for w in warnings) or "- no data warnings"
    message += "\n"
(state / "latest.md").write_text(message)
(state / "latest.html").write_text(markdown_to_html(message))
(state / "latest.txt").write_text(markdown_to_text(message))
summary = {
    "run_date": datetime.now(ZoneInfo("America/Chicago")).date().isoformat(),
    "session": run["as_of"],
    "new_report": new == "1",
    "score": run["score"],
    "unknown": run["unknown"],
    "regime": run["regime"],
}
(state / "latest.json").write_text(json.dumps(summary, indent=1) + "\n")
PY

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  cat "$state/latest.md" >> "$GITHUB_STEP_SUMMARY"
fi

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "DRY_RUN: session $as_of, new report: $new. State changes not committed:"
  git -C "$state" status --short
  echo "--- latest.json ---"
  cat "$state/latest.json"
  echo "--- latest.md ---"
  cat "$state/latest.md"
  exit 0
fi

git -C "$state" config user.name "github-actions[bot]"
git -C "$state" config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git -C "$state" add -A
git -C "$state" diff --cached --quiet || git -C "$state" commit -q -m "Checklist for the $as_of session"
git -C "$state" push -q origin HEAD:bottom-bots-state
