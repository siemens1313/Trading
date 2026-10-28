#!/usr/bin/env bash
# Daily checklist run for .github/workflows/daily.yml.
#
# Scores the last completed session against the state kept on the bottom-bots-state
# branch (checked out at ./state), then posts the report as a comment on the "Daily
# bottom checklist" issue, @mentioning the repo owner so GitHub emails it, and pushes
# the new state. DRY_RUN=1 (pull requests) stops before the commit, comment and push.
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
report="$state/reports/$as_of.md"
if git -C "$state" ls-files --error-unmatch "reports/$as_of.md" > /dev/null 2>&1; then
  new=0  # already posted: a holiday, or the VIX download failed and the cache ends earlier
else
  new=1
fi

python - "$out/run.json" "$report" "$new" "${GITHUB_REPOSITORY_OWNER:-}" > "$out/comment.md" <<'PY'
import json, sys

run_path, report_path, new, owner = sys.argv[1:]
run = json.load(open(run_path))
mention = f"@{owner} " if owner else ""
if new == "1":
    print(f"{mention}Bottom checklist for the {run['as_of']} session:\n")
    print(open(report_path).read())
else:
    warnings = run["data_warnings"]
    vix_failed = any(w.startswith("Cboe VIX:") for w in warnings)
    reason = "the Cboe VIX download failed" if vix_failed else "probably a market holiday"
    print(f"{mention}No new trading session since {run['as_of']} ({reason}), so no new report today.\n")
    print("\n".join(f"- {w}" for w in warnings) or "- no data warnings")
PY

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  cat "$out/comment.md" >> "$GITHUB_STEP_SUMMARY"
fi

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "DRY_RUN: session $as_of, new report: $new. State changes not committed:"
  git -C "$state" status --short
  echo "--- comment that would be posted ---"
  cat "$out/comment.md"
  exit 0
fi

git -C "$state" config user.name "github-actions[bot]"
git -C "$state" config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git -C "$state" add -A
git -C "$state" diff --cached --quiet || git -C "$state" commit -q -m "Checklist for the $as_of session"

# Post before pushing: if the push then fails, tomorrow re-posts rather than skipping a day.
issue=$(gh issue list --label bottom-checklist --state open --json number --jq '.[0].number // empty')
if [ -z "$issue" ]; then
  gh label create bottom-checklist --color 0E8A16 --description "Daily bottom-checklist reports" --force > /dev/null
  url=$(gh issue create --title "Daily bottom checklist" --label bottom-checklist \
    --body "One comment per trading day around noon Central, posted by .github/workflows/daily.yml.")
  issue=${url##*/}
fi
gh issue comment "$issue" --body-file "$out/comment.md"
git -C "$state" push -q origin HEAD:bottom-bots-state
