#!/usr/bin/env bash
# Publish the small dashboard status file on its own 'status' branch (cheap, every ~5 min).
set -euo pipefail
sd="${NASDAQ_STATUS_DIR:-}"
[ -n "$sd" ] && [ -f "$sd/nasdaq_bot_status.json" ] || exit 0
work="$RUNNER_TEMP/statusbranch"
rm -rf "$work"; mkdir -p "$work"
cp "$sd/nasdaq_bot_status.json" "$work/status.json"
cd "$work"
git init -q -b status
git config user.name "nasdaq-bot"
git config user.email "nasdaq-bot@users.noreply.github.com"
git add -A
git commit -q -m "status $(date -u +%H:%MZ)"
url="$(git -C "$GITHUB_WORKSPACE" remote get-url origin)"
auth="$(git -C "$GITHUB_WORKSPACE" config --get http.https://github.com/.extraheader || true)"
git -c "http.https://github.com/.extraheader=$auth" push -q -f "$url" status:status
