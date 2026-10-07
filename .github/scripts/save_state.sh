#!/usr/bin/env bash
# Encrypt a snapshot of the bot's memory and save it (plus the public dashboard status file)
# on the 'state' branch. Usage: save_state.sh <snapshot.db> [reason]
set -euo pipefail
snap="$1"; reason="${2:-checkpoint}"
[ -n "${STATE_KEY:-}" ] || { echo "STATE_KEY missing" >&2; exit 1; }
work="$RUNNER_TEMP/statebranch"
rm -rf "$work"; mkdir -p "$work"
# keep the previous good copy as a fallback
if git fetch --depth=1 origin state 2>/dev/null; then
  git show FETCH_HEAD:brain.enc > "$work/brain.prev.enc" 2>/dev/null || true
fi
openssl enc -aes-256-cbc -pbkdf2 -iter 200000 -salt -pass env:STATE_KEY -in "$snap" -out "$work/brain.enc"
sd="${NASDAQ_STATUS_DIR:-}"
if [ -n "$sd" ] && [ -f "$sd/nasdaq_bot_status.json" ]; then
  cp "$sd/nasdaq_bot_status.json" "$work/status.json"
  cp "$sd/nasdaq_bot_status.js" "$work/nasdaq_bot_status.js" 2>/dev/null || true
fi
cat > "$work/README.md" <<'MD'
Saved memory of the Nasdaq day-trading bot. `brain.enc` is encrypted (AES-256, key only in the
repo's STATE_KEY secret). `status.json` is the paper-trading status shown on the Projects Dashboard.
MD
cd "$work"
git init -q -b state
git config user.name "nasdaq-bot"
git config user.email "nasdaq-bot@users.noreply.github.com"
git add -A
git commit -q -m "memory: $reason $(date -u +%Y-%m-%dT%H:%MZ)"
url="$(git -C "$GITHUB_WORKSPACE" remote get-url origin)"
auth="$(git -C "$GITHUB_WORKSPACE" config --get http.https://github.com/.extraheader || true)"
for i in 1 2 3; do
  if git -c "http.https://github.com/.extraheader=$auth" push -q -f "$url" state:state; then exit 0; fi
  sleep $((i * 5))
done
exit 1
