#!/usr/bin/env bash
# Restore the bot's memory (data/brain.db) from the encrypted copy on the 'state' branch.
set -euo pipefail
mkdir -p data
if [ -z "${STATE_KEY:-}" ]; then echo "::error::STATE_KEY secret is missing"; exit 1; fi
if ! git fetch --depth=1 origin state 2>/dev/null; then
  # first run: start from the memory copied from your PC (encrypted with the same STATE_KEY)
  if [ -f brain.seed.enc ]; then
    openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -pass env:STATE_KEY -in brain.seed.enc -out data/brain.db \
      || { echo "::error::Could not decrypt brain.seed.enc - STATE_KEY secret must be the one in STATE_KEY.txt"; exit 1; }
    echo "First run: memory imported from your PC"
  else
    echo "No saved memory yet - starting fresh."
  fi
  exit 0
fi
for f in brain.enc brain.prev.enc; do
  if git show "FETCH_HEAD:$f" > "$RUNNER_TEMP/$f" 2>/dev/null && \
     openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -pass env:STATE_KEY \
       -in "$RUNNER_TEMP/$f" -out "$RUNNER_TEMP/brain.db" 2>/dev/null && \
     python -c "import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); assert c.execute('pragma integrity_check').fetchone()[0]=='ok'" "$RUNNER_TEMP/brain.db"; then
    mv "$RUNNER_TEMP/brain.db" data/brain.db
    echo "Memory restored from $f"
    exit 0
  fi
done
echo "::error::Saved memory exists but could not be decrypted - was STATE_KEY changed?"
exit 1
