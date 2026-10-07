# Running the bot on GitHub (PC can be off)

GitHub starts the bot before every US session (Mon–Fri), it trades exactly like AUTO mode,
closes everything 5 min before the bell, does the after-close review + strategy search,
saves its memory (encrypted) and stops. Free on a public repo.

## One-time setup (you do this – keys never go in chat)
1. github.com/new → name `nasdaq-day-bot`, **Public**, no README → Create.
2. Repo → Settings → Secrets and variables → Actions → **New repository secret**, four times:
   - `ALPACA_KEY`     – your Alpaca **paper** key (same as in config.yaml)
   - `ALPACA_SECRET`  – your Alpaca paper secret
   - `FRED_API_KEY`   – your FRED key
   - `STATE_KEY`      – the text inside `STATE_KEY.txt` in this folder (encrypts the bot's memory)
3. Stop the bot on this PC for good (close its window). Only ONE bot may trade the account.

## Searching for strategies (on your PC)
- Double-click **SEARCH_AND_SEND_TO_GITHUB.bat**. When the search ends, the strategies it validated
  are sent (encrypted) to GitHub; the next session trades them, a running one within 30 min.
- Re-send by hand: `python run.py publish`. Needs Git installed (the first time it may ask you to sign in to GitHub).
- This PC never trades any more (`trading_host: github` in config.yaml) – START refuses on purpose.

## Day to day
- Watch it: repo → **Actions** tab (each run's log shows every trade decision).
- Start one by hand: Actions → "Nasdaq bot - trading session" → Run workflow.
- Pause it: Actions → the workflow → "…" → Disable workflow.
- The memory lives on the `state` branch: `brain.enc` (encrypted) + `status.json`
  (paper balances/trades for the dashboard – public, no keys).
- GitHub pauses schedules on public repos after 60 days with no activity; the bot's daily
  memory saves count as activity.
