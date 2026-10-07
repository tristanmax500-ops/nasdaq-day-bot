# Ideas parked for later (don't build until Tristan says go)

## New building blocks for the strategy search (asked about 5 Oct 2026)
1. **Today's opening move vs the previous day**: e.g. today's gap compared with yesterday's open->close
   move (continuation vs reversal), and yesterday's return in daily-vol units.
   (Already exists: `gap` = today's gap on its own.)
2. **Volatility regime (calm vs wild days)**: yesterday's / last-5-days realised range vs the
   60-day normal, as a day-level value (calm < 0.8, normal, wild > 1.3).
   (Already exists: `volat` = last 12/24 candles vs the 20-day normal, intraday only.)
3. **Weekly trend**: QQQ's 5-day return in vol units and price vs its 5-day / 20-day daily average
   (up week / down week / sideways).

Notes for building them: daily values must use only data up to the previous close (no lookahead),
add to `trader/features.py` FEATURES + `_compute`, re-run the unit tests, live-day test and self-test.
More building blocks = more ideas tried, so the luck test gets stricter; that's intended.
