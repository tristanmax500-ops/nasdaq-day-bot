# Nasdaq Day-Trading Bot (paper trading)

A bot that **day trades QQQ and the 15 biggest Nasdaq-100 stocks** in an Alpaca paper account,
**learns from its own trades**, and **rewrites its strategy** when it finds a weakness.

* **Day trading only:** 5- or 15-minute candles. Every trade has a **take-profit and a stop-loss**
  (held by Alpaca's servers), and **everything is closed before the bell**, so nothing is held overnight.
* **Uses news and events:** news sentiment and buzz, CPI / jobs report / PPI days, Fed decision days,
  and company earnings dates. It learns *whether* these help; it doesn't assume.
* **Learns from mistakes:** after a losing streak, when it trades worse than its backtest promised, or
  when it spots a weak spot (e.g. "loses on CPI days"), it **writes a new version** of the strategy.
  The new version paper-trades side by side with the old one and takes over **only if it does better
  on new data**.
* **Hard safety rules:** max risk per trade, max open positions, max 2 losing trades per stock per
  day, a daily loss limit that closes everything, no new trades in the last 20 minutes, and no
  new trades around the Fed's 2:00 pm announcement.
* **TradingView:** each strategy is exported as a Pine Script, so you can see its trades on your own charts.

> **Honest expectations.** Nobody can promise good P&L. Most day-trading ideas lose money once
> costs are included, and this bot is built to **tell you the truth** rather than show pretty
> backtests. Many searches will find nothing, which is the bot refusing to fool you. Paper trade
> for months before you even think about real money, and if you ever do, start tiny.
> This is a research tool, not financial advice.

---

## 1. Get your free keys (10 minutes, once)

**Alpaca** (prices, news, paper trading):
1. Sign up at <https://alpaca.markets>. The free plan is fine.
2. In the dashboard, switch to **Paper Trading** (top-left).
3. On the right, under **API Keys**, click **Generate New Keys**. Copy the **Key** and the **Secret**
   (the secret is shown only once).

**FRED** (economic calendar: Fed, CPI, jobs, PPI dates). Optional, but recommended:
1. Create a free account at <https://fred.stlouisfed.org>.
2. Go to <https://fred.stlouisfed.org/docs/api/api_key.html>, request an API key, and copy it.

**Put the keys in `config.yaml`.** Open it with Notepad (Windows) or TextEdit (Mac):

```yaml
account:
  alpaca_key: "PKXXXXXXXXXXXXXXXX"
  alpaca_secret: "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  fred_api_key: "abcdef0123456789abcdef0123456789"
```

> **Give this bot its own paper account.** In Alpaca: click the account name (top-left) → **Open new
> paper account**, and use *that* account's keys. Your other Alpaca bot keeps its own account.
> (The bot never touches positions it didn't open, but sharing an account mixes up the balance,
> the daily loss limit and the results.)

## 2. Install (5 minutes, once)

1. Install **Python 3.12** from <https://www.python.org/downloads/>.
   On Windows, **tick "Add python.exe to PATH"**.
2. Unzip this folder somewhere easy, such as Documents.
3. Run the setup:
   * **Windows:** double-click **`1_SETUP_WINDOWS.bat`**
   * **Mac:** right-click **`1_SETUP_MAC.command`** → Open

## 3. First run

Double-click **`2_START_WINDOWS.bat`** (Windows) or **`2_START_MAC.command`** (Mac). A menu appears:

```
 1) Check my setup (Alpaca + FRED keys)
 2) Download / update data (prices, news, events)
 3) Find new strategies
 4) START THE BOT  (trades + learns automatically)
 5) Show status
 6) Open dashboard
 7) Export strategies to TradingView
 8) Self-test (proves it works, no internet needed)
```

1. **Press 8** (self-test, about 2 minutes). Everything should say PASS.
2. **Press 1** to check your keys.
3. **Press 2** to download data. The first time takes a while: 5½ years of 5-minute prices take a few
   minutes, and the news history can take 20–60 minutes. It's safe to stop and resume.
4. **Press 3** and give it 45–120 minutes. You'll see it test thousands of strategies and throw
   most of them away. Don't worry if none survive; run it again another time.
5. **Press 4** to start the bot and leave it running. It also searches for new strategies by
   itself after each close whenever it has a free slot.
6. **Press 6** any time to open the dashboard. It shows paper P&L, open trades with their TP/SL,
   the last trades, upcoming CPI/Fed/earnings days, the latest headlines, and everything the bot decided.

### Checking on it

* **Your Projects Dashboard**: the bot keeps a live **Nasdaq Day-Trading Bot** section up to date
  (equity, today's P&L, open trades with TP/SL, recent trades, strategies, and every decision it made).
  It writes `nasdaq_bot_status.js` into `Documents/Projects Dashboard` after every 5-minute candle.
  Change the folder in `config.yaml` → `dashboard: status_folder`.
* **Alpaca app** on your phone (Paper Trading): balance, positions and orders from anywhere.
* Menu **6** opens the bot's own detailed report.

### When does it need to run? (Bangkok time)

The US market is open **20:30 – 03:00 Bangkok time** until Nov 1 2026, and
**21:30 – 04:00** after that (US winter time). The computer must be **on and awake** during those hours
(Windows: Settings → System → Power → Sleep = Never; Mac: System Settings → Energy → prevent sleep).

If the computer sleeps anyway, Alpaca still holds each trade's take-profit and stop-loss. Positions
left over from a previous day are closed automatically when the bot comes back. If you stop the bot
(Ctrl+C) while the market is open, it closes its day trades first.

---

## How it works

```
 every 5 minutes while the market is open                 after the close
 ───────────────────────────────────────────            ───────────────────────────────────────
 new candle + news ──► strategies decide ──► Alpaca      JUDGE    live trades vs backtest promise
                          (TP + SL on every order)       DIAGNOSE where do losses cluster?
 safety rules: flat by close, daily loss limit,          REWRITE  targeted fixes + thousands of
 max positions, Fed blackout, no stale data                       variations, re-validated
                                                         PROVE    new version must beat the old
                                                                  one on NEW data to take over
                                                         SEARCH   find new strategies if a slot is free
```

**A strategy** is a readable rule, for example:

```
BUY on 5-min candles when: 2-candle momentum is below -1.9  AND  CPI/jobs/PPI release day: NO
EXIT: take-profit 2 ATR, stop-loss 1.5 ATR, after 12 candles, always before the close
```

The bot builds these from 26 ingredients: price action (RSI, momentum, VWAP, opening-range
breakouts, gaps, yesterday's high/low, relative volume, QQQ's direction), time of day, events
(CPI/jobs/PPI, Fed days, earnings) and news (sentiment, buzz, market mood).

**Before a strategy may trade**, it must pass 9 checks against 5½ years of data:

* enough trades
* profitable in at least 3 of 4 separate periods
* profitable on most of the stocks it trades
* profit factor ≥ 1.15 after costs
* better than 95% of copies of itself that enter at random times
* adds something beyond QQQ's own daily move
* not explainable by luck, given how many ideas were tried
* still works when its numbers are nudged slightly
* passes a one-time exam on the most recent 25% of data, which it has never seen

**Costs are learned.** The bot measures the real slippage on every Alpaca fill. If it's worse than
assumed, all future testing uses the real number.

### Proof it works (self-test, fake markets with a known answer)

```
 1  no false discoveries in noise        : PASS
 2  found the planted edge                : PASS
 3a noticed it was losing in a situation   : PASS
 3b diagnosed 'event days' as the cause   : PASS
 3c rewrote itself                        : PASS
 3d new version proved itself and took over: PASS
 4  positions held overnight              : 0 (PASS)
```

A real excerpt from that run:

```
WEAKSPOT  on event day it averages -0.05R over 45 trades vs +0.05R otherwise (p=0.0077)
REWRITE   v2 written: BUY when 2-candle momentum < -1.92 AND CPI/jobs/PPI release day: NO ...
PROMOTED  v2 beat v1 on new data: +1.46% vs +1.38% over 27 trades. Now live in paper trading.
```

The `tests` folder also runs the real bot against a simulated Alpaca for full trading days and
checks every safety rule (`python -m tests.test_live_days`, `python -m tests.test_units`).

In tests on fake markets, the search found real planted edges about half the time, and found
nothing in 6 out of 6 pure-noise markets. It is deliberately cautious.

---

## TradingView

Menu **7** writes one script per active strategy into the `tradingview` folder. To use one:
TradingView → open the stock on the same candle size (5m or 15m, regular hours) → **Pine Editor** →
paste → **Add to chart**. You'll see every entry and exit on the chart.

News, CPI/Fed and earnings conditions don't exist on TradingView, so those parts are switched off in
the script (it says which at the top). If TradingView ever shows an error, send it to me.

## Settings (`config.yaml`)

Every line has a comment. The most useful settings:

| Setting | Meaning |
|---|---|
| `risk_per_trade_pct` | % of the account lost if a stop is hit (default 0.5) |
| `max_daily_loss_pct` | bot stops for the day after this loss (default 2) |
| `max_open_positions` | default 6 |
| `allow_short` | also bet on falling prices (default off) |
| `timeframes` | candle sizes the AI may use |
| `loss_streak_trigger` | losses in a row that trigger an immediate rewrite (default 6) |
| `target_champions` | strategies trading at the same time (default 3) |
| `broker` | `alpaca` (real paper orders) or `sim` (simulation only) |

## Troubleshooting

* **"Alpaca refused the request (401/403)"**: the keys are wrong or are not *paper* keys.
* **Very slow:** check that `numba` installed (menu 1 shows it). Lower `workers` if the computer runs out of memory.
* **Dashboard empty:** no strategy has passed the checks yet. Run menu 3 for longer.
* **Anything else:** see `logs/bot.log`.

Files: `data/brain.db` holds everything the bot learned (delete it to start over), `reports/dashboard.html`
is the dashboard, `tradingview/` holds the scripts, and `logs/bot.log` is the full log.

*Paper trading only. Simulated and past performance does not guarantee future results. Not financial advice.*
