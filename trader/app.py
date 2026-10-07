"""The bot: commands used by run.py, and the automatic market-hours loop."""
import logging
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .config import LOG_DIR, TV_DIR, load_config, cost_per_side, watch_symbols
from .registry import Registry

NY = "America/New_York"


def make_logger():
    logger = logging.getLogger("nasdaq_bot")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(LOG_DIR / "bot.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
        logger.addHandler(fh)

    def log(msg):
        print(msg, flush=True)
        logger.info(msg)
    return log


class Bot:
    def __init__(self, cfg=None, log=None, api=None, provider=None, registry=None, news=None, events=None):
        self.cfg = cfg or load_config()
        self.log = log or make_logger()
        self.reg = registry or Registry(self.cfg["db_path"])
        self._api = api
        self._provider = provider
        self._news = news
        self._events = events
        self._last_report = 0
        if self.reg.meta("first_run") is None:
            self.reg.set_meta("first_run", pd.Timestamp.now(tz=NY).date().isoformat())

    def feed(self, state="running", note=""):
        """Update the live card on your Projects Dashboard (never stops the bot if it fails)."""
        try:
            from .dashboard_feed import write_status
            write_status(self, state, note)
        except Exception as e:
            self.log(f"  ! dashboard update failed: {e}")

    # ---------------- lazy services ----------------
    @property
    def api(self):
        if self._api is None:
            from .alpaca_api import Alpaca
            a = self.cfg["account"]
            self._api = Alpaca(a["alpaca_key"], a["alpaca_secret"], log=self.log)
        return self._api

    @property
    def provider(self):
        if self._provider is None:
            from .data import AlpacaProvider
            self._provider = AlpacaProvider(self.cfg, self.api, self.log)
        return self._provider

    @property
    def news(self):
        if self._news is None and self.cfg["news"].get("enabled", True):
            from .news import NewsStore
            self._news = NewsStore()
        return self._news

    @property
    def events(self):
        if self._events is None:
            from .events import EventCalendar
            self._events = EventCalendar(self.cfg, self.log).load()
        return self._events

    # ---------------- building the market view ----------------
    def market(self, research_frac=1.0):
        from .evaluate import EvalContext
        panels, extras = {}, {}
        for tf in self.cfg["market"]["timeframes"]:
            p = self.provider.panel(tf)
            panels[tf] = p
            ex = {}
            if self.news is not None:
                ex.update(self.news.features(p))
            ex.update(self.events.features(p))
            extras[tf] = {k: v.astype(np.float32) for k, v in ex.items()}
        fomc = set(self.events.fomc_dates())
        cost = cost_per_side(self.cfg, self.reg)
        ctx = EvalContext(panels, self.cfg, extras, fomc, research_frac=research_frac, cost=cost)
        return SimpleNamespace(panels=panels, extras=extras, fomc=fomc, cost=cost, ctx=ctx)

    # ---------------- commands ----------------
    def check_setup(self):
        ok = True
        self.log("Checking your setup ...")
        try:
            acct = self.api.account()
            self.log(f"  Alpaca paper account OK - equity ${float(acct['equity']):,.2f}, "
                     f"buying power ${float(acct['buying_power']):,.2f}")
            if acct.get("trading_blocked") or acct.get("account_blocked"):
                self.log("  ! Alpaca says this account is blocked from trading")
                ok = False
        except Exception as e:
            self.log(f"  ! Alpaca: {e}")
            ok = False
        if self.cfg["account"].get("fred_api_key"):
            try:
                import requests
                r = requests.get("https://api.stlouisfed.org/fred/release",
                                 params={"release_id": 10, "api_key": self.cfg["account"]["fred_api_key"],
                                         "file_type": "json"}, timeout=20)
                r.raise_for_status()
                self.log("  FRED key OK (economic calendar)")
            except Exception as e:
                self.log(f"  ! FRED key problem: {e}")
                ok = False
        else:
            self.log("  note: no FRED key - the bot will work, but can't learn from CPI/jobs/Fed days")
        from .backtest import HAVE_NUMBA
        self.log("  speed-up (numba): " + ("installed" if HAVE_NUMBA else "MISSING - testing will be ~40x slower"))
        return ok

    def download(self, force=False):
        self.log("Updating market data ...")
        self.provider.refresh(force=force)
        self.provider.sessions(force=force)
        if self.news is not None:
            n = self.news.update(self.api, watch_symbols(self.cfg), self.cfg["news"]["history_start"], self.log)
            self.log(f"  news: {n:,} new articles")
        self.events.refresh(force=force)
        for tf in self.cfg["market"]["timeframes"]:
            p = self.provider.panel(tf)
            self.log(f"  {tf}: {len(p.symbols)} symbols, {p.T:,} candles, {p.n_days} trading days "
                     f"({p.day_dates[0]} -> {p.day_dates[-1]})")

    def discover(self, minutes=None, trials=None, seed=None):
        from .evolve import discover
        from .healer import fill_slots
        M = self.market(research_frac=None)
        ids = discover(self.cfg, M.panels, self.reg, minutes=minutes, max_trials=trials, seed=seed, log=self.log,
                       extras=M.extras, fomc_dates=M.fomc, cost=M.cost)
        fill_slots(self.cfg, self.reg, self.log)
        self.export_tradingview()
        self.report(force=True)
        self.feed("idle", "strategy search finished")
        return ids

    def bench(self, n=300, write=None):
        """Measure how many strategies per second this computer tests (1 core and all cores)."""
        import platform
        import numpy as np
        from .evaluate import EvalContext
        from .evolve import auto_workers
        from .genome import random_genome
        from .backtest import HAVE_NUMBA
        M = self.market(research_frac=None)
        ctx = EvalContext(M.panels, self.cfg, M.extras, M.fomc, None, M.cost)
        rng = np.random.default_rng(7)
        tfs = list(M.panels)
        gs = [random_genome(rng, tfs) for _ in range(n)]
        for g in gs[:40]:
            ctx.score(g)                    # warm up: compile + build features
        t = time.time()
        for g in gs:
            ctx.score(g)
        one = n / (time.time() - t)
        workers, why = auto_workers(self.cfg)
        arch = f"{platform.machine()} / Python {platform.python_version()} ({platform.architecture()[0]})"
        self.log(f"SPEED TEST: {one:.0f} strategies/s on 1 core | {workers} cores in a search ({why}) "
                 f"-> about {one * workers * 0.85:.0f}/s | {arch} | numba {'ON' if HAVE_NUMBA else 'OFF'}")
        if write:
            from pathlib import Path
            Path(write).write_text(f"{one:.3f}\n")
        return one

    def paper(self, quiet=False, M=None):
        from .paper import paper_step
        M = M or self.market()
        return paper_step(M.ctx, self.reg, (lambda m: None) if quiet else self.log)

    def heal(self, seed=None, M=None):
        from .healer import heal_cycle
        self.log("Self-improvement check ...")
        M = M or self.market()
        heal_cycle(self.cfg, M, self.reg, self.log, seed=seed)

    def export_tradingview(self):
        from .pine import export
        return export(self.reg, TV_DIR)

    def report(self, force=False):
        from .report import build
        if not force and time.time() - self._last_report < 60:
            return None
        self._last_report = time.time()
        return build(self.cfg, self.reg, extra_html=self._upcoming_html())

    def _upcoming_html(self):
        try:
            today = pd.Timestamp.now(tz=NY).date()
            horizon = today + pd.Timedelta(days=7)
            items = []
            names = {"fomc": "Fed decision 2:00pm ET", "cpi": "CPI 8:30am ET", "nfp": "Jobs report 8:30am ET",
                     "ppi": "PPI 8:30am ET"}
            for k, label in names.items():
                for d in self.events.macro.get(k, ()):
                    if today <= d <= horizon:
                        items.append((d, label))
            for s, rows in self.events.earnings.items():
                for d, tm in rows:
                    if today <= d <= horizon:
                        items.append((d, f"{s} earnings ({ {'amc': 'after close', 'bmo': 'before open'}.get(tm, 'time n/a') })"))
            items.sort()
            li = "".join(f"<li><b>{d:%a %b %d}</b> - {l}</li>" for d, l in items) or "<li class='muted'>nothing scheduled</li>"
            mood = ""
            if self.news is not None:
                rows = self.news.recent_headlines(hours=24, limit=8)
                if rows:
                    avg = float(np.mean([r[2] for r in rows]))
                    mood = ("<h2>Latest headlines</h2><ul class='small'>" +
                            "".join(f"<li>{'🟢' if r[2] > 0 else ('🔴' if r[2] < 0 else '⚪')} {r[3]}</li>" for r in rows)
                            + "</ul>")
            return f"<h2>Next 7 days</h2><ul>{li}</ul>{mood}"
        except Exception:
            return ""

    def status(self):
        for st in ["champion", "challenger", "reserve"]:
            for s in self.reg.list([st]):
                ps = (self.reg.paper_state(s["id"]) or {}).get("stats", {})
                self.log(f"  {st:<10} {s['id']} v{s['version']} {s['genome']['tf']:<5} {ps.get('trades', 0):>4} trades  "
                         f"P&L ${ps.get('pnl_usd', 0):+,.0f}  [{s.get('health')}]")
        self.log(f"  lifetime strategies tested: {self.reg.meta('total_trials', 0):,}")

    # ---------------- the automatic loop ----------------
    def bar_cycle(self):
        self.provider.refresh()
        if self.news is not None:
            try:
                self.news.update(self.api, watch_symbols(self.cfg), self.cfg["news"]["history_start"], self.log)
            except Exception as e:
                self.log(f"  ! news update failed: {e}")
        M = self.market()
        results = self.paper(quiet=True, M=M)
        if self.cfg["paper"]["broker"] == "alpaca":
            from .broker import Broker
            champs = {s["id"] for s in self.reg.list(["champion"])}
            Broker(self.cfg, self.api, self.reg, self.log).sync({k: v for k, v in results.items() if k in champs})
        self.report()
        self.feed("running")
        return M

    def after_close(self):
        self.log("Market closed - daily review ...")
        if self.cfg["paper"]["broker"] == "alpaca":
            from .broker import Broker
            try:
                Broker(self.cfg, self.api, self.reg, self.log).learn_costs()
            except Exception as e:
                self.log(f"  ! could not update costs from fills: {e}")
        self.download()
        M = self.market()
        self.paper(quiet=True, M=M)
        self.heal(M=M)
        self.paper(quiet=True, M=M)
        need = int(self.cfg["paper"]["target_champions"]) - len(self.reg.list(["champion"]))
        if need > 0 and not self.reg.list(["reserve"]) and self.cfg["auto"].get("search_after_close", True):
            mins = float(self.cfg["auto"]["discovery_minutes_after_close"])
            self.log(f"{need} strategy slot(s) empty - searching for new strategies for {mins:g} minutes")
            self.discover(minutes=mins)
        self.export_tradingview()
        self.report(force=True)
        self.feed("market closed", "daily review done")
        self.status()

    def auto(self):
        self.log("AUTO MODE: trades every candle while the market is open, reviews and improves itself "
                 "after the close. Press Ctrl+C to stop (open positions keep their TP/SL at Alpaca).")
        flat_m = float(self.cfg["day_trading"]["flatten_minutes_before_close"])
        self._keep_awake()
        while True:
            try:
                clock = self.api.clock()
                now = pd.Timestamp(clock["timestamp"]).tz_convert("UTC")
                if clock["is_open"]:
                    self.bar_cycle()
                    nxt = now.floor("5min") + pd.Timedelta(minutes=5, seconds=10)
                    flat_t = pd.Timestamp(clock["next_close"]).tz_convert("UTC") - pd.Timedelta(minutes=flat_m) \
                        + pd.Timedelta(seconds=5)
                    wake = flat_t if now < flat_t < nxt else nxt
                    self._sleep_until(wake)
                else:
                    today = now.tz_convert(NY).date().isoformat()
                    nyh = now.tz_convert(NY).hour
                    traded_today = self._was_trading_day(now)
                    if self.reg.meta("after_close_done") != today and nyh >= 16 and traded_today:
                        self.after_close()
                        self.reg.set_meta("after_close_done", today)
                    nopen = pd.Timestamp(clock["next_open"]).tz_convert("UTC")
                    wake = min(now + pd.Timedelta(minutes=30), nopen - pd.Timedelta(minutes=10))
                    if wake <= now:
                        wake = nopen + pd.Timedelta(seconds=5)
                    self.feed("market closed", f"next open {nopen.tz_convert(NY):%a %b %d %H:%M} ET")
                    self.log(f"Market closed. Next open {nopen.tz_convert(NY):%a %H:%M} ET. "
                             f"Sleeping until {wake.tz_convert(NY):%H:%M} ET ...")
                    self._sleep_until(wake)
            except KeyboardInterrupt:
                self._stop_safely()
                self.feed("stopped", "stopped by you")
                raise
            except Exception as e:
                if "network error" in str(e):
                    # internet down / computer just woke up: one short line, then try again
                    self.log("! no internet connection right now (Alpaca unreachable) - retrying in 1 minute")
                    self.feed("error", "no internet connection - retrying")
                    time.sleep(60)
                    continue
                self.log(f"! error: {e}\n{traceback.format_exc()}")
                self.feed("error", str(e)[:200])
                try:
                    self.reg.event(None, "error", str(e)[:300])
                except Exception:
                    pass
                time.sleep(60)

    # ---------------- one GitHub Actions job ----------------
    def session(self):
        """Used on GitHub (your PC can be off). Trades today's session exactly like AUTO mode,
        does the after-close review + search, saves the bot's memory and exits.
        GitHub stops a job after 6 hours, so before that it hands over to a fresh job -
        never in the last 40 minutes of the session, when it closes the day's trades."""
        import os
        from . import cloud
        t0 = time.monotonic()
        handoff = float(os.environ.get("NASDAQ_HANDOFF_MIN", 300)) * 60
        hard = float(os.environ.get("NASDAQ_HARD_STOP_MIN", 340)) * 60
        save_every = 30 * 60
        last_save = time.monotonic()
        flat_m = float(self.cfg["day_trading"]["flatten_minutes_before_close"])
        cloud.import_seed(self.reg, self.log)
        cloud.check_inbox(self.reg, self.cfg, self.log)
        self.cfg["auto"]["search_after_close"] = False      # searching is done on your PC
        self.log("SESSION MODE (GitHub): trades today's session, reviews after the close, saves and exits.")

        def elapsed():
            return time.monotonic() - t0

        def finish(reason, continue_=False):
            if continue_:
                cloud.dispatch_continuation(self.log)
            cloud.save_state(self.log, reason)
            return reason

        def busy_window(now):
            # 15:20-16:05 ET: flatten + close. A handover here could miss the flatten.
            et = now.tz_convert(NY)
            m = et.hour * 60 + et.minute
            return 15 * 60 + 20 <= m <= 16 * 60 + 5

        errors = 0
        while True:
            try:
                clock = self.api.clock()
                now = pd.Timestamp(clock["timestamp"]).tz_convert("UTC")
                today = now.tz_convert(NY).date().isoformat()
                if elapsed() >= hard or (elapsed() >= handoff and not busy_window(now)):
                    self.log(f"Job has run {elapsed() / 60:.0f} min - handing over to a fresh GitHub job")
                    self.feed("running", "handing over to the next GitHub job")
                    return finish("handover", continue_=True)
                if time.monotonic() - last_save >= save_every:
                    cloud.check_inbox(self.reg, self.cfg, self.log)
                    cloud.save_state(self.log, "checkpoint")
                    last_save = time.monotonic()
                if clock["is_open"]:
                    self.bar_cycle()
                    nxt = now.floor("5min") + pd.Timedelta(minutes=5, seconds=10)
                    flat_t = pd.Timestamp(clock["next_close"]).tz_convert("UTC") - pd.Timedelta(minutes=flat_m) \
                        + pd.Timedelta(seconds=5)
                    wake = flat_t if now < flat_t < nxt else nxt
                    self._sleep_until(wake)
                    errors = 0
                    continue
                nopen = pd.Timestamp(clock["next_open"]).tz_convert("UTC")
                traded_today = self._was_trading_day(now)
                nyh = now.tz_convert(NY).hour
                if traded_today and nyh >= 12:          # closed after midday = today's session is over
                    if self.reg.meta("after_close_done") != today:
                        cloud.check_inbox(self.reg, self.cfg, self.log)
                        left = (hard - elapsed()) / 60 - 25          # keep time to save the memory
                        mins = float(self.cfg["auto"]["discovery_minutes_after_close"])
                        if left < 30:
                            self.log("Not enough time left in this job for the review - next job does it")
                            return finish("handover before review", continue_=True)
                        self.cfg["auto"]["discovery_minutes_after_close"] = max(10.0, min(mins, left - 20))
                        self.after_close()
                        self.cfg["auto"]["discovery_minutes_after_close"] = mins
                        self.reg.set_meta("after_close_done", today)
                    self.feed("market closed", f"next open {nopen.tz_convert(NY):%a %b %d %H:%M} ET")
                    self.log("Day finished. Saving memory and stopping until the next session.")
                    return finish("day done")
                # before the open: wait for it if it's today and soon, otherwise nothing to do
                wait = (nopen - now).total_seconds() / 60
                if wait > 150:
                    self.feed("market closed", f"next open {nopen.tz_convert(NY):%a %b %d %H:%M} ET")
                    self.log(f"No session soon (next open {nopen.tz_convert(NY):%a %H:%M} ET) - nothing to do.")
                    return finish("no session")
                self.feed("market closed", f"opens {nopen.tz_convert(NY):%H:%M} ET")
                self.log(f"Waiting for the open at {nopen.tz_convert(NY):%H:%M} ET ({wait:.0f} min) ...")
                self._sleep_until(min(nopen + pd.Timedelta(seconds=5), now + pd.Timedelta(minutes=10)))
            except KeyboardInterrupt:
                self.log("GitHub stopped the job - saving memory (positions keep their TP/SL at Alpaca)")
                finish("cancelled")
                raise
            except Exception as e:
                errors += 1
                self.log(f"! error: {e}\n{traceback.format_exc()}")
                self.feed("error", str(e)[:200])
                try:
                    self.reg.event(None, "error", str(e)[:300])
                except Exception:
                    pass
                if errors >= 30:
                    return finish("too many errors", continue_=True)
                time.sleep(60)

    def _keep_awake(self):
        """Windows: ask Windows not to put the computer to sleep while the bot runs
        (the same request video players make). Closing the laptop lid can still sleep it."""
        import os
        if os.name != "nt":
            return
        try:
            import ctypes
            ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
            if ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
                self.log("Keeping the computer awake while the bot runs (the screen may still turn off).")
        except Exception:
            pass

    def _was_trading_day(self, now):
        """True if the US market was open today (weekends and holidays: no after-close review)."""
        d = now.tz_convert(NY).date().isoformat()
        try:
            cal = self.api.calendar(d, d)
            return any(str(c.get("date")) == d for c in (cal or []))
        except Exception:
            return now.tz_convert(NY).weekday() < 5

    def _stop_safely(self):
        """Stopping during market hours: close day trades so nothing is held overnight."""
        if self.cfg["paper"]["broker"] != "alpaca":
            return
        try:
            if self.api.clock().get("is_open") and self.api.positions():
                from .broker import Broker
                Broker(self.cfg, self.api, self.reg, self.log).flatten_all("bot stopped by user")
        except Exception as e:
            self.log(f"! could not close positions on stop: {e} - check your Alpaca dashboard")

    @staticmethod
    def _sleep_until(t):
        while True:
            left = (t - pd.Timestamp.now(tz="UTC")).total_seconds()
            if left <= 0:
                return
            time.sleep(min(left, 30))
