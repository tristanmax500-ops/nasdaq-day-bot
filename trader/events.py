"""Important scheduled events: Fed decisions, CPI, jobs report, PPI, and earnings.

Sources
  * FRED (St. Louis Fed) release calendar - official past AND upcoming dates.
      release 10  = Consumer Price Index (8:30 am ET)
      release 50  = Employment Situation / jobs report (8:30 am ET)
      release 46  = Producer Price Index (8:30 am ET)
  * Fed decision dates 2021-2027 from the Fed's official meeting calendar
    (federalreserve.gov/monetarypolicy/fomccalendars.htm).
  * Earnings dates: Yahoo Finance.

Features per candle (known before the open, so no peeking):
  macro      1 on CPI / jobs report / PPI days
  fomc       1 on Fed decision days
  earn       1 on the first session that reacts to this company's earnings
             (for QQQ: any heavyweight's reaction day)
  earn_next  1 if the company reports after today's close or before tomorrow's open
"""
import datetime as dt
import json
import time

import numpy as np
import pandas as pd
import requests

from .config import DATA_DIR

# NOTE: FRED's "FOMC Press Release" release (101) lists almost every day (it tracks the
# rate series), so Fed decision days come from the Fed's own meeting calendar below.
FRED_RELEASES = {10: ("cpi", "Consumer Price Index"),
                 50: ("nfp", "Employment Situation"), 46: ("ppi", "Producer Price Index")}

# Decision day (2nd day of each meeting), federalreserve.gov/monetarypolicy/fomccalendars.htm
FOMC_BUILTIN = [
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09", "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08",
]

CACHE_FILE = DATA_DIR / "cache" / "events.json"


class EventCalendar:
    def __init__(self, cfg, log=print):
        self.cfg = cfg
        self.log = log
        self.macro = {}            # kind -> set of dates
        self.earnings = {}         # symbol -> list of (date, timing) timing in {"amc","bmo","unknown"}
        self.have_fred = False
        self.have_earnings = False
        self.updated = 0

    # ------------------------------------------------------------------
    def load(self):
        if CACHE_FILE.exists():
            blob = json.loads(CACHE_FILE.read_text())
            self.macro = {k: set(dt.date.fromisoformat(d) for d in v) for k, v in blob["macro"].items()}
            self.earnings = {s: [(dt.date.fromisoformat(d), tm) for d, tm in v] for s, v in blob["earnings"].items()}
            self.have_fred = blob.get("have_fred", False)
            self.have_earnings = blob.get("have_earnings", False)
            self.updated = blob.get("time", 0)
        # Fed decision days always come from the official meeting calendar (never from FRED)
        self.macro["fomc"] = {dt.date.fromisoformat(d) for d in FOMC_BUILTIN}
        return self

    def save(self):
        CACHE_FILE.write_text(json.dumps({
            "time": time.time(), "have_fred": self.have_fred, "have_earnings": self.have_earnings,
            "macro": {k: sorted(d.isoformat() for d in v) for k, v in self.macro.items()},
            "earnings": {s: [(d.isoformat(), tm) for d, tm in v] for s, v in self.earnings.items()},
        }))

    def refresh(self, force=False):
        self.load()
        if not force and time.time() - self.updated < 12 * 3600:
            return self
        key = self.cfg["account"].get("fred_api_key")
        if key:
            try:
                self._fred(key)
                self.have_fred = True
            except Exception as e:
                self.log(f"  ! FRED calendar download failed: {e}")
        else:
            self.log("  note: no FRED key - the bot can't learn from CPI/jobs/Fed days "
                     "(the Fed-day safety blackout still works)")
        try:
            self._earnings()
        except Exception as e:
            self.log(f"  ! earnings dates download failed: {e}")
        self.updated = time.time()
        self.save()
        return self

    def _fred(self, key):
        base = "https://api.stlouisfed.org/fred"
        for rid, (kind, name) in FRED_RELEASES.items():
            info = requests.get(f"{base}/release", params={"release_id": rid, "api_key": key,
                                                           "file_type": "json"}, timeout=30)
            info.raise_for_status()
            rel_name = info.json()["releases"][0]["name"]
            if name.lower() not in rel_name.lower():
                self.log(f"  ! FRED release {rid} is '{rel_name}', expected '{name}' - skipped")
                continue
            r = requests.get(f"{base}/release/dates", params={
                "release_id": rid, "api_key": key, "file_type": "json", "realtime_start": "2000-01-01",
                "realtime_end": "9999-12-31", "include_release_dates_with_no_data": "true",
                "sort_order": "asc", "limit": 10000}, timeout=30)
            r.raise_for_status()
            dates = {dt.date.fromisoformat(x["date"]) for x in r.json().get("release_dates", [])}
            years = max(1, len({d.year for d in dates}))
            if len(dates) / years > 20:          # monthly releases: ~12 a year. Anything far above is not a release calendar
                self.log(f"  ! FRED {kind.upper()} has {len(dates)} dates ({len(dates)/years:.0f}/year) - ignored as unreliable")
                continue
            self.macro[kind] = dates
            self.log(f"  FRED: {len(dates)} {kind.upper()} dates")

    def _earnings(self):
        import yfinance as yf
        from .config import watch_symbols
        syms = [s for s in watch_symbols(self.cfg) if s != self.cfg["market"]["benchmark"]]
        got = 0
        for s in syms:
            try:
                df = yf.Ticker(s).get_earnings_dates(limit=100)
            except Exception:
                df = None
            if df is None or len(df) == 0:
                continue
            rows = []
            for ts in df.index:
                ts = pd.Timestamp(ts)
                if ts.tzinfo is not None:
                    ts = ts.tz_convert("America/New_York")
                h = ts.hour
                timing = "amc" if h >= 12 else ("bmo" if 0 < h < 10 else "unknown")
                rows.append((ts.date(), timing))
            self.earnings[s] = sorted(set(rows))
            got += 1
        self.have_earnings = got > 0
        self.log(f"  earnings dates: {got}/{len(syms)} companies")

    # ------------------------------------------------------------------
    def fomc_dates(self):
        return self.macro.get("fomc", set())

    def features(self, panel):
        """Per-candle event flags. Missing sources -> NaN (strategies can't use them)."""
        T, S = panel.T, len(panel.symbols)
        dates = panel.day_dates
        nd = len(dates)
        out = {}
        if self.have_fred and any(self.macro.get(k) for k in ("cpi", "nfp", "ppi")):
            macro_days = self.macro.get("cpi", set()) | self.macro.get("nfp", set()) | self.macro.get("ppi", set())
            md = np.array([1.0 if d in macro_days else 0.0 for d in dates])
            out["macro"] = np.repeat(md[panel.day][:, None], S, axis=1)
        else:
            out["macro"] = np.full((T, S), np.nan)
        # Fed days: known from the official calendar for 2021-2027; outside that range -> unknown
        fomc = self.macro.get("fomc", set())
        lo, hi = (min(fomc), max(fomc)) if fomc else (None, None)
        fd = np.array([np.nan if (lo is None or d < dt.date(lo.year, 1, 1) or d > dt.date(hi.year, 12, 31))
                       else (1.0 if d in fomc else 0.0) for d in dates])
        out["fomc"] = np.repeat(fd[panel.day][:, None], S, axis=1)
        if not self.have_earnings:
            out["earn"] = np.full((T, S), np.nan)
            out["earn_next"] = np.full((T, S), np.nan)
            return out
        # next trading day for each trading day (last one: next weekday)
        nxt = list(dates[1:]) + [_next_weekday(dates[-1])] if nd else []
        heavy = set(self.cfg["market"].get("qqq_heavyweights", []))
        bench = self.cfg["market"]["benchmark"]
        e_day = np.zeros((nd, S))
        e_next = np.zeros((nd, S))
        react = {}
        soon = {}
        for s, rows in self.earnings.items():
            r, n = set(), set()
            for d, timing in rows:
                if timing == "bmo":
                    r.add(d)
                elif timing == "amc":
                    r.add(_next_trading(d, dates))
                else:
                    r.add(d)
                    r.add(_next_trading(d, dates))
                if timing in ("amc", "unknown"):
                    n.add(d)                       # reports after today's close
                if timing in ("bmo", "unknown"):
                    n.add(("prev", d))             # reports before the next open
            react[s] = r
            soon[s] = n
        idx = {d: i for i, d in enumerate(dates)}
        prev_idx = {d: i for i, d in enumerate(nxt)}
        for j, s in enumerate(panel.symbols):
            members = list(heavy) if s == bench else [s]
            for m in members:
                for d in react.get(m, ()):
                    if d in idx:
                        e_day[idx[d], j] = 1.0
                for x in soon.get(m, ()):
                    if isinstance(x, tuple):        # BMO on day x[1] -> flag the trading day before it
                        i = prev_idx.get(x[1])
                        if i is not None:
                            e_next[i, j] = 1.0
                    elif x in idx:
                        e_next[idx[x], j] = 1.0
        out["earn"] = e_day[panel.day]
        out["earn_next"] = e_next[panel.day]
        return out


def _next_weekday(d):
    d = d + dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


def _next_trading(d, trading_dates):
    """First trading date strictly after d (falls back to next weekday)."""
    i = np.searchsorted(np.array(trading_dates, dtype="datetime64[D]"), np.datetime64(d), side="right")
    if i < len(trading_dates):
        return trading_dates[i]
    return _next_weekday(d)
