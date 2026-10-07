"""Intraday market data.

* Downloads 5-minute bars from Alpaca (SIP = all US exchanges, split/dividend adjusted)
  and builds 5/15/30/60-minute candles from them, regular trading hours only.
* Every candle knows its trading day, its position in the day, and whether it is the
  last candle of the day - that is what makes "always flat by the close" possible.
* Free Alpaca plan: SIP data is delayed 15 minutes, so the most recent bars are
  topped up from the free real-time IEX feed (volume rescaled to SIP size).
* Candles that have not closed yet are never used.
"""
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DATA_DIR, TF_MINUTES

NY = "America/New_York"
CACHE = DATA_DIR / "cache"


@dataclass
class Panel:
    tf: str
    index: pd.DatetimeIndex     # candle CLOSE time (UTC)
    symbols: list
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    day: np.ndarray             # trading-day number of each candle (0, 1, 2 ...)
    mins: np.ndarray            # minutes from the open to this candle's close
    eod: np.ndarray             # True on the last candle of each day
    day_dates: list             # calendar date (datetime.date) of each trading day
    day_len: np.ndarray         # session length in minutes for each day (390, or 210 on half days)
    benchmark: str = "QQQ"
    meta: dict = field(default_factory=dict)

    @property
    def T(self):
        return len(self.index)

    @property
    def n_days(self):
        return int(self.day[-1]) + 1 if self.T else 0

    @property
    def bars_per_day(self):
        return 390 / TF_MINUTES[self.tf]

    def upto(self, n):
        nd = int(self.day[n - 1]) + 1 if n else 0
        return Panel(self.tf, self.index[:n], self.symbols, self.o[:n], self.h[:n], self.l[:n],
                     self.c[:n], self.v[:n], self.day[:n], self.mins[:n], self.eod_upto(n),
                     self.day_dates[:nd], self.day_len[:nd], self.benchmark, self.meta)

    def eod_upto(self, n):
        e = self.eod[:n].copy()
        return e   # the last bar of a truncated panel is NOT treated as end of day

    def bench_col(self):
        return self.symbols.index(self.benchmark) if self.benchmark in self.symbols else 0


# ----------------------------------------------------------------------------
# building panels
# ----------------------------------------------------------------------------

def sessions_from_calendar(cal_rows):
    """Alpaca calendar rows -> DataFrame indexed by date with open/close UTC timestamps."""
    rows = []
    for r in cal_rows:
        d = pd.Timestamp(r["date"])
        o = pd.Timestamp(f"{r['date']} {r['open']}").tz_localize(NY).tz_convert("UTC")
        c = pd.Timestamp(f"{r['date']} {r['close']}").tz_localize(NY).tz_convert("UTC")
        rows.append((d.date(), o, c))
    return pd.DataFrame(rows, columns=["date", "open", "close"]).set_index("date").sort_index()


def build_panel(frames, sessions, tf, benchmark, now=None, base_minutes=5, log=print):
    """frames: {symbol: DataFrame[open,high,low,close,volume] indexed by bar START (UTC), base_minutes bars}."""
    step = TF_MINUTES[tf]
    if step % base_minutes:
        raise ValueError(f"{tf} is not a multiple of the {base_minutes}-minute base bars")
    frames = {s: f for s, f in frames.items() if f is not None and len(f)}
    if not frames:
        raise RuntimeError("no price data")
    # --- keep only regular-session bars, and the full grid of session timestamps
    first = {s: f.index[0] for s, f in frames.items()}
    earliest = min(first.values())
    dropped = [s for s, t in first.items() if t > earliest + pd.Timedelta(days=365)]
    for s in dropped:
        frames.pop(s)
    start = max(f.index[0] for f in frames.values())
    sess = sessions[(sessions["open"] >= start.floor("D") - pd.Timedelta(days=1))]
    if now is not None:
        sess = sess[sess["open"] <= now]
    grid = []
    for d, row in sess.iterrows():
        g = pd.date_range(row["open"], row["close"] - pd.Timedelta(minutes=base_minutes),
                          freq=f"{base_minutes}min")
        grid.append(g)
    if not grid:
        raise RuntimeError("no trading sessions in range")
    grid = grid[0].append(grid[1:]) if len(grid) > 1 else grid[0]
    grid = grid[grid >= start]
    symbols = list(frames)
    T0 = len(grid)
    fld = {k: np.full((T0, len(symbols)), np.nan) for k in ["open", "high", "low", "close", "volume"]}
    for j, s in enumerate(symbols):
        f = frames[s].reindex(grid)
        for k in fld:
            fld[k][:, j] = f[k].values
    # bars with no trades: flat candle at previous close, zero volume
    close = pd.DataFrame(fld["close"]).ffill().values
    miss = np.isnan(fld["close"])
    for k in ["open", "high", "low", "close"]:
        fld[k][miss] = close[miss]
    fld["volume"][np.isnan(fld["volume"])] = 0.0
    ok = ~np.isnan(fld["close"]).any(axis=1)
    grid = grid[ok]
    for k in fld:
        fld[k] = fld[k][ok]
    # --- drop bars that have not finished yet
    if now is not None:
        done = grid + pd.Timedelta(minutes=base_minutes) <= now
        grid = grid[done]
        for k in fld:
            fld[k] = fld[k][done]
    # --- session bookkeeping on the base grid
    ny = grid.tz_convert(NY)
    dates = ny.date
    sess_open = sessions["open"].reindex(dates).values
    sess_close = sessions["close"].reindex(dates).values
    since_open = (grid.values - sess_open).astype("timedelta64[m]").astype(int)
    bucket = since_open // step
    # --- aggregate to the requested candle size (segments are contiguous in time)
    grp = pd.Series(np.arange(len(grid))).groupby([pd.Index(dates), pd.Index(bucket)], sort=False)
    fi = grp.min().values
    la = grp.max().values
    order = np.argsort(fi)
    fi, la = fi[order], la[order]
    O = fld["open"][fi]
    C = fld["close"][la]
    H = np.maximum.reduceat(fld["high"], fi, axis=0)
    L = np.minimum.reduceat(fld["low"], fi, axis=0)
    V = np.add.reduceat(fld["volume"], fi, axis=0)
    bar_start = grid[fi]
    bar_close = bar_start + pd.Timedelta(minutes=step)
    sc = pd.DatetimeIndex(sess_close[fi]).tz_localize("UTC") if pd.DatetimeIndex(sess_close[fi]).tz is None \
        else pd.DatetimeIndex(sess_close[fi])
    bar_close = pd.DatetimeIndex(np.minimum(bar_close.values, sc.values)).tz_localize("UTC")
    # a candle is complete only when its last base bar is the final one of its slot,
    # or the session ended: drop an incomplete last candle
    last_base_close = grid[la] + pd.Timedelta(minutes=base_minutes)
    complete = (last_base_close >= bar_close)
    if now is not None:
        complete &= bar_close <= now
    keep = np.asarray(complete)
    O, H, L, C, V = O[keep], H[keep], L[keep], C[keep], V[keep]
    bar_close = bar_close[keep]
    bdates = np.array(dates)[fi][keep]
    so = pd.DatetimeIndex(sess_open[fi][keep]).tz_localize("UTC") if pd.DatetimeIndex(sess_open[fi][keep]).tz is None \
        else pd.DatetimeIndex(sess_open[fi][keep])
    mins = ((bar_close - so).total_seconds() / 60).astype(int).values
    # trading-day numbering
    uniq, day = np.unique(bdates, return_inverse=True)
    day_len_map = ((sessions["close"] - sessions["open"]).dt.total_seconds() / 60).astype(int)
    day_len = np.array([int(day_len_map.loc[d]) for d in uniq])
    eod = np.zeros(len(day), dtype=bool)
    if len(day):
        eod[:-1] = day[1:] != day[:-1]
        # the very last candle is end-of-day only if it closes at the session close
        eod[-1] = mins[-1] >= day_len[day[-1]]
    if dropped:
        log(f"  note: not enough history, left out: {dropped}")
    return Panel(tf, bar_close, symbols, O, H, L, C, V, day.astype(np.int64), mins.astype(np.int64),
                 eod, list(uniq), day_len, benchmark if benchmark in symbols else symbols[0],
                 meta={"dropped": dropped})


# ----------------------------------------------------------------------------
# provider
# ----------------------------------------------------------------------------

class AlpacaProvider:
    BASE = "5Min"

    def __init__(self, cfg, api, log=print):
        self.cfg = cfg
        self.api = api
        self.log = log
        self.symbols = cfg["market"]["symbols"]
        self.bench = cfg["market"]["benchmark"]
        self.paid = str(cfg["data"].get("plan", "free")).lower() == "paid"
        self._frames = None
        self._topup = {}
        self._sessions = None
        self._panels = {}

    def now(self):
        return pd.Timestamp.now(tz="UTC")

    # ---------- calendar ----------
    def sessions(self, force=False):
        f = CACHE / "calendar.pkl"
        if self._sessions is not None and not force:
            return self._sessions
        if f.exists() and not force:
            blob = pd.read_pickle(f)
            if time.time() - blob["time"] < 12 * 3600:
                self._sessions = blob["sessions"]
                return self._sessions
        start = self.cfg["market"]["history_start"]
        end = (pd.Timestamp.now() + pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        rows = self.api.calendar(start, end)
        self._sessions = sessions_from_calendar(rows)
        pd.to_pickle({"time": time.time(), "sessions": self._sessions}, f)
        return self._sessions

    # ---------- bars ----------
    def _cache_file(self):
        return CACHE / "bars_5Min.pkl"

    def refresh(self, force=False):
        self._panels = {}
        now = self.now()
        sip_end = now if self.paid else now - pd.Timedelta(minutes=16)
        f = self._cache_file()
        frames = None
        if f.exists() and not force:
            frames = pd.read_pickle(f)["frames"]
        if frames is None or set(self.symbols) - set(frames):
            self.log(f"  downloading 5-minute history for {len(self.symbols)} symbols since "
                     f"{self.cfg['market']['history_start']} (first time only, a few minutes) ...")
            frames = self.api.bars(self.symbols, self.BASE, self.cfg["market"]["history_start"], sip_end)
        else:
            last = min(fr.index[-1] for fr in frames.values())
            new = self.api.bars(self.symbols, self.BASE, last - pd.Timedelta(days=5), sip_end)
            redo = []
            for s, nf in new.items():
                old = frames.get(s)
                if old is None:
                    redo.append(s)
                    continue
                common = old.index.intersection(nf.index)
                if len(common):
                    diff = np.abs(old.loc[common, "close"].values / nf.loc[common, "close"].values - 1).max()
                    if diff > 0.005:   # split or dividend re-adjustment -> refetch that symbol
                        redo.append(s)
                        continue
                frames[s] = pd.concat([old[old.index < nf.index[0]], nf])
            if redo:
                self.log(f"  prices re-adjusted (split/dividend) for {redo} - refreshing their history")
                frames.update(self.api.bars(redo, self.BASE, self.cfg["market"]["history_start"], sip_end))
        pd.to_pickle({"time": time.time(), "frames": frames}, f)
        self._frames = frames
        self._topup = {}
        if not self.paid:
            self._topup = self._iex_topup(frames, now)

    def _iex_topup(self, frames, now):
        """Latest bars from the free real-time IEX feed, volume rescaled to full-market size."""
        try:
            start = now - pd.Timedelta(days=4)
            iex = self.api.bars(self.symbols, self.BASE, start, now, feed="iex")
        except Exception as e:
            self.log(f"  ! live IEX top-up failed ({e}); using 15-minute delayed data")
            return {}
        out = {}
        for s, df in iex.items():
            sip = frames.get(s)
            if sip is None or not len(df):
                continue
            common = sip.index.intersection(df.index)
            ratio = np.nan
            if len(common) >= 20:
                a = sip.loc[common, "volume"].values
                b = df.loc[common, "volume"].values
                m = b > 0
                if m.sum() >= 10:
                    ratio = float(np.median(a[m] / b[m]))
            newer = df[df.index > sip.index[-1]].copy()
            if len(newer):
                out[s] = (newer, ratio)
        ratios = [r for _, r in out.values() if np.isfinite(r)]
        fallback = float(np.median(ratios)) if ratios else 30.0
        res = {}
        for s, (newer, r) in out.items():
            newer["volume"] = newer["volume"] * (r if np.isfinite(r) else fallback)
            res[s] = newer
        return res

    def panel(self, tf):
        if tf in self._panels:
            return self._panels[tf]
        if self._frames is None:
            f = self._cache_file()
            if not f.exists():
                self.refresh()
            else:
                self._frames = pd.read_pickle(f)["frames"]
        frames = {}
        for s, fr in self._frames.items():
            if s not in self.symbols:
                continue
            t = self._topup.get(s)
            frames[s] = pd.concat([fr, t]) if t is not None and len(t) else fr
        p = build_panel(frames, self.sessions(), tf, self.bench, now=self.now(), log=self.log)
        self._panels[tf] = p
        return p
