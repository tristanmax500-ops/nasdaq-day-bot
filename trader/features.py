"""Everything a strategy can look at. All values are normalised so one rule works on
every stock and candle size, and each value uses only information available when
the candle closed.
"""
import numpy as np
import pandas as pd

# name: (parameter choices, low, high, plain-English label)
FEATURES = {
    # price action
    "rsi":      ([2, 3, 5, 7, 14], 5.0, 95.0, "RSI({p})"),
    "zscore":   ([10, 20, 50, 100], -3.0, 3.0, "{p}-candle z-score of price"),
    "mom":      ([1, 2, 3, 6, 12, 24], -3.0, 3.0, "{p}-candle momentum (vol-adjusted)"),
    "trend":    ([20, 50, 100, 200], -6.0, 6.0, "distance from {p}-candle average (ATRs)"),
    "macd":     ([6, 12, 24], -3.0, 3.0, "MACD {p}/{p4} (ATRs)"),
    "stoch":    ([5, 10, 20, 50], 0.0, 100.0, "{p}-candle stochastic %K"),
    "ibs":      ([1], 0.0, 1.0, "close position inside the candle"),
    "volz":     ([1], -2.0, 3.0, "relative volume vs normal for this time of day (log)"),
    "volat":    ([12, 24], 0.4, 2.5, "{p}-candle volatility vs 20-day normal"),
    # intraday structure
    "tod":      ([1], 0.0, 390.0, "minutes since the open"),
    "vwap":     ([1], -4.0, 4.0, "distance from VWAP (ATRs)"),
    "orb_hi":   ([15, 30, 60], -4.0, 4.0, "distance above the first-{p}-minute high (ATRs)"),
    "orb_lo":   ([15, 30, 60], -4.0, 4.0, "distance above the first-{p}-minute low (ATRs)"),
    "dchg":     ([1], -3.0, 3.0, "move since today's open (daily-vol units)"),
    "gap":      ([1], -3.0, 3.0, "today's opening gap (daily-vol units)"),
    "pdh":      ([1], -6.0, 6.0, "distance from yesterday's high (ATRs)"),
    "pdl":      ([1], -6.0, 6.0, "distance from yesterday's low (ATRs)"),
    "mkt":      ([1], -3.0, 3.0, "QQQ move since today's open (daily-vol units)"),
    "mkt_vwap": ([1], -4.0, 4.0, "QQQ distance from its VWAP (ATRs)"),
    # ICT / smart-money style price action (from candles; real order-book data isn't available)
    "sweep_lo": ([1, 12, 48], 0.0, 0.6, "liquidity sweep: dipped below {sweep_ref} and closed back above (depth, ATRs)"),
    "sweep_hi": ([1, 12, 48], 0.0, 0.6, "liquidity sweep: poked above {sweep_ref} and closed back below (depth, ATRs)"),
    "fvg_bull": ([6, 12, 24], -3.0, 4.0, "distance above the latest bullish fair-value gap (last {p} candles, ATRs)"),
    "fvg_bear": ([6, 12, 24], -4.0, 3.0, "distance below the latest bearish fair-value gap (last {p} candles, ATRs)"),
    "swing_hi": ([12, 24, 48], -6.0, 1.5, "break of structure: distance above the {p}-candle swing high (ATRs)"),
    "swing_lo": ([12, 24, 48], -1.5, 6.0, "distance above the {p}-candle swing low (ATRs)"),
    "body":     ([1, 3], -3.0, 3.0, "displacement: candle body over the last {p} candle(s) (ATRs, + = up)"),
    "dloc":     ([1], 0.0, 1.0, "where price is in today's range (0 = day low / discount, 1 = day high / premium)"),
    "flow":     ([6, 12, 24], -1.0, 1.0, "order-flow proxy: buying vs selling pressure over {p} candles (-1..+1)"),
    # the day and the week
    "pd_ret":   ([1], -3.0, 3.0, "yesterday's move, open to close (daily-vol units)"),
    "vol_reg":  ([5, 20], 0.5, 2.0, "volatility regime: last {p} days' range vs the 60-day normal (calm < 0.8, wild > 1.3)"),
    "wk_trend": ([5, 20], -3.0, 3.0, "trend of the last {p} days (vol units, + = up)"),
    # scheduled events (0 = no, 1 = yes)
    "macro":    ([1], 0.0, 1.0, "CPI/jobs/PPI release day (0/1)"),
    "fomc":     ([1], 0.0, 1.0, "Fed decision day (0/1)"),
    "earn":     ([1], 0.0, 1.0, "earnings reaction day (0/1)"),
    "earn_next": ([1], 0.0, 1.0, "earnings due tonight/tomorrow morning (0/1)"),
    # news
    "sent":     ([1], -1.0, 1.0, "news sentiment, last 24h"),
    "newsv":    ([1], -2.0, 3.0, "news buzz vs normal"),
    "mkt_sent": ([1], -1.0, 1.0, "market-wide news sentiment, last 24h"),
}
EXTERNAL = {"macro", "fomc", "earn", "earn_next", "sent", "newsv", "mkt_sent"}


def feature_label(name, p):
    ref = "yesterday's low" if name == "sweep_lo" else "yesterday's high"
    if name in ("sweep_lo", "sweep_hi") and p != 1:
        ref = f"the {p}-candle swing {'low' if name == 'sweep_lo' else 'high'}"
    return FEATURES[name][3].format(p=p, p4=p * 4, sweep_ref=ref)


class FeatureBook:
    """Lazily computes and caches features for one Panel."""

    def __init__(self, panel, extra=None):
        self.p = panel
        self.extra = extra or {}
        self.cols = list(range(len(panel.symbols)))
        self.cache = {}
        df = lambda a: pd.DataFrame(a, columns=self.cols)
        self.C, self.H, self.L, self.O, self.V = (df(x) for x in (panel.c, panel.h, panel.l, panel.o, panel.v))
        self.logret = np.log(self.C).diff()
        pc = self.C.shift(1)
        tr = np.fmax.reduce([(self.H - self.L).values, (self.H - pc).abs().values, (self.L - pc).abs().values])
        self.atr = np.array(df(tr).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().values,
                            dtype=np.float64, copy=True)
        # session helpers
        day = panel.day
        T = panel.T
        starts = np.r_[0, np.flatnonzero(np.diff(day)) + 1] if T else np.array([], dtype=int)
        self.day_start = starts
        first_of = np.repeat(starts, np.diff(np.r_[starts, T])) if T else np.array([], dtype=int)
        self.bod = np.arange(T) - first_of
        self.first_of = first_of
        ends = np.r_[starts[1:] - 1, T - 1] if T else np.array([], dtype=int)
        self.d_open = panel.o[starts]                       # [days, S]
        self.d_close = panel.c[ends]
        self.d_high = np.maximum.reduceat(panel.h, starts, axis=0) if T else np.zeros((0, len(self.cols)))
        self.d_low = np.minimum.reduceat(panel.l, starts, axis=0) if T else np.zeros((0, len(self.cols)))
        # daily volatility of open->close moves, prior 20 days only
        oc = np.log(self.d_close / self.d_open)
        prev_c = np.vstack([np.full((1, len(self.cols)), np.nan), self.d_close[:-1]])
        gap = np.log(self.d_open / prev_c)
        dvol = pd.DataFrame(oc).rolling(20, min_periods=10).std().shift(1).values
        self.dvol = np.where(np.isfinite(dvol) & (dvol > 0), dvol, np.nan)
        self.gap_d = gap
        self.prev_c = prev_c

    def get(self, name, p):
        key = (name, p)
        if key not in self.cache:
            a = np.array(self._compute(name, p), dtype=np.float32, copy=True)   # float32 halves memory
            a[~np.isfinite(a)] = np.nan
            self.cache[key] = a
        return self.cache[key]

    def _bcast_day(self, arr_days):
        return arr_days[self.p.day]

    def _compute(self, name, p):
        C, H, L = self.C, self.H, self.L
        atr = self.atr
        P = self.p
        if name in EXTERNAL:
            a = self.extra.get(name)
            return a if a is not None else np.full((P.T, len(self.cols)), np.nan)
        if name == "rsi":
            d = C.diff()
            up = d.clip(lower=0).ewm(alpha=1 / p, adjust=False, min_periods=p).mean()
            dn = (-d.clip(upper=0)).ewm(alpha=1 / p, adjust=False, min_periods=p).mean()
            r = 100 - 100 / (1 + up / dn.replace(0, np.nan))
            return r.where(dn != 0, 100.0).where(up.notna()).values
        if name == "zscore":
            return ((C - C.rolling(p).mean()) / C.rolling(p).std()).values
        if name == "mom":
            vol = self.logret.rolling(100, min_periods=50).std()
            return (np.log(C / C.shift(p)) / (vol * np.sqrt(p))).values
        if name == "trend":
            return (C - C.rolling(p).mean()).values / atr
        if name == "macd":
            f = C.ewm(span=p, adjust=False, min_periods=p).mean()
            s = C.ewm(span=p * 4, adjust=False, min_periods=p * 4).mean()
            return (f - s).values / atr
        if name == "stoch":
            ll, hh = L.rolling(p).min(), H.rolling(p).max()
            return (100 * (C - ll) / (hh - ll).replace(0, np.nan)).values
        if name == "ibs":
            return ((C - L) / (H - L).replace(0, np.nan)).values
        if name == "volz":
            lv = np.log1p(self.V)
            base = lv.groupby(self.bod).transform(lambda x: x.rolling(20, min_periods=5).mean().shift(1))
            return (lv - base).values
        if name == "volat":
            short = self.logret.rolling(p).std()
            long = self.logret.rolling(int(20 * P.bars_per_day), min_periods=int(5 * P.bars_per_day)).std()
            return (short / long).values
        if name == "tod":
            return np.repeat(P.mins[:, None].astype(float), len(self.cols), axis=1)
        if name == "vwap":
            tp = ((self.H + self.L + self.C) / 3).values
            v = P.v
            g = P.day
            cum_pv = pd.DataFrame(tp * v).groupby(g).cumsum().values
            cum_v = pd.DataFrame(v).groupby(g).cumsum().values
            vw = np.where(cum_v > 0, cum_pv / np.where(cum_v > 0, cum_v, 1), np.nan)
            return (P.c - vw) / atr
        if name in ("orb_hi", "orb_lo"):
            in_or = P.mins <= p                    # candles closing within the first p minutes
            hh = pd.DataFrame(np.where(in_or[:, None], P.h, np.nan)).groupby(P.day).transform("max").values
            ll = pd.DataFrame(np.where(in_or[:, None], P.l, np.nan)).groupby(P.day).transform("min").values
            ref = hh if name == "orb_hi" else ll
            out = (P.c - ref) / atr
            out[P.mins < p] = np.nan               # range not finished yet
            return out
        if name == "dchg":
            return np.log(P.c / self._bcast_day(self.d_open)) / self._bcast_day(self.dvol)
        if name == "gap":
            return self._bcast_day(self.gap_d / self.dvol)
        if name in ("pdh", "pdl"):
            src = self.d_high if name == "pdh" else self.d_low
            prev = np.vstack([np.full((1, len(self.cols)), np.nan), src[:-1]])
            return (P.c - self._bcast_day(prev)) / atr
        if name in ("sweep_lo", "sweep_hi"):
            lo = name == "sweep_lo"
            if p == 1:      # yesterday's low / high
                src = self.d_low if lo else self.d_high
                ref = self._bcast_day(np.vstack([np.full((1, len(self.cols)), np.nan), src[:-1]]))
            else:           # swing low / high of the previous p candles
                ref = (L.rolling(p).min() if lo else H.rolling(p).max()).shift(1).values
            if lo:
                depth = (ref - P.l) / atr
                hit = (P.l < ref) & (P.c > ref)
            else:
                depth = (P.h - ref) / atr
                hit = (P.h > ref) & (P.c < ref)
            out = np.where(hit, depth, 0.0)
            return np.where(np.isfinite(ref) & np.isfinite(atr), out, np.nan)
        if name in ("fvg_bull", "fvg_bear"):
            h2 = np.vstack([np.full((2, len(self.cols)), np.nan), P.h[:-2]])
            l2 = np.vstack([np.full((2, len(self.cols)), np.nan), P.l[:-2]])
            same_day = np.r_[False, False, (P.day[2:] == P.day[:-2])][:, None]
            if name == "fvg_bull":      # this candle's low is above the high two candles back
                gap = np.where((P.l > h2) & same_day, h2, np.nan)            # gap bottom = old high
            else:                       # this candle's high is below the low two candles back
                gap = np.where((P.h < l2) & same_day, l2, np.nan)            # gap top = old low
            lvl = pd.DataFrame(gap).groupby(P.day).ffill(limit=p).values
            return (P.c - lvl) / atr
        if name in ("swing_hi", "swing_lo"):
            ref = (H.rolling(p).max() if name == "swing_hi" else L.rolling(p).min()).shift(1).values
            return (P.c - ref) / atr
        if name == "body":
            b = (P.c - P.o) if p == 1 else (C - pd.DataFrame(P.o, columns=self.cols).shift(p - 1)).values
            return b / atr
        if name == "dloc":
            hh = pd.DataFrame(P.h).groupby(P.day).cummax().values
            ll = pd.DataFrame(P.l).groupby(P.day).cummin().values
            rng_ = hh - ll
            return np.where(rng_ > 0, (P.c - ll) / np.where(rng_ > 0, rng_, 1), np.nan)
        if name == "flow":
            rng_ = (P.h - P.l)
            clv = np.where(rng_ > 0, ((P.c - P.l) - (P.h - P.c)) / np.where(rng_ > 0, rng_, 1), 0.0)
            mv = pd.DataFrame(clv * P.v).rolling(p).sum().values
            vv = pd.DataFrame(P.v).rolling(p).sum().values
            return np.where(vv > 0, mv / np.where(vv > 0, vv, 1), np.nan)
        if name == "pd_ret":
            oc = np.log(self.d_close / self.d_open)
            prev = np.vstack([np.full((1, len(self.cols)), np.nan), oc[:-1]])
            return self._bcast_day(prev / self.dvol)
        if name == "vol_reg":
            rng_d = pd.DataFrame(np.log(self.d_high / self.d_low))
            ratio = (rng_d.rolling(p).mean() / rng_d.rolling(60, min_periods=30).mean()).shift(1).values
            return self._bcast_day(ratio)
        if name == "wk_trend":
            lc = pd.DataFrame(np.log(self.d_close))
            ret = (lc - lc.shift(p)).shift(1).values          # up to yesterday's close
            return self._bcast_day(ret / (self.dvol * np.sqrt(p)))
        if name == "mkt":
            b = P.bench_col()
            return np.repeat(self.get("dchg", 1)[:, b:b + 1], len(self.cols), axis=1)
        if name == "mkt_vwap":
            b = P.bench_col()
            return np.repeat(self.get("vwap", 1)[:, b:b + 1], len(self.cols), axis=1)
        raise KeyError(name)

    # regimes used by the self-healing diagnosis
    def regimes(self, t, s):
        g = lambda n, p=1: self.get(n, p)[t, s]
        out = {
            "day": "up" if g("dchg") > 0 else "down",
            "vol": "high" if g("volat", 24) > 1.3 else ("low" if g("volat", 24) < 0.8 else "normal"),
            "market": "up" if g("mkt") > 0 else "down",
            "time": "first hour" if g("tod") <= 60 else ("last 90 min" if g("tod") >= 300 else "midday"),
        }
        mac, fo = g("macro"), g("fomc")
        if np.isfinite(mac) and np.isfinite(fo):
            out["event"] = "event day" if (mac > 0.5 or fo > 0.5) else "normal day"
        e = g("earn")
        if np.isfinite(e):
            out["earnings"] = "earnings day" if e > 0.5 else "no earnings"
        se = g("sent")
        if np.isfinite(se):
            out["news"] = "negative" if se < -0.3 else ("positive" if se > 0.3 else "neutral")
        return out
