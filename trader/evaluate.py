"""Scoring strategies and the anti-overfitting gauntlet (on DAILY account P&L).

Testing thousands of strategies guarantees some look brilliant by luck. Each
check below kills a different kind of fluke:
  1 enough trades          5 beats the SAME strategy with random entry times
  2 walk-forward           6 adds value beyond QQQ's own intraday move
  3 works on most stocks   7 not luck (deflated Sharpe, adjusted for ideas tried)
  4 profit factor          8 robust to small changes of its numbers
                           9 one-time exam on recent data it has never seen
"""
import copy
import math

import numpy as np
from scipy import stats

from . import backtest as bt
from .features import FEATURES, FeatureBook
from .genome import _round, BINARY

WARMUP_DAYS = 25


class EvalContext:
    def __init__(self, panels, cfg, extras=None, fomc_dates=(), research_frac=None, cost=None):
        self.cfg = cfg
        self.cost = cost if cost is not None else \
            (float(cfg["costs"]["slippage_bps"]) + float(cfg["costs"]["commission_bps"])) / 1e4
        extras = extras or {}
        self.books = {tf: FeatureBook(p, extras.get(tf)) for tf, p in panels.items()}
        self.rules = {tf: bt.Rules(cfg, p, fomc_dates) for tf, p in panels.items()}
        hf = cfg["discovery"]["holdout_fraction"] if research_frac is None else 1 - research_frac
        self.split = {}
        for tf, p in panels.items():
            nd = p.n_days
            d0 = min(WARMUP_DAYS, max(nd - 2, 0))
            d_end = d0 + int((nd - d0) * (1 - hf))
            self.split[tf] = (self.day_to_bar(p, d0), self.day_to_bar(p, d_end), p.T)
        self.folds = int(cfg["discovery"]["folds"])
        self.min_trades = int(cfg["discovery"]["min_trades"])
        # day trading: also demand a minimum number of trades per trading day (research + exam)
        tpd = float(cfg["discovery"].get("min_trades_per_day", 0) or 0)
        self.min_trades_holdout = 0
        if tpd > 0 and panels:
            tf0 = next(iter(panels))
            p0 = panels[tf0]
            s0, s1, sT = self.split[tf0]
            nd_r = int(p0.day[s1 - 1] - p0.day[s0] + 1) if s1 > s0 else 0
            nd_h = int(p0.day[sT - 1] - p0.day[s1] + 1) if sT > s1 else 0
            self.min_trades = max(self.min_trades, int(np.ceil(tpd * nd_r)))
            self.min_trades_holdout = int(np.ceil(0.75 * tpd * nd_h))   # a little slack for a quieter year
        self.recent_start = {}
        self._bench = {}
        self.min_folds = int(cfg["validation"]["min_folds_profitable"])
        self.early_exit = bool(cfg["discovery"].get("early_exit", True))
        self._cuts = {}

    def _stage1_end(self, tf, start, end):
        """First bar after the first half of the folds (on a day boundary), or None."""
        k = (tf, start, end)
        if k not in self._cuts:
            p = self.books[tf].p
            nd = int(p.day[end - 1] - p.day[start] + 1)
            half = self.folds // 2
            sizes = [len(x) for x in np.array_split(np.arange(nd), self.folds)]
            dcut = int(p.day[start]) + sum(sizes[:half])
            cut = self.day_to_bar(p, dcut)
            self._stage1_folds = half
            self._cuts[k] = cut if (half >= 1 and start + 50 < cut < end - 50) else None
        self._stage1_folds = self.folds // 2
        return self._cuts[k]

    @staticmethod
    def day_to_bar(p, d):
        if d >= p.n_days:
            return p.T
        return int(np.searchsorted(p.day, d, side="left"))

    def bench_daily(self, tf, start, end):
        """QQQ open->close return for each day (the 'just buy QQQ at the open' day trade)."""
        b = self.books[tf]
        j = b.p.bench_col()
        oc = b.d_close[:, j] / b.d_open[:, j] - 1
        d0, d1 = b.p.day[start], b.p.day[end - 1] + 1
        return oc[d0:d1]

    def run(self, g, start, end, entry_override=None):
        tf = g["tf"]
        rets, trades, final = bt.run(self.books[tf], g, self.cost, self.rules[tf], start, end, entry_override)
        daily = bt.daily_returns(rets, self.books[tf].p.day, start, end)
        return rets, trades, daily, final

    # ------------------------------------------------------------------
    def score(self, g, start=None, end=None, early_exit=None):
        tf = g["tf"]
        if tf not in self.books:
            return {"fitness": -9.0, "trades": 0, "sr_day": 0.0}
        s0, r_end, _ = self.split[tf]
        start = s0 if start is None else start
        end = r_end if end is None else end
        if end - start < 50:
            return {"fitness": -9.0, "trades": 0, "sr_day": 0.0}
        book = self.books[tf]
        entry, ex = bt.signals(book, g)
        # quick reject: a strategy can't trade more often than its entry signal fires
        n_sig = int(np.count_nonzero(entry[start:end]))
        if n_sig < self.min_trades * 0.5:
            return {"fitness": -5.0 + n_sig / max(self.min_trades, 1), "trades": 0, "sr_day": 0.0}
        early = early_exit if early_exit is not None else self.early_exit
        cut = self._stage1_end(tf, start, end) if early else None
        if cut is None:
            drets, trades, _ = bt.run(book, g, self.cost, self.rules[tf], start, end, lean=True, entry=entry, ex=ex)
        else:
            # Stage 1: test only the first half of the research years (folds 1-2). Every day is
            # independent (flat by the close), so stage 1 + stage 2 give exactly the full result.
            d1, t1, _ = bt.run(book, g, self.cost, self.rules[tf], start, cut, lean=True, entry=entry, ex=ex)
            f1 = np.array_split(d1.sum(axis=1), self._stage1_folds)
            fs1 = np.array([bt.sharpe(f) for f in f1])
            if int((fs1 <= 0).sum()) > self.folds - self.min_folds:
                # already too many losing periods: it can never pass the 'works in most periods'
                # check, so don't spend time on the second half
                fit = float(fs1.mean() - 0.5 * fs1.std()) - 1.0
                return {"fitness": min(fit, -0.01), "trades": int(len(t1)), "sr_day": 0.0,
                        "fold_sharpes": [round(float(x), 3) for x in fs1]}
            d2, t2, _ = bt.run(book, g, self.cost, self.rules[tf], cut, end, lean=True, entry=entry, ex=ex)
            drets = np.vstack([d1, d2])
            trades = np.vstack([t1, t2]) if len(t2) else t1
        daily = drets.sum(axis=1)
        n = len(trades)
        if n < self.min_trades * 0.5:
            return {"fitness": -5.0 + n / max(self.min_trades, 1), "trades": n, "sr_day": 0.0}
        folds = np.array_split(daily, self.folds)
        fs = np.array([bt.sharpe(f) for f in folds])
        sym = drets.sum(axis=0)
        traded = np.bincount(trades[:, 0].astype(int), minlength=drets.shape[1]) > 0
        sym_pos = float((sym[traded] > 0).mean()) if traded.any() else 0.0
        fit = fs.mean() - 0.5 * fs.std()
        fit *= min(1.0, n / self.min_trades) ** 0.5
        fit *= 0.5 + 0.5 * sym_pos
        rs = self.recent_start.get(tf)
        if rs is not None and start < rs < end - 50:
            d_rs = self.books[tf].p.day[rs] - self.books[tf].p.day[start]
            fit = 0.5 * fit + 0.5 * bt.sharpe(daily[d_rs:])
        sd = daily.std()
        return {"fitness": float(fit), "trades": int(n), "sr_day": float(daily.mean() / sd) if sd > 0 else 0.0,
                "fold_sharpes": [round(float(x), 3) for x in fs], "sym_pos": round(sym_pos, 3),
                "sharpe": round(bt.sharpe(daily), 3)}


# ----------------------------------------------------------------------------
def deflated_sharpe(daily, n_trials, sr_var_trials=None):
    """P(true Sharpe > 0) after accounting for the number of distinct ideas tried
    (Bailey & Lopez de Prado). Uses per-day Sharpe. The spread of Sharpe ratios that
    pure luck produces is the estimator's own sampling variance under 'no skill' (1/T),
    unless a different value is given."""
    r = np.asarray(daily)
    T = len(r)
    if T < 30 or r.std() == 0:
        return 0.0
    if sr_var_trials is None:
        sr_var_trials = 1.0 / T
    sr = r.mean() / r.std()
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    n = max(int(n_trials), 2)
    emc = 0.5772156649
    sr0 = math.sqrt(max(sr_var_trials, 1e-12)) * (
        (1 - emc) * stats.norm.ppf(1 - 1.0 / n) + emc * stats.norm.ppf(1 - 1.0 / (n * math.e)))
    den = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr * sr, 1e-9))
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(T - 1) / den))


def alpha_tstat(daily, bench):
    daily, bench = np.asarray(daily), np.asarray(bench)
    n = min(len(daily), len(bench))
    daily, bench = daily[:n], bench[:n]
    if n < 30 or daily.std() == 0:
        return 0.0
    X = np.column_stack([np.ones(n), bench])
    beta, *_ = np.linalg.lstsq(X, daily, rcond=None)
    resid = daily - X @ beta
    s2 = resid @ resid / (n - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    return float(beta[0] / math.sqrt(cov[0, 0])) if cov[0, 0] > 0 else 0.0


def random_entry_test(ctx, g, start, end, rng, n=24):
    """Daily Sharpe of copies that enter at RANDOM times (same exits, stops, sizing,
    trade count and time in market per stock)."""
    book = ctx.books[g["tf"]]
    rets, trades, _, _ = ctx.run(g, start, end)
    S = rets.shape[1]
    n_tr = np.zeros(S)
    held = np.zeros(S)
    for tr in trades:
        s = int(tr[0])
        n_tr[s] += 1
        held[s] += tr[2] - tr[1] + 1
    flat = np.maximum((end - start) - held, 1)
    freq = np.clip(n_tr / flat, 0, 1)
    out = []
    for _ in range(n):
        rnd = np.zeros((book.p.T, S), dtype=bool)
        rnd[start:end] = rng.random((end - start, S)) < freq[None, :]
        _, _, daily, _ = ctx.run(g, start, end, entry_override=rnd)
        out.append(bt.sharpe(daily))
    return np.array(out)


def perturb(g, rng):
    h = copy.deepcopy(g)
    for c in h["entry"] + ([h["exit"]] if h["exit"] else []):
        if c["f"] in BINARY:
            continue
        lo, hi = FEATURES[c["f"]][1], FEATURES[c["f"]][2]
        c["v"] = _round(c["v"] + rng.uniform(-0.05, 0.05) * (hi - lo), c["f"])
        if rng.random() < 0.25:
            ps = FEATURES[c["f"]][0]
            i = ps.index(c["p"]) if c["p"] in ps else 0
            c["p"] = ps[int(np.clip(i + rng.choice([-1, 1]), 0, len(ps) - 1))]
    for k in ("stop", "tp", "trail"):
        if h[k]:
            h[k] = float(round(h[k] * rng.uniform(0.8, 1.25), 2))
    if rng.random() < 0.3:
        h["hold"] = max(1, int(round(h["hold"] * rng.uniform(0.7, 1.4))))
    return h


def gauntlet(ctx, g, n_trials, sr_var, rng, start=None, end=None, use_holdout=True, probation_ok=False):
    """probation_ok: a strategy that passes everything EXCEPT the luck test still passes, marked
    rep["probation"] = True (it then paper-trades at reduced size until live results confirm it)."""
    v = ctx.cfg["validation"]
    tf = g["tf"]
    s0, r_end, T = ctx.split[tf]
    start = s0 if start is None else start
    end = r_end if end is None else end
    rep = {"checks": {}}
    sc = ctx.score(g, start, end, early_exit=False)
    rets, trades, daily, _ = ctx.run(g, start, end)
    rep["research"] = bt.summarize(daily, trades)
    rep["fold_sharpes"] = sc.get("fold_sharpes", [])

    def check(name, ok, detail):
        rep["checks"][name] = {"pass": bool(ok), "detail": detail}
        return ok

    if not check("enough_trades", len(trades) >= ctx.min_trades, f"{len(trades)} trades (need {ctx.min_trades})"):
        return False, rep
    nf = sum(1 for x in sc["fold_sharpes"] if x > 0)
    if not check("walk_forward", nf >= v["min_folds_profitable"], f"profitable in {nf}/{len(sc['fold_sharpes'])} periods"):
        return False, rep
    if not check("most_stocks", sc["sym_pos"] >= v["min_symbols_profitable"],
                 f"profitable on {sc['sym_pos']:.0%} of the stocks it traded"):
        return False, rep
    pf = rep["research"]["profit_factor"]
    if not check("profit_factor", pf >= v["min_profit_factor"], f"profit factor {pf:.2f} after costs"):
        return False, rep
    base = bt.sharpe(daily)
    rnd = random_entry_test(ctx, g, start, end, rng)
    pct = float((rnd < base).mean())
    if not check("beats_random", pct >= v["random_entry_pctile"],
                 f"beats {pct:.0%} of random-entry copies (Sharpe {base:.2f} vs {rnd.mean():.2f})"):
        return False, rep
    at = alpha_tstat(daily, ctx.bench_daily(tf, start, end))
    if not check("beats_qqq", at >= v["min_alpha_t"], f"edge beyond QQQ's own daily move: t={at:.2f}"):
        return False, rep
    dsr = deflated_sharpe(daily, n_trials)
    rep["probation"] = False
    if not check("not_luck", dsr >= v["dsr_min"], f"deflated Sharpe prob {dsr:.3f} after {n_trials:,} distinct ideas"):
        if not probation_ok:
            return False, rep
        rep["probation"] = True
    shs = np.array([bt.sharpe(ctx.run(perturb(g, rng), start, end)[2]) for _ in range(int(v["robustness_tests"]))])
    frac, med = float((shs > 0).mean()), float(np.median(shs))
    if not check("robust", frac >= v["robustness_min_profitable"] and med >= 0.4 * base,
                 f"{frac:.0%} of nudged copies profitable, median Sharpe {med:.2f} vs {base:.2f}"):
        return False, rep
    rep["research_trade_R"] = [round(float(x), 4) for x in trades[:, 8][-3000:]]
    rep["research_exit_mix"] = exit_mix(trades)
    rep["research_regimes"] = regime_stats(ctx.books[tf], trades)
    if use_holdout and end < T:
        _, trades_h, daily_h, _ = ctx.run(g, end, T)
        rep["holdout"] = bt.summarize(daily_h, trades_h)
        hs = rep["holdout"]["sharpe"]
        need = max(v["holdout_min_sharpe"], v["holdout_min_ratio"] * rep["research"]["sharpe"])
        rnd_h = random_entry_test(ctx, g, end, T, rng)
        pct_h = float((rnd_h < hs).mean())
        min_h = max(int(v.get("holdout_min_trades", 30)), int(getattr(ctx, "min_trades_holdout", 0)))
        if not check("holdout_exam", hs >= need and rep["holdout"]["total_return"] > 0
                     and pct_h >= v["holdout_random_pctile"] and len(trades_h) >= min_h,
                     f"Sharpe {hs:.2f} (needed {need:.2f}) on {len(daily_h)} never-seen days, "
                     f"{len(trades_h)} trades, beats {pct_h:.0%} of random copies"):
            return False, rep
        allR = np.concatenate([trades[:, 8], trades_h[:, 8]])
        rep["expected_max_dd"] = min(rep["research"]["max_dd"], rep["holdout"]["max_dd"])
    else:
        allR = trades[:, 8]
        rep["expected_max_dd"] = rep["research"]["max_dd"]
    rep["expected_trade_R"] = [round(float(x), 4) for x in allR[-3000:]]
    rep["expected_exit_mix"] = rep["research_exit_mix"]
    rep["passed"] = True
    return True, rep


def exit_mix(trades):
    out = {}
    if len(trades):
        for code, name in bt.REASONS.items():
            m = trades[:, 6] == code
            if m.any():
                out[name] = round(float(m.mean()), 3)
    return out


def regime_stats(book, trades):
    buckets = {}
    for tr in trades[-3000:]:
        s, ei = int(tr[0]), int(tr[1])
        for k, val in book.regimes(max(ei - 1, 0), s).items():
            buckets.setdefault(f"{k}:{val}", []).append(tr[8])
    return {k: {"n": len(v), "mean_R": round(float(np.mean(v)), 4)} for k, v in buckets.items()}
