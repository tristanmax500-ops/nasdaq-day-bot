"""Day-trading backtester (also used, unchanged, for paper trading).

Honesty rules built in:
  * decide on a candle's CLOSE, fill at the NEXT candle's OPEN
  * every trade has a stop-loss and a take-profit; stops fill at the stop price
    or worse if price gaps through it; if stop and target are both touched in
    one candle we assume the STOP was hit first
  * every position is closed at the open of the day's last candle (never overnight)
  * position size = risk_per_trade / stop distance, capped at max_position
  * account rules: max open positions, max losing trades per stock per day,
    and a daily loss limit that closes everything and stops trading for the day
  * slippage + commission on every buy and sell
"""
import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f

REASONS = {1: "exit signal", 2: "stop-loss", 3: "take-profit", 4: "time limit", 5: "trailing stop",
           6: "still open", 7: "end of day", 8: "daily loss limit"}
TRADE_COLS = 9   # sym, entry_i, exit_i, entry_px, exit_px, ret(net, unlevered), reason, weight, R


@njit(cache=True)
def _core(o, h, l, c, atr, entry, exitsig, block_next, flat_next, eod, day,
          direction, stop_m, tp_m, trail_m, max_hold, per_day, max_losses,
          cost, risk, maxw, max_open, daily_loss, start, end, lean):
    T, S = c.shape
    n_days = day[end - 1] - day[start] + 1 if end > start else 1
    d0 = day[start] if end > start else 0
    # lean=True (strategy search): one row per DAY instead of per candle - same numbers, ~2x faster
    rets = np.zeros((n_days, S)) if lean else np.zeros((T, S))
    trades = np.empty(((n_days * per_day + 1) * S, 9))   # at most per_day entries per stock per day
    nt = 0
    d = float(direction)
    pos = np.zeros(S, np.int64)
    epx = np.zeros(S)
    ei = np.zeros(S, np.int64)
    spx = np.zeros(S)
    tpx = np.zeros(S)
    best = np.zeros(S)
    w = np.zeros(S)
    rdist = np.zeros(S)
    pin = np.zeros(S, np.bool_)
    pout = np.zeros(S, np.bool_)
    preason = np.zeros(S, np.int64)
    tday = np.zeros(S, np.int64)
    lday = np.zeros(S, np.int64)
    exited = np.zeros(S, np.bool_)
    n_open = 0
    day_pnl = 0.0
    halted = False
    for t in range(start, end):
        ri = day[t] - d0 if lean else t
        if t == start or day[t] != day[t - 1]:
            day_pnl = 0.0
            halted = False
            for s in range(S):
                tday[s] = 0
                lday[s] = 0
                if t != start:
                    pin[s] = False
        if n_open == 0:
            # fast path: flat and nothing scheduled for this open -> only look for new entries
            any_pin = False
            for s in range(S):
                if pin[s]:
                    any_pin = True
                    break
            if not any_pin:
                blk = block_next[t] or halted
                for s in range(S):
                    pout[s] = False
                    pin[s] = (not blk) and entry[t, s]
                continue
        for s in range(S):
            exited[s] = False
        # 1) scheduled exits at the open
        for s in range(S):
            if pos[s] != 0 and pout[s]:
                px = o[t, s]
                r = w[s] * (d * (px / c[t - 1, s] - 1.0) - cost)
                rets[ri, s] += r
                tr = d * (px / epx[s] - 1.0) - 2 * cost
                trades[nt, 0] = s; trades[nt, 1] = ei[s]; trades[nt, 2] = t
                trades[nt, 3] = epx[s]; trades[nt, 4] = px; trades[nt, 5] = tr
                trades[nt, 6] = preason[s]; trades[nt, 7] = w[s]; trades[nt, 8] = tr / rdist[s]
                nt += 1
                if tr < 0:
                    lday[s] += 1
                pos[s] = 0
                n_open -= 1
                exited[s] = True
        # 2) scheduled entries at the open (rotating priority so no stock is always first)
        for k in range(S):
            s = (k + t) % S
            if pos[s] == 0 and pin[s] and not exited[s] and not halted and n_open < max_open \
                    and tday[s] < per_day and lday[s] < max_losses:
                a = atr[t - 1, s]
                px = o[t, s]
                if a == a and a > 0 and px > 0:
                    dist = stop_m * a
                    wt = risk * px / dist
                    if wt > maxw:
                        wt = maxw
                    pos[s] = 1
                    epx[s] = px
                    ei[s] = t
                    w[s] = wt
                    rdist[s] = dist / px
                    spx[s] = px - d * dist
                    tpx[s] = px + d * tp_m * a
                    best[s] = px
                    rets[ri, s] -= wt * cost
                    n_open += 1
                    tday[s] += 1
        # 3) manage open positions during the candle
        for s in range(S):
            if pos[s] == 0:
                continue
            ref = epx[s] if ei[s] == t else c[t - 1, s]
            eff = spx[s]
            why_stop = 2
            if trail_m > 0:
                a = atr[t - 1, s]
                if a == a:
                    tr_px = best[s] - d * trail_m * a
                    if (d > 0 and tr_px > eff) or (d < 0 and tr_px < eff):
                        eff = tr_px
                        why_stop = 5
            fill = 0.0
            why = 0
            if d > 0 and l[t, s] <= eff:
                fill = min(o[t, s], eff)
                why = why_stop
            elif d < 0 and h[t, s] >= eff:
                fill = max(o[t, s], eff)
                why = why_stop
            if why == 0:
                if d > 0 and h[t, s] >= tpx[s]:
                    fill = max(o[t, s], tpx[s])
                    why = 3
                elif d < 0 and l[t, s] <= tpx[s]:
                    fill = min(o[t, s], tpx[s])
                    why = 3
            if why == 0 and eod[t]:
                fill = c[t, s]          # safety net: never hold past the close
                why = 7
            if why != 0:
                rets[ri, s] += w[s] * (d * (fill / ref - 1.0) - cost)
                tr = d * (fill / epx[s] - 1.0) - 2 * cost
                trades[nt, 0] = s; trades[nt, 1] = ei[s]; trades[nt, 2] = t
                trades[nt, 3] = epx[s]; trades[nt, 4] = fill; trades[nt, 5] = tr
                trades[nt, 6] = why; trades[nt, 7] = w[s]; trades[nt, 8] = tr / rdist[s]
                nt += 1
                if tr < 0:
                    lday[s] += 1
                pos[s] = 0
                n_open -= 1
            else:
                rets[ri, s] += w[s] * d * (c[t, s] / ref - 1.0)
                if (d > 0 and c[t, s] > best[s]) or (d < 0 and c[t, s] < best[s]):
                    best[s] = c[t, s]
        # 4) daily loss limit
        if lean:                      # the day's row already holds today's running total
            day_pnl = 0.0
        for s in range(S):
            day_pnl += rets[ri, s]
        if not halted and day_pnl <= -daily_loss:
            halted = True
        # 5) decide what happens at the next open
        for s in range(S):
            pin[s] = False
            pout[s] = False
            if pos[s] != 0:
                if flat_next[t]:
                    pout[s] = True
                    preason[s] = 7
                elif halted:
                    pout[s] = True
                    preason[s] = 8
                elif exitsig[t, s]:
                    pout[s] = True
                    preason[s] = 1
                elif t - ei[s] + 1 >= max_hold:
                    pout[s] = True
                    preason[s] = 4
            else:
                pin[s] = entry[t, s] and not block_next[t] and not halted
    final = np.zeros((S, 8))
    for s in range(S):
        if pos[s] != 0:
            px = c[end - 1, s]
            tr = d * (px / epx[s] - 1.0) - 2 * cost
            trades[nt, 0] = s; trades[nt, 1] = ei[s]; trades[nt, 2] = end - 1
            trades[nt, 3] = epx[s]; trades[nt, 4] = px; trades[nt, 5] = tr
            trades[nt, 6] = 6; trades[nt, 7] = w[s]; trades[nt, 8] = tr / rdist[s]
            nt += 1
        final[s, 0] = pos[s]
        final[s, 1] = 1.0 if (pin[s] and n_open < max_open and tday[s] < per_day and lday[s] < max_losses) else 0.0
        final[s, 2] = 1.0 if pout[s] else 0.0
        final[s, 3] = epx[s] if pos[s] != 0 else 0.0
        final[s, 4] = spx[s] if pos[s] != 0 else 0.0
        final[s, 5] = tpx[s] if pos[s] != 0 else 0.0
        final[s, 6] = w[s] if pos[s] != 0 else 0.0
        final[s, 7] = 1.0 if halted else 0.0
    return rets, trades[:nt], final


# ----------------------------------------------------------------------------

def cond_mask(book, c):
    x = book.get(c["f"], c["p"])
    v = c["v"]
    with np.errstate(invalid="ignore"):
        if c["op"] == "<":
            return x < v
        if c["op"] == ">":
            return x > v
        prev = np.empty_like(x)
        prev[0] = np.nan
        prev[1:] = x[:-1]
        if c["op"] == "xa":
            return (x > v) & (prev <= v)
        return (x < v) & (prev >= v)


def signals(book, g):
    entry = cond_mask(book, g["entry"][0])
    for c in g["entry"][1:]:
        entry = entry & cond_mask(book, c)
    ex = cond_mask(book, g["exit"]) if g.get("exit") else np.zeros_like(entry)
    return entry, ex


class Rules:
    """Account-level day-trading rules + per-candle timing masks for one panel."""

    def __init__(self, cfg, panel, fomc_dates=()):
        dtc = cfg["day_trading"]
        self.risk = float(dtc["risk_per_trade_pct"]) / 100
        self.maxw = float(dtc["max_position_pct"]) / 100
        self.max_open = int(dtc["max_open_positions"])
        self.max_losses = int(dtc["max_losses_per_symbol_day"])
        self.daily_loss = float(dtc["max_daily_loss_pct"]) / 100
        T = panel.T
        eod = panel.eod.copy()
        from .config import TF_MINUTES
        step = TF_MINUTES[panel.tf]
        mins_left = panel.day_len[panel.day] - panel.mins     # minutes from the next open to the close
        # flatten at the open of the day's last candle (never hold overnight)
        self.flat_next = eod | (mins_left <= step)
        block = self.flat_next | (mins_left < int(dtc["no_entries_last_minutes"]))
        if dtc.get("fomc_blackout", True) and fomc_dates:
            fd = np.array([d in fomc_dates for d in panel.day_dates], dtype=bool)
            on_fomc = fd[panel.day]
            next_open = 570 + panel.mins                       # minutes after midnight NY (open = 9:30)
            block |= on_fomc & (next_open >= 13 * 60 + 45) & (next_open < 14 * 60 + 30)
        self.block_next = block
        self.eod = eod


def run(book, g, cost, rules, start=None, end=None, entry_override=None, lean=False, entry=None, ex=None):
    """lean=False: returns per-candle returns [T, S] (paper trading, reports).
    lean=True: returns per-day returns [days, S] for start..end (strategy search; faster)."""
    p = book.p
    start = 1 if start is None else max(1, int(start))
    end = p.T if end is None else int(end)
    if entry is None:
        entry, ex = signals(book, g)
    if entry_override is not None:
        entry = entry_override
    entry = np.ascontiguousarray(entry)   # row-major = fast candle-by-candle access
    ex = np.ascontiguousarray(ex)
    return _core(p.o, p.h, p.l, p.c, book.atr, entry, ex, rules.block_next, rules.flat_next, rules.eod,
                 p.day, 1 if g["dir"] == "long" else -1, float(g["stop"]), float(g["tp"]),
                 float(g["trail"]), int(g["hold"]), int(g["per_day"]), rules.max_losses, float(cost),
                 rules.risk, rules.maxw, rules.max_open, rules.daily_loss, start, end, bool(lean))


# ----------------------------------------------------------------------------
# metrics (daily)
# ----------------------------------------------------------------------------

def daily_returns(rets, day, start, end):
    """Account return per trading day (flat overnight, so days are independent)."""
    port = rets[start:end].sum(axis=1)
    d = day[start:end] - day[start]
    return np.bincount(d, weights=port) if len(d) else np.zeros(0)


def sharpe(r, periods=252):
    r = np.asarray(r)
    if len(r) < 5:
        return 0.0
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(periods)) if sd > 1e-12 else 0.0


def max_drawdown(r):
    if len(r) == 0:
        return 0.0
    eq = np.cumprod(1 + np.asarray(r))
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def summarize(daily, trades, capital=100000.0):
    n_days = len(daily)
    tr = trades[:, 5] if len(trades) else np.zeros(0)
    R = trades[:, 8] if len(trades) else np.zeros(0)
    w = trades[:, 7] if len(trades) else np.zeros(0)
    pnl_tr = tr * w
    wins = pnl_tr[pnl_tr > 0].sum()
    losses = -pnl_tr[pnl_tr < 0].sum()
    total = float(np.prod(1 + daily) - 1) if n_days else 0.0
    years = n_days / 252 if n_days else 1
    return {
        "sharpe": round(sharpe(daily), 3),
        "total_return": round(total, 4),
        "pnl_usd": round(total * capital, 0),
        "cagr": round(float((1 + total) ** (1 / years) - 1) if total > -1 and years > 0 else -1.0, 4),
        "max_dd": round(max_drawdown(daily), 4),
        "trades": int(len(tr)),
        "trades_per_day": round(len(tr) / n_days, 2) if n_days else 0.0,
        "win_rate": round(float((tr > 0).mean()), 3) if len(tr) else 0.0,
        "avg_R": round(float(R.mean()), 3) if len(R) else 0.0,
        "profit_factor": round(float(wins / losses), 2) if losses > 0 else (99.0 if wins > 0 else 0.0),
        "green_days": round(float((daily > 0).sum() / max((daily != 0).sum(), 1)), 3) if n_days else 0.0,
        "days": int(n_days),
    }
