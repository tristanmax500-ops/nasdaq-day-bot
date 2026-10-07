"""Paper trading (forward test): every champion/challenger is simulated on candles
that closed AFTER it was created, with exactly the same engine as the backtest.
The result also says what the strategy wants to do at the next open, which the
broker module turns into real Alpaca paper orders.
"""
import numpy as np
import pandas as pd

from . import backtest as bt


def run_forward(ctx, g, since):
    """ctx: EvalContext built on the latest data (research_frac=1)."""
    book = ctx.books[g["tf"]]
    rules = ctx.rules[g["tf"]]
    p = book.p
    since = pd.Timestamp(since)
    since = since.tz_localize("UTC") if since.tzinfo is None else since.tz_convert("UTC")
    start = int(np.searchsorted(p.index, since, side="right"))
    if start >= p.T:
        return None
    start = max(start, 1)
    rets, trades, final = bt.run(book, g, ctx.cost, rules, start, p.T)
    daily = bt.daily_returns(rets, p.day, start, p.T)
    from .config import TF_MINUTES
    step = pd.Timedelta(minutes=TF_MINUTES[p.tf])
    out = []
    for tr in trades:
        s, ei, xi = int(tr[0]), int(tr[1]), int(tr[2])
        is_open = int(tr[6]) == 6
        out.append({
            "symbol": p.symbols[s], "entry_time": (p.index[ei] - step).isoformat(),   # open of candle ei
            "exit_time": None if is_open else p.index[xi].isoformat(),
            "entry_px": round(float(tr[3]), 4), "exit_px": round(float(tr[4]), 4),
            "ret": float(tr[5]), "R": float(tr[8]), "weight": float(tr[7]),
            "reason": bt.REASONS[int(tr[6])], "regime": book.regimes(max(ei - 1, 0), s), "open": is_open,
        })
    d = 1 if g["dir"] == "long" else -1
    last = p.T - 1
    targets = {}
    for s, sym in enumerate(p.symbols):
        pos, pin, pout = final[s, 0], final[s, 1], final[s, 2]
        a = book.atr[last, s]
        px = p.c[last, s]
        if pos == 0 and pin > 0 and np.isfinite(a) and a > 0:
            dist = g["stop"] * a
            targets[sym] = {
                "action": "BUY" if d > 0 else "SELL_SHORT", "side": d,
                "ref_price": round(float(px), 4),
                "stop": round(float(px - d * dist), 2), "tp": round(float(px + d * g["tp"] * a), 2),
                "weight": float(min(ctx.rules[g["tf"]].risk * px / dist, ctx.rules[g["tf"]].maxw)),
                "stop_atr": g["stop"], "tp_atr": g["tp"], "atr": float(a),
            }
        elif pos != 0 and pout > 0:
            targets[sym] = {"action": "EXIT", "side": 0, "ref_price": round(float(px), 4)}
        elif pos != 0:
            targets[sym] = {"action": "HOLD", "side": d, "ref_price": round(float(px), 4),
                            "stop": round(float(final[s, 4]), 2), "tp": round(float(final[s, 5]), 2),
                            "weight": float(final[s, 6])}
    eq = np.cumprod(1 + daily)
    days = [p.day_dates[p.day[start] + i].isoformat() for i in range(len(daily))]
    curve = [[days[i], round(float(eq[i]), 5)] for i in range(len(eq))]
    closed = np.array([[0, 0, 0, 0, 0, t["ret"], 1, t["weight"], t["R"]] for t in out if not t["open"]]).reshape(-1, 9)
    stats = bt.summarize(daily, closed)
    stats["candles"] = int(p.T - start)
    stats["open_positions"] = sum(1 for t in out if t["open"])
    stats["halted_today"] = bool(final[0, 7]) if len(final) else False
    stats["last_candle"] = p.index[last].isoformat()
    return {"trades": out, "targets": targets, "equity": curve, "stats": stats}


def paper_step(ctx, registry, log=print):
    res_all = {}
    for s in registry.list(["champion", "challenger"]):
        g = s["genome"]
        if g["tf"] not in ctx.books:
            continue
        res = run_forward(ctx, g, s["paper_start"])
        if res is None:
            registry.set_paper_state(s["id"], [], {}, {"trades": 0, "candles": 0})
            continue
        registry.replace_paper_trades(s["id"], res["trades"])
        registry.set_paper_state(s["id"], res["equity"], res["targets"], res["stats"])
        res_all[s["id"]] = res
        st = res["stats"]
        log(f"  {s['status']:<10} {s['id']} v{s['version']} | {st['trades']} trades, P&L {st['total_return']:+.2%} "
            f"(${st['pnl_usd']:+,.0f}), win {st['win_rate']:.0%}, PF {st['profit_factor']:.2f}, "
            f"{st['open_positions']} open")
    return res_all
