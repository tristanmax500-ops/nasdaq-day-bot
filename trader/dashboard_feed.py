"""Feeds your Projects Dashboard.

After every 5-minute cycle (and after the close) the bot writes
    <status_folder>/nasdaq_bot_status.js     (loaded by the dashboard page)
    <status_folder>/nasdaq_bot_status.json   (same data, for anything else)
The default folder is Documents/Projects Dashboard. Nothing secret is written:
no API keys, only balances, positions, strategies and recent activity.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .genome import describe

NY = "America/New_York"
_hist_cache = {"at": 0.0, "data": None}


def status_folder(cfg):
    f = (cfg.get("dashboard") or {}).get("status_folder", "~/Documents/Projects Dashboard")
    if not f:
        return None
    p = Path(os.path.expanduser(f))
    return p if p.is_dir() else None


def _account(bot):
    if bot.cfg["paper"]["broker"] != "alpaca":
        return None, [], []
    api = bot.api
    acct = api.account()
    positions = api.positions()
    if time.time() - _hist_cache["at"] > 600:
        try:
            h = api.portfolio_history("3M")
            pts = []
            for ts, eq in zip(h.get("timestamp") or [], h.get("equity") or []):
                if eq is not None:
                    pts.append([pd.Timestamp(ts, unit="s", tz="UTC").tz_convert(NY).date().isoformat(), round(float(eq), 2)])
            _hist_cache.update(at=time.time(), data=pts)
        except Exception:
            _hist_cache.update(at=time.time(), data=_hist_cache["data"] or [])
    return acct, positions, _hist_cache["data"] or []


def build_status(bot, state="running", note=""):
    reg = bot.reg
    out = {"version": 1, "updated": pd.Timestamp.now(tz="UTC").isoformat(), "state": state, "note": note,
           "broker": bot.cfg["paper"]["broker"], "symbols": bot.cfg["market"]["symbols"],
           "lifetime_tested": reg.meta("total_trials", 0), "ideas_tried": len(reg.meta("families", []) or []),
           "slippage_bps": reg.meta("learned_slippage_bps"), "started": reg.meta("first_run")}
    # account
    try:
        acct, positions, hist = _account(bot)
    except Exception as e:
        acct, positions, hist = None, [], []
        out["account_error"] = str(e)[:200]
    if acct:
        eq, last = float(acct["equity"]), float(acct.get("last_equity") or acct["equity"])
        out["account"] = {"equity": round(eq, 2), "day_pnl": round(eq - last, 2),
                          "day_pct": round((eq / last - 1) * 100, 3) if last else 0.0,
                          "buying_power": round(float(acct.get("buying_power") or 0), 2)}
        out["equity_history"] = hist
        try:
            out["market_open"] = bool(bot.api.clock().get("is_open"))
        except Exception:
            pass
    owners = reg.owners()
    orders = {r["symbol"]: dict(r) for r in reg.db.execute(
        "SELECT * FROM broker_orders WHERE status='submitted' ORDER BY submitted")}
    out["positions"] = []
    for p in positions:
        sym = p["symbol"]
        o = orders.get(sym, {})
        out["positions"].append({
            "symbol": sym, "qty": float(p.get("qty", 0)), "side": p.get("side", "long"),
            "entry": float(p.get("avg_entry_price") or 0), "price": float(p.get("current_price") or 0),
            "pnl": float(p.get("unrealized_pl") or 0), "pnl_pct": float(p.get("unrealized_plpc") or 0) * 100,
            "tp": o.get("tp"), "sl": o.get("stop"), "strategy": owners.get(sym), "bots": sym in owners})
    # strategies
    strats = []
    total_pnl = 0.0
    for status in ("champion", "challenger", "reserve"):
        for s in reg.list([status]):
            st = (reg.paper_state(s["id"]) or {}).get("stats", {}) or {}
            if status == "champion":
                total_pnl += st.get("pnl_usd", 0) or 0
            strats.append({
                "id": s["id"], "version": s["version"], "status": status, "health": s.get("health"),
                "probation": bool(s["report"].get("probation")),
                "tf": s["genome"]["tf"], "rules": describe(s["genome"]),
                "trades": st.get("trades", 0), "pnl_usd": st.get("pnl_usd", 0), "win_rate": st.get("win_rate", 0),
                "profit_factor": st.get("profit_factor", 0), "avg_R": st.get("avg_R", 0),
                "max_dd": st.get("max_dd", 0), "curve": (reg.paper_state(s["id"]) or {}).get("equity", [])[-120:],
                "backtest": {k: s["report"].get("research", {}).get(k) for k in ("sharpe", "profit_factor", "win_rate")},
                "holdout_sharpe": s["report"].get("holdout", {}).get("sharpe")})
    out["strategies"] = strats
    out["paper_pnl_usd"] = round(total_pnl, 2)
    # recent trades (simulated forward test of current strategies)
    trades = []
    for s in reg.list(["champion"]):
        for t in reg.paper_trades(s["id"])[-30:]:
            trades.append({"strategy": s["id"], "symbol": t["symbol"], "entry_time": t["entry_time"],
                           "exit_time": t["exit_time"], "R": t["rmult"], "ret": t["ret"], "reason": t["reason"]})
    trades.sort(key=lambda t: t["exit_time"] or "", reverse=True)
    out["recent_trades"] = trades[:25]
    closed = [t for t in trades]
    out["win_rate_recent"] = round(float(np.mean([t["ret"] > 0 for t in closed])), 3) if closed else None
    # number of entries the bot made per trading day (for the dashboard's balance log)
    daily = {}
    for r in reg.db.execute("SELECT submitted FROM broker_orders WHERE status='submitted'"):
        try:
            d = pd.Timestamp(r[0]).tz_convert(NY).date().isoformat()
            daily[d] = daily.get(d, 0) + 1
        except Exception:
            pass
    out["daily_trades"] = daily
    # history of strategy searches (for the progress chart)
    try:
        from .runlog import backfill
        backfill(reg)
        out["search_runs"] = reg.search_runs()[-40:]
    except Exception:
        out["search_runs"] = []
    out["universe"] = "QQQ only" if bot.cfg["market"]["symbols"] == ["QQQ"] else \
        f"{len(bot.cfg['market']['symbols'])} symbols"
    out["validation_needed"] = {"min_trades": bot.cfg["discovery"]["min_trades"],
                                "holdout_min_sharpe": bot.cfg["validation"]["holdout_min_sharpe"]}
    # orders and decisions
    out["recent_orders"] = [dict(r) for r in reg.db.execute(
        "SELECT client_id, strategy, symbol, side, qty, ref_price, stop, tp, submitted, status, fill_price, "
        "slippage_bps FROM broker_orders ORDER BY submitted DESC LIMIT 15")]
    out["events"] = [{"ts": e["ts"], "kind": e["kind"], "strategy": e["strategy"], "message": e["message"]}
                     for e in reg.events(40)]
    # upcoming scheduled events
    try:
        today = pd.Timestamp.now(tz=NY).date()
        horizon = today + pd.Timedelta(days=7)
        up = []
        names = {"fomc": "Fed decision 2:00pm ET", "cpi": "CPI 8:30am ET", "nfp": "Jobs report 8:30am ET",
                 "ppi": "PPI 8:30am ET"}
        for k, label in names.items():
            for d in bot.events.macro.get(k, ()):
                if today <= d <= horizon:
                    up.append([d.isoformat(), label])
        for sym, rows in bot.events.earnings.items():
            for d, tm in rows:
                if today <= d <= horizon:
                    up.append([d.isoformat(), f"{sym} earnings " + {"amc": "after close", "bmo": "before open"}.get(tm, "")])
        out["upcoming"] = sorted(up)
    except Exception:
        out["upcoming"] = []
    try:
        if bot.news is not None:
            out["headlines"] = [{"t": pd.Timestamp(r[0], unit="s", tz="UTC").isoformat(), "score": r[2], "text": r[3]}
                                for r in bot.news.recent_headlines(hours=24, limit=8)]
    except Exception:
        pass
    return out


def write_status(bot, state="running", note=""):
    folder = status_folder(bot.cfg)
    if folder is None:
        return None
    try:
        data = build_status(bot, state, note)
    except Exception as e:
        data = {"version": 1, "updated": pd.Timestamp.now(tz="UTC").isoformat(), "state": "error",
                "note": f"status error: {e}"[:300]}
    js = "window.NASDAQ_BOT = " + json.dumps(data, default=str) + ";\n"
    for name, body in (("nasdaq_bot_status.js", js), ("nasdaq_bot_status.json", json.dumps(data, default=str, indent=1))):
        tmp = folder / (name + ".tmp")
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, folder / name)     # atomic: the dashboard never reads half a file
    return folder / "nasdaq_bot_status.js"
