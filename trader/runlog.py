"""Rebuilds the history of strategy searches from logs/bot.log.

New searches are recorded directly by the bot (registry.add_search_run). This only
recovers the searches that ran before that existed, so the progress chart is complete.
"""
import re
from datetime import datetime, timezone

from .config import LOG_DIR

TS = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\s+"
R_START = re.compile(TS + r"Discovery starting: .*?(\d+) symbol-series, budget\s*([\d.]+)?")
R_CORES = re.compile(TS + r"\s*using (\d+) CPU cores")
R_GEN = re.compile(TS + r"\s*\[discovery\] gen\s+(\d+) \| tested ([\d,]+) strategies \(([\d,]+)/s\) \| best score (-?[\d.]+)")
R_RESTART = re.compile(TS + r"\s*\[discovery\] stuck for")
R_TESTED = re.compile(TS + r"\s*tested ([\d,]+) new strategies this run \(([\d,]+) in the AI's lifetime, ([\d,]+) genuinely")
R_FINAL = re.compile(TS + r"\s*(\d+) finalists enter")
R_ELIM = re.compile(TS + r"\s*eliminated: (.*)$")
R_SURV = re.compile(TS + r"\s*(\d+) survived research checks")
R_EXAM = re.compile(TS + r"\s*(PASS|fail)\s+research Sharpe (-?[\d.]+), PF ([\d.]+) \| holdout Sharpe (-?[\d.]+), "
                         r"return ([+-]?[\d.]+)%, (\d+) trades")
R_DONE = re.compile(TS + r"Discovery finished: (\d+) new")


def _iso(ts):
    # log times are the computer's local time
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").astimezone().astimezone(timezone.utc).isoformat()


def _mins(a, b):
    f = "%Y-%m-%d %H:%M:%S"
    return (datetime.strptime(b, f) - datetime.strptime(a, f)).total_seconds() / 60


def parse_log(path=None):
    path = path or (LOG_DIR / "bot.log")
    runs, cur, t0 = [], None, None
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except Exception:
        return []

    def close(status):
        if cur is not None:
            cur["status"] = status
            if len(cur["curve"]) > 80:
                step = len(cur["curve"]) / 80
                cur["curve"] = [cur["curve"][int(i * step)] for i in range(80)] + [cur["curve"][-1]]
            runs.append(cur)

    for ln in lines:
        m = R_START.match(ln)
        if m:
            close("stopped")
            t0 = m.group(1)
            series = int(m.group(2))
            n = max(1, series // 2)
            cur = {"started": _iso(t0), "universe": "QQQ only" if n == 1 else f"{n} symbols",
                   "budget_min": float(m.group(3)) if m.group(3) else None, "curve": [], "restarts": 0,
                   "exam": [], "eliminated": {}, "from_log": True}
            continue
        if cur is None:
            continue
        if (m := R_CORES.match(ln)):
            cur["cores"] = int(m.group(2))
        elif (m := R_GEN.match(ln)):
            best = max(float(m.group(5)), cur["curve"][-1][1] if cur["curve"] else -9.0)   # best so far
            cur["curve"].append([round(_mins(t0, m.group(1)), 2), best])
            cur["trials"] = int(m.group(3).replace(",", ""))
            # the running average speed (the last value can drop if the computer slept)
            cur["rate"] = max(cur.get("rate") or 0.0, float(m.group(4).replace(",", "")))
            cur["best_score"] = best
        elif R_RESTART.match(ln):
            cur["restarts"] += 1
        elif (m := R_TESTED.match(ln)):
            cur["trials"] = int(m.group(2).replace(",", ""))
            cur["lifetime"] = int(m.group(3).replace(",", ""))
            cur["ideas"] = int(m.group(4).replace(",", ""))
            cur["seconds"] = round(_mins(t0, m.group(1)) * 60, 1)
        elif (m := R_FINAL.match(ln)):
            cur["finalists"] = int(m.group(2))
        elif (m := R_ELIM.match(ln)):
            for part in m.group(2).split(","):
                if "=" in part:
                    k, v = part.strip().split("=")
                    cur["eliminated"][k] = int(v)
        elif (m := R_SURV.match(ln)):
            cur["survived"] = int(m.group(2))
        elif (m := R_EXAM.match(ln)):
            cur["exam"].append({"pass": m.group(2) == "PASS", "research_sharpe": float(m.group(3)),
                                "pf": float(m.group(4)), "holdout_sharpe": float(m.group(5)),
                                "holdout_return": float(m.group(6)) / 100, "holdout_trades": int(m.group(7))})
        elif (m := R_DONE.match(ln)):
            cur["validated"] = int(m.group(2))
            cur["finished"] = _iso(m.group(1))
            close("finished")
            cur = None
    close("stopped")
    return runs


def backfill(reg, path=None):
    """Once: add searches found in the log that the bot didn't record itself."""
    if reg.meta("search_runs_backfilled"):
        return 0
    have = {r.get("started", "")[:16] for r in reg.search_runs()}
    added = 0
    for r in parse_log(path):
        if r["started"][:16] in have:
            continue
        if r.get("status") == "stopped" and not r.get("trials"):
            continue
        reg.add_search_run(r)
        added += 1
    reg.set_meta("search_runs_backfilled", True)
    return added
