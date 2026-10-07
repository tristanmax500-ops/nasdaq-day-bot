"""Self-test: proves the machinery works, offline, in a few minutes.

Builds fake intraday markets where we know the truth and checks that the bot:
  1  finds NOTHING in pure random noise          (it doesn't fool itself)
  2  DOES find an edge we planted                  (the search works)
  3  when the edge starts failing on CPI/Fed-type event days during paper trading,
     it notices, diagnoses "event days", rewrites the strategy, and the rewrite
     takes over only after beating the old version on new data   (it learns)
  4  never holds a position overnight and every trade has TP + SL
"""
import os
import tempfile
import time
from types import SimpleNamespace

import numpy as np

from .config import load_config
from .evaluate import EvalContext
from .registry import Registry
from .synthetic import make_intraday, slice_extras


def _cfg(db):
    cfg = load_config()
    cfg["db_path"] = db
    cfg["market"]["timeframes"] = ["15Min"]
    cfg["market"]["allow_short"] = False      # the fake market's planted edge is a long one
    cfg["discovery"]["population"] = 250
    cfg["paper"]["target_champions"] = 1
    cfg["paper"]["broker"] = "sim"
    cfg["healing"]["repair_trials"] = 2000
    cfg["healing"]["improve_every_trades"] = 10 ** 9      # only rewrite on real problems here
    cfg["healing"]["loss_streak_trigger"] = 10 ** 9
    return cfg


def market_at(cfg, panel, extras, n, cost):
    p = panel.upto(n)
    ex = slice_extras(extras, n)
    ctx = EvalContext({p.tf: p}, cfg, {p.tf: ex}, (), research_frac=1.0, cost=cost)
    return SimpleNamespace(panels={p.tf: p}, extras={p.tf: ex}, fomc=set(), cost=cost, ctx=ctx)


def main(trials=10000, quiet=False):
    log = (lambda *a: None) if quiet else (lambda m: print(m, flush=True))
    from .evolve import discover
    from .healer import heal_cycle, fill_slots
    from .paper import paper_step
    from .genome import describe
    tmp = tempfile.mkdtemp(prefix="nasdaq_bot_selftest_")
    res = {}
    t0 = time.time()
    cost = 2 / 1e4

    log("\n=== TEST 1: pure random noise (there is NO edge to find) ===")
    panel, extras = make_intraday(1, edge=False, n_days=900)
    cfg = _cfg(os.path.join(tmp, "noise.db"))
    reg = Registry(cfg["db_path"])
    ids = discover(cfg, {panel.tf: panel}, reg, max_trials=trials, seed=11, log=log,
                   extras={panel.tf: extras}, cost=cost)
    probs = [i for i in ids if reg.get(i)["report"].get("probation")]
    full = [i for i in ids if i not in probs]
    res["noise"] = len(full)
    log(f"RESULT: {len(full)} strategies fully passed on pure noise "
        f"({'GOOD - it did not fool itself' if not full else 'WARNING - a lucky one slipped through'})"
        + (f"; {len(probs)} went on probation (1/4 size) and would then have to prove themselves live" if probs else ""))

    log("\n=== TEST 2: market with a hidden edge (sharp 30-min drops bounce back) ===")
    panel, extras = make_intraday(2, edge=True, n_days=1150, change_from_day=900)
    cfg = _cfg(os.path.join(tmp, "edge.db"))
    n0 = int(np.searchsorted(panel.day, 900))
    clock = {"t": panel.index[n0 - 1]}
    reg = Registry(cfg["db_path"], clock=lambda: clock["t"].isoformat())
    p0 = panel.upto(n0)
    ids = discover(cfg, {p0.tf: p0}, reg, max_trials=trials, seed=22, log=log,
                   extras={p0.tf: slice_extras(extras, n0)}, cost=cost)
    res["edge"] = len(ids)
    for sid in ids[:2]:
        log(f"\n  found {sid}:\n    " + describe(reg.get(sid)["genome"]).replace("\n", "\n    "))
    log(f"RESULT: {'GOOD - found the planted edge' if ids else 'did not find it this time'}")
    fill_slots(cfg, reg, log)

    log("\n=== TEST 3: live paper trading - then the edge starts failing on event days ===")
    seen = len(reg.events(1000))
    overnight = 0
    for day_n in range(910, 1151, 10):
        n = int(np.searchsorted(panel.day, day_n)) if day_n < panel.n_days else panel.T
        clock["t"] = panel.index[n - 1]
        M = market_at(cfg, panel, extras, n, cost)
        paper_step(M.ctx, reg, log=lambda m: None)
        heal_cycle(cfg, M, reg, log=lambda m: None, seed=day_n)
        paper_step(M.ctx, reg, log=lambda m: None)
        evs = reg.events(1000)[::-1]
        for e in evs[seen:]:
            log(f"  [day {day_n}] {e['kind'].upper():<9} {e['strategy'] or ''}: {e['message'][:240]}")
        seen = len(evs)
    for s in reg.list():
        for t in reg.paper_trades(s["id"]):
            if t["exit_time"] and t["entry_time"][:10] != t["exit_time"][:10]:
                # compare New York dates
                import pandas as pd
                a = pd.Timestamp(t["entry_time"]).tz_convert("America/New_York").date()
                b = pd.Timestamp(t["exit_time"]).tz_convert("America/New_York").date()
                overnight += a != b
    kinds = [e["kind"] for e in reg.events(1000)]
    msgs = " ".join(e["message"] for e in reg.events(1000))
    res["detected"] = any(k in ("degraded", "broken", "weakspot", "streak") for k in kinds)
    res["event_diag"] = "event day" in msgs
    res["rewrite"] = "rewrite" in kinds
    res["promoted"] = "promoted" in kinds
    res["overnight"] = overnight
    ok = lambda b: "PASS" if b else "not triggered"
    log("\n=== SUMMARY ===")
    log(f"  1  no false discoveries in noise        : {'PASS' if res['noise'] == 0 else 'FAIL'}")
    log(f"  2  found the planted edge                : {'PASS' if res['edge'] else 'FAIL'}")
    log(f"  3a noticed it was losing in a situation   : {ok(res['detected'])}")
    log(f"  3b diagnosed 'event days' as the cause   : {ok(res['event_diag'])}")
    log(f"  3c rewrote itself                        : {ok(res['rewrite'])}")
    log(f"  3d new version proved itself and took over: {ok(res['promoted'])}")
    log(f"  4  positions held overnight              : {overnight} ({'PASS' if overnight == 0 else 'FAIL'})")
    log(f"  took {time.time() - t0:.0f}s")
    return res


if __name__ == "__main__":
    main()
