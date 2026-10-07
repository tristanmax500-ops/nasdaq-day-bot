"""Learning from mistakes: judge, diagnose, rewrite, prove.

For every strategy that is paper trading:
  1 JUDGE     compare live trades with what the backtest promised (bootstrap test on
              R-multiples). One bad trade proves nothing; a significant gap does.
              A losing streak (default 6 in a row) triggers a rewrite straight away.
  2 DIAGNOSE  where do the losses cluster? (down days, high volatility, first hour,
              CPI/Fed days, earnings days, bad news ...) Are stops hit far more often?
  3 REWRITE   build targeted fixes ("skip CPI/Fed days", "wider stop", "no trades in
              the first hour" ...) plus thousands of small mutations, evolve them on
              ALL data including the recent paper period, re-run the gauntlet.
  4 PROVE     the new version paper-trades side by side with the old one and only
              takes over if it does better on NEW data.
Healthy strategies also get periodic improvement attempts the same way.
"""
import copy

import numpy as np
import pandas as pd

from .evaluate import gauntlet
from .evolve import Engine, evolve, pick_finalists
from .features import FEATURES
from .genome import describe, key, BINARY
from .paper import run_forward


def loss_streak(trades):
    n = 0
    for t in reversed(trades):
        if t["ret"] < 0:
            n += 1
        else:
            break
    return n


def judge(s, trades, stats, cfg, rng):
    h = cfg["healing"]
    rep = s["report"]
    exp = np.array(rep.get("expected_trade_R") or [0.0])
    R = np.array([t["rmult"] if t.get("rmult") is not None else 0.0 for t in trades])
    n = len(R)
    out = {"n": n, "paper_R": float(R.mean()) if n else 0.0, "expected_R": float(exp.mean()),
           "streak": loss_streak(trades)}
    if n < h["min_trades_to_judge"]:
        out["verdict"] = "warming up"
        return out
    means = exp[rng.integers(0, len(exp), size=(4000, n))].mean(axis=1)
    out["p_value"] = p = float((means <= R.mean()).mean())
    exp_dd = float(rep.get("expected_max_dd", -0.05))
    dd = float(stats.get("max_dd", 0))
    out["paper_dd"], out["expected_dd"] = dd, exp_dd
    if dd < h["drawdown_break_mult"] * exp_dd and dd < -0.02:
        out["verdict"] = "broken"
    elif p < h["degrade_pvalue"]:
        out["verdict"] = "degraded"
    else:
        out["verdict"] = "healthy"
    return out


def probation_review(registry, s, trades, j, need, log=print):
    """A strategy on probation goes to full size once its live paper trades are clearly positive
    (t-stat >= 1 over at least `need` trades) and in line with its backtest ('healthy').
    Returns True while it is still on probation. (Bad results are handled by the normal
    degraded / broken -> retire path.)"""
    if not s["report"].get("probation"):
        return False
    R = np.array([t["rmult"] if t.get("rmult") is not None else 0.0 for t in trades])
    t_live = float(R.mean() / (R.std(ddof=1) / np.sqrt(len(R)))) if len(R) > 2 and R.std() > 0 else 0.0
    if j.get("verdict") == "healthy" and j["n"] >= need and R.mean() > 0 and t_live >= 1.0:
        registry.set_probation(s["id"], False)
        registry.event(s["id"], "promoted", f"Probation passed: {j['n']} live paper trades averaging "
                       f"{R.mean():+.2f}R (t={t_live:.1f}), in line with its backtest. Now trading at full size.")
        log(f"  {s['id']}: probation passed - full size from now on")
        return False
    return True


def diagnose(s, trades):
    rep = s["report"]
    tags, notes = [], []
    if not trades:
        return tags, notes
    mix = rep.get("expected_exit_mix", {})
    for reason, tag, label in (("stop-loss", "stops", "stop-loss"), ("take-profit", "targets", "take-profit")):
        exp_rate = mix.get(reason, 0)
        rate = float(np.mean([t["reason"] == reason for t in trades]))
        if tag == "stops" and rate - exp_rate > 0.15:
            tags.append(tag)
            notes.append(f"stop-loss hit on {rate:.0%} of trades vs {exp_rate:.0%} in backtest")
        if tag == "targets" and exp_rate - rate > 0.15:
            tags.append(tag)
            notes.append(f"take-profit reached on only {rate:.0%} of trades vs {exp_rate:.0%} in backtest")
    bt_reg = rep.get("research_regimes", {})
    worst = None
    for dim in ("day", "vol", "market", "time", "event", "earnings", "news"):
        groups = {}
        for t in trades:
            groups.setdefault(t["regime"].get(dim), []).append(t["rmult"] or 0.0)
        for val, rs in groups.items():
            if val is None or len(rs) < 5:
                continue
            m = float(np.mean(rs))
            others = [t["rmult"] or 0.0 for t in trades if t["regime"].get(dim) != val]
            if m < 0 and m < (np.mean(others) if others else 0):
                damage = -float(np.sum(rs))
                if worst is None or damage > worst[0]:
                    worst = (damage, dim, val, m, len(rs), bt_reg.get(f"{dim}:{val}", {}).get("mean_R"))
    if worst:
        _, dim, val, m, n, btm = worst
        tags.append(f"regime|{dim}|{val}")
        notes.append(f"losses concentrated on {val} ({dim}): {n} trades averaging {m:+.2f}R"
                     + (f" (backtest {btm:+.2f}R there)" if btm is not None else ""))
    tags.append("tighten")
    return tags, notes


def weak_spot(trades, min_n=20, p_max=0.01):
    """A situation where the strategy does significantly worse than everywhere else
    (Welch t-test) and loses money on average. Returns (tag, note) or None."""
    from scipy import stats as st
    best = None
    for dim in ("day", "vol", "market", "time", "event", "earnings", "news"):
        groups = {}
        for t in trades:
            v = t["regime"].get(dim)
            if v is not None:
                groups.setdefault(v, []).append(t["rmult"] or 0.0)
        for val, rs in groups.items():
            others = [x for v2, xs in groups.items() if v2 != val for x in xs]
            if len(rs) < min_n or len(others) < min_n or np.mean(rs) >= 0:
                continue
            p = st.ttest_ind(rs, others, equal_var=False).pvalue
            if p < p_max and (best is None or p < best[0]):
                best = (p, dim, val, float(np.mean(rs)), float(np.mean(others)), len(rs))
    if not best:
        return None
    p, dim, val, m, mo, n = best
    return (f"regime|{dim}|{val}",
            f"weak spot: on {val} ({dim}) it averages {m:+.2f}R over {n} trades vs {mo:+.2f}R otherwise (p={p:.4f})")


_FIX = {
    ("day", "down"): [[("dchg", 1, ">", 0.0)]], ("day", "up"): [[("dchg", 1, "<", 0.0)]],
    ("vol", "high"): [[("volat", 24, "<", 1.3)]], ("vol", "low"): [[("volat", 24, ">", 0.8)]],
    ("market", "down"): [[("mkt", 1, ">", 0.0)]], ("market", "up"): [[("mkt", 1, "<", 0.0)]],
    ("time", "first hour"): [[("tod", 1, ">", 60.0)]], ("time", "last 90 min"): [[("tod", 1, "<", 300.0)]],
    ("time", "midday"): [[("tod", 1, "<", 120.0)], [("tod", 1, ">", 270.0)]],
    ("event", "event day"): [[("macro", 1, "<", 0.5)], [("fomc", 1, "<", 0.5)],
                             [("macro", 1, "<", 0.5), ("fomc", 1, "<", 0.5)]],
    ("earnings", "earnings day"): [[("earn", 1, "<", 0.5)]],
    ("news", "negative"): [[("sent", 1, ">", -0.3)]], ("news", "positive"): [[("sent", 1, "<", 0.3)]],
}


def _add_conds(g, conds):
    h = copy.deepcopy(g)
    for f, p, op, v in conds:
        c = {"f": f, "p": p, "op": op, "v": v}
        if len(h["entry"]) >= 3:
            h["entry"][-1] = c
        else:
            h["entry"].append(c)
    return h


def targeted_fixes(g, tags):
    fixes = []
    for t in tags:
        if t.startswith("regime|"):
            _, dim, val = t.split("|")
            for conds in _FIX.get((dim, val), []):
                fixes.append(_add_conds(g, conds))
        elif t == "stops":
            for m in (1.5, 2.0):
                h = copy.deepcopy(g)
                h["stop"] = round(h["stop"] * m, 2)
                fixes.append(h)
        elif t == "targets":
            for m in (0.6, 0.8):
                h = copy.deepcopy(g)
                h["tp"] = round(max(0.3, h["tp"] * m), 2)
                fixes.append(h)
        elif t == "tighten":
            for frac in (0.04, 0.08):
                h = copy.deepcopy(g)
                for c in h["entry"]:
                    if c["f"] in BINARY:
                        continue
                    lo, hi = FEATURES[c["f"]][1], FEATURES[c["f"]][2]
                    c["v"] = round(c["v"] + (-1 if c["op"] in ("<", "xb") else 1) * frac * (hi - lo), 4)
                fixes.append(h)
    return fixes


def repair(cfg, M, registry, s, tags, rng, log, reason):
    g = s["genome"]
    tf = g["tf"]
    pan = M.panels[tf]
    seeds = targeted_fixes(g, tags) + [g]
    log(f"  rewriting {s['id']} ({reason}): {len(seeds) - 1} targeted fixes + mutations ...")
    ps = pd.Timestamp(s["paper_start"] or pan.index[-1])
    ps = ps.tz_localize("UTC") if ps.tzinfo is None else ps.tz_convert("UTC")
    ps_i = int(np.searchsorted(pan.index, ps))
    recent = max(0, min(ps_i, pan.T - int(40 * pan.bars_per_day)))
    engine = Engine({tf: pan}, cfg, research_frac=1.0, log=log, recent_start={tf: recent},
                    extras={tf: M.extras.get(tf)}, fomc_dates=M.fomc, cost=M.cost)
    try:
        ranked, trials, srs, fams = evolve(engine, rng, max_trials=int(cfg["healing"]["repair_trials"]),
                                           seeds=seeds, timeframes=[tf], allow_short=cfg["market"]["allow_short"],
                                           population=200, mutation_strength=0.5, log=log, label="rewrite")
        registry.add_trials(trials, srs, fams)
        n_eff, sr_var = registry.effective_trials(), registry.sr_variance()
        parent_fit = engine.ctx.score(g)["fitness"]
        known = registry.known_keys()
        for fit, g2, _ in pick_finalists(ranked, 15, per_family=4):
            if key(g2) in known:
                continue
            if fit <= parent_fit + max(0.05, 0.05 * abs(parent_fit)):
                break
            ok, rep = gauntlet(engine.ctx, g2, n_eff, sr_var, rng, use_holdout=False,
                               probation_ok=bool(cfg["validation"].get("probation", True)))
            if ok and s["report"].get("probation"):
                rep["probation"] = True          # a rewrite of a probation strategy stays on probation
            if not ok:
                continue
            sid = registry.add_strategy(g2, rep, "challenger", fitness=fit, parent=s["id"], family=s["family"],
                                        version=s["version"] + 1, note=reason)
            registry.event(sid, "rewrite", f"v{s['version'] + 1} of {s['id']} written ({reason}). Score {fit:.2f} vs "
                           f"{parent_fit:.2f}. Must now beat the old version in paper trading. "
                           + describe(g2).replace("\n", " | "))
            log(f"  -> new version {sid} (score {fit:.2f} vs {parent_fit:.2f})")
            return sid
        log(f"  no rewrite beat the current version and passed validation ({trials:,} variations tried)")
        return None
    finally:
        engine.close()


def heal_cycle(cfg, M, registry, log=print, seed=None, allow_repairs=True):
    rng = np.random.default_rng(seed)
    h = cfg["healing"]
    ctx = M.ctx

    # ---- 1. challengers: promote / keep proving / drop
    for c in registry.list(["challenger"]):
        parent = registry.get(c["parent"]) if c["parent"] else None
        trades = registry.paper_trades(c["id"])
        n = len(trades)
        if c["genome"]["tf"] not in ctx.books:
            continue
        mine = run_forward(ctx, c["genome"], c["paper_start"])
        my_ret = mine["stats"]["total_return"] if mine else 0.0
        if n < h["promote_min_trades"]:
            registry.set_health(c["id"], f"proving ({n}/{h['promote_min_trades']} trades)")
            continue
        verdict = judge(c, trades, mine["stats"] if mine else {}, cfg, rng)["verdict"]
        if parent and parent["status"] == "champion":
            theirs = run_forward(ctx, parent["genome"], c["paper_start"])
            their_ret = theirs["stats"]["total_return"] if theirs else 0.0
            if my_ret > their_ret and my_ret > 0 and verdict not in ("degraded", "broken"):
                registry.set_status(parent["id"], "retired", note=f"replaced by v{c['version']} ({c['id']})")
                registry.set_status(c["id"], "champion")
                registry.event(c["id"], "promoted", f"v{c['version']} beat v{parent['version']} on new data: "
                               f"{my_ret:+.2%} vs {their_ret:+.2%} over {n} trades. Now live in paper trading.")
                log(f"  PROMOTED {c['id']} v{c['version']} ({my_ret:+.2%} vs {their_ret:+.2%})")
            elif n >= 3 * h["promote_min_trades"] or verdict in ("degraded", "broken"):
                registry.set_status(c["id"], "retired", note="did not beat the version it tried to replace")
                registry.event(c["id"], "dropped", f"Rewrite did not beat the original ({my_ret:+.2%} vs {their_ret:+.2%}).")
                log(f"  dropped rewrite {c['id']} ({my_ret:+.2%} vs {their_ret:+.2%})")
            else:
                registry.set_health(c["id"], f"behind ({my_ret:+.2%} vs {their_ret:+.2%})")
        else:
            if my_ret > 0 and verdict not in ("degraded", "broken"):
                full = len(registry.list(["champion"])) >= int(cfg["paper"]["target_champions"])
                registry.set_status(c["id"], "reserve" if full else "champion")
                registry.event(c["id"], "promoted", f"Rewrite proved itself on new data ({my_ret:+.2%}, {n} trades)."
                               + (" All slots full - waiting in reserve." if full else ""))
            else:
                registry.set_status(c["id"], "retired", note="rewrite failed on new data")
                registry.event(c["id"], "dropped", f"Rewrite did not work on new data ({my_ret:+.2%}).")

    # ---- 2. champions
    n_chal = len(registry.list(["challenger"]))
    for s in registry.list(["champion"]):
        trades = registry.paper_trades(s["id"])
        st = (registry.paper_state(s["id"]) or {}).get("stats", {})
        j = judge(s, trades, st, cfg, rng)
        v = j["verdict"]
        need = int(cfg["validation"].get("probation_trades", 40))
        prob = probation_review(registry, s, trades, j, need, log)
        registry.set_health(s["id"], v + (f" (p={j['p_value']:.2f})" if "p_value" in j else "")
                            + (f", {j['streak']} losses in a row" if j["streak"] >= 3 else "")
                            + (f" · probation {min(j['n'], need)}/{need} trades, 1/4 size" if prob else ""))
        has_chal = any(c["parent"] == s["id"] for c in registry.list(["challenger"]))
        can_rewrite = allow_repairs and not has_chal and n_chal < cfg["paper"]["max_challengers"]
        if v in ("degraded", "broken"):
            tags, notes = diagnose(s, trades)
            msg = (f"{v.upper()}: live avg {j['paper_R']:+.2f}R per trade vs expected {j['expected_R']:+.2f}R "
                   f"over {j['n']} trades (p={j.get('p_value', 0):.3f}). Diagnosis: "
                   + ("; ".join(notes) if notes else "no single clear cause"))
            if not str(s.get("health") or "").startswith(v) or v == "broken":
                registry.event(s["id"], v, msg)
            log(f"  {s['id']}: {msg}")
            new = repair(cfg, M, registry, s, tags, rng, log, reason=f"fix after {v}") if can_rewrite else None
            if new:
                n_chal += 1
            if v == "broken" or (not new and not has_chal):
                registry.set_status(s["id"], "retired", note=f"{v}; edge gone")
                registry.event(s["id"], "retired", f"Stopped trading it: {v}." + (" Its rewrite is being tested." if new else ""))
                log(f"  retired {s['id']} ({v})")
            continue
        if j["streak"] >= h["loss_streak_trigger"] and can_rewrite and j["n"] >= 10:
            tags, notes = diagnose(s, trades)
            registry.event(s["id"], "streak", f"{j['streak']} losing trades in a row - writing a fix. "
                           + ("; ".join(notes) if notes else ""))
            if repair(cfg, M, registry, s, tags, rng, log, reason=f"{j['streak']}-trade losing streak"):
                n_chal += 1
            continue
        ws = weak_spot(trades) if can_rewrite else None
        if ws and str(s.get("note") or "") != ws[0]:
            registry.event(s["id"], "weakspot", ws[1] + " - writing an improved version")
            log(f"  {s['id']}: {ws[1]}")
            registry.db.execute("UPDATE strategies SET note=? WHERE id=?", (ws[0], s["id"]))
            registry.db.commit()
            if repair(cfg, M, registry, s, [ws[0], "tighten"], rng, log, reason=ws[1].split(":")[0] + " fix"):
                n_chal += 1
            continue
        if v == "healthy" and can_rewrite and j["n"] - (s["last_judged_trades"] or 0) >= h["improve_every_trades"]:
            registry.set_health(s["id"], "healthy", judged_trades=j["n"])
            tags, _ = diagnose(s, trades)
            if repair(cfg, M, registry, s, tags, rng, log, reason="routine improvement"):
                n_chal += 1

    fill_slots(cfg, registry, log)


def fill_slots(cfg, registry, log=print):
    champs = registry.list(["champion"])
    orphans = [c for c in registry.list(["challenger"])
               if not c["parent"] or (registry.get(c["parent"]) or {}).get("status") != "champion"]
    free = int(cfg["paper"]["target_champions"]) - len(champs) - len(orphans)
    if free <= 0:
        return 0
    res = sorted(registry.list(["reserve"]), key=lambda s: -s["report"].get("holdout", {}).get("sharpe", 0))
    for s in res[:free]:
        registry.set_status(s["id"], "champion")
        registry.event(s["id"], "deployed", "Started paper trading.")
        log(f"  deployed {s['id']} to paper trading")
    return min(free, len(res))
