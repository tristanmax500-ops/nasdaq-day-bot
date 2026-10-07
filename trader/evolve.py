"""The strategy factory: a genetic algorithm that invents, tests and breeds
day-trading strategies by the thousands, then sends the best through the gauntlet.
"""
import multiprocessing as mp
import os
import time

import numpy as np

from .evaluate import EvalContext, gauntlet
from .genome import random_genome, mutate, crossover, key, family_signature, describe

_CTX = None


def _init_worker(panels, cfg, research_frac, recent_start, extras, fomc_dates, cost):
    global _CTX
    # Ctrl+C is for the main process only: it stops the search gracefully, the testers keep going
    try:
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass
    if os.name == "nt":
        # 'below normal' priority: the testers still use every idle bit of CPU, but Windows
        # gives Chrome, Spotify etc. priority, so the computer stays smooth while it searches
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)
        except Exception:
            pass
    _CTX = EvalContext(panels, cfg, extras, fomc_dates, research_frac, cost)
    _CTX.recent_start = dict(recent_start or {})


def _score_batch(genomes):
    return [_CTX.score(g) for g in genomes]


def free_memory_gb():
    """Memory currently available to programs, in GB (None if unknown)."""
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return m.ullAvailPhys / 2 ** 30
        elif os.path.exists("/proc/meminfo"):
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2 ** 20
        elif hasattr(os, "sysconf"):
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES") / 2 ** 30
    except Exception:
        pass
    return None


def auto_workers(cfg):
    """As many parallel testers as the computer can take: all cores but one, and
    ~1.2 GB of free memory per tester (each holds its own copy of 5 years of data),
    keeping 2 GB free for Windows and your other apps."""
    d = cfg["discovery"]
    cores = os.cpu_count() or 2
    by_cpu = max(1, cores - 1)
    free = free_memory_gb()
    per = float(d.get("memory_per_worker_gb", 1.2))
    by_mem = max(1, int((free - 2.0) / per)) if free else 4
    cap = int(d.get("max_workers", 12) or 12)
    n = max(1, min(by_cpu, by_mem, cap))
    return n, f"{cores} cores, {free:.1f} GB free memory" if free else f"{cores} cores"


class _Done:
    def __init__(self, res):
        self.res = res

    def get(self):
        return self.res


class _Async:
    def __init__(self, ar):
        self.ar = ar

    def get(self):
        out = []
        for part in self.ar.get():
            out.extend(part)
        return out


class Engine:
    """Owns a worker pool; scores genomes in parallel."""

    def __init__(self, panels, cfg, research_frac=None, workers=None, log=print, recent_start=None,
                 extras=None, fomc_dates=(), cost=None):
        self.cfg = cfg
        self.log = log
        n = workers if workers is not None else int(cfg["discovery"].get("workers", 0) or 0)
        if n <= 0:
            n, why = auto_workers(cfg)
            log(f"  using {n} CPU cores for testing ({why})")
        self.n = n
        self.ctx = EvalContext(panels, cfg, extras, fomc_dates, research_frac, cost)
        self.ctx.recent_start = dict(recent_start or {})
        self.pool = None
        if n > 1:
            self.pool = mp.get_context("spawn").Pool(
                n, initializer=_init_worker,
                initargs=(panels, cfg, research_frac, recent_start, extras, set(fomc_dates), cost))

    def available_features(self):
        """Features with real data (e.g. news/events are left out if their source is missing)."""
        from .features import FEATURES, EXTERNAL
        out = []
        for f in FEATURES:
            if f in EXTERNAL:
                if not any(np.isfinite(b.get(f, 1)).any() for b in self.ctx.books.values()):
                    continue
            out.append(f)
        return out

    def score_many(self, genomes):
        if not genomes:
            return []
        if self.pool is None:
            return [self.ctx.score(g) for g in genomes]
        k = max(1, len(genomes) // (self.n * 4))
        chunks = [genomes[i:i + k] for i in range(0, len(genomes), k)]
        out = []
        for res in self.pool.map(_score_batch, chunks):
            out.extend(res)
        return out

    def submit(self, genomes):
        """Start scoring in the background; .get() returns the results (in order).
        Lets the main process breed the next generation while the cores are busy."""
        if not genomes or self.pool is None:
            res = self.score_many(genomes)
            return _Done(res)
        k = max(1, len(genomes) // (self.n * 4))
        chunks = [genomes[i:i + k] for i in range(0, len(genomes), k)]
        return _Async(self.pool.map_async(_score_batch, chunks))

    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.pool = None


def evolve(engine, rng, minutes=None, max_trials=None, seeds=None, timeframes=None,
           allow_short=False, population=None, mutation_strength=1.0, log=print, label="discovery",
           skip_keys=None, stats=None, checkpoint=None, checkpoint_every=300):
    """Run the GA. Returns (hall_of_fame, unique trials, random-strategy sharpes, idea families)."""
    cfg = engine.cfg
    timeframes = timeframes or list(engine.ctx.books)
    names = engine.available_features()
    pop_n = int(population or cfg["discovery"]["population"])
    deadline = time.time() + minutes * 60 if minutes else None
    seen = set(skip_keys or ())
    hof = {}
    srs = []            # Sharpe of purely RANDOM strategies (the 'luck' baseline)
    families = set()    # distinct strategy ideas tried
    fresh_keys = set()
    trials = 0
    t0 = time.time()

    def fresh():
        g = random_genome(rng, timeframes, allow_short, names)
        fresh_keys.add(key(g))
        return g

    pop = []
    for s in (seeds or []):
        pop.append(s)
        for _ in range(max(1, pop_n // max(len(seeds), 1) - 1)):
            if len(pop) >= pop_n:
                break
            pop.append(mutate(s, rng, mutation_strength, timeframes, allow_short, names))
    while len(pop) < pop_n:
        pop.append(fresh())

    gen = 0
    scored = []
    hof_cap = 400
    per_fam = int(cfg["discovery"].get("hof_per_family", 6))      # keep the hall of fame diverse
    fam_of = {}                                                    # hof key -> family
    restart_after = int(cfg["discovery"].get("restart_after_gens", 40))
    stuck_min = float(cfg["discovery"].get("stop_when_stuck_minutes", 20) or 0)
    best_seen, last_gain_gen, restarts = -1e9, 0, 0
    top_sig, top_changed_at = None, time.time()
    n_final = int(cfg["discovery"].get("finalists", 30))
    n_min_trades = int(cfg["discovery"]["min_trades"])

    fam_members = {}                                               # family -> its keys in the hof
    worst = {"v": None}                                            # cached weakest hof entry (key)

    def _drop(w):
        fm = fam_members.get(fam_of[w])
        if fm is not None:
            fm.discard(w)
        del hof[w]
        fam_of.pop(w, None)
        if worst["v"] == w:
            worst["v"] = None

    def hof_add(k, item):
        if k in hof:
            return False
        g = item[1]
        f = family_signature(g)
        same = fam_members.setdefault(f, set())
        if len(same) >= per_fam:                       # family full: replace its weakest member
            w = min(same, key=lambda x: hof[x][0])
            if item[0] <= hof[w][0]:
                return False
            _drop(w)
        elif len(hof) >= hof_cap:
            if worst["v"] is None or worst["v"] not in hof:
                worst["v"] = min(hof, key=lambda x: hof[x][0])
            w = worst["v"]
            if item[0] <= hof[w][0]:
                return False
            _drop(w)
        hof[k] = item
        fam_of[k] = f
        same.add(k)
        if worst["v"] is not None and item[0] < hof[worst["v"]][0]:
            worst["v"] = k
        return True

    from collections import deque
    submit = getattr(engine, "submit", None)
    # with several cores: keep 2 generations in flight, so the cores never wait for the
    # main process while it breeds the next one
    pipelined = submit is not None and getattr(engine, "pool", None) is not None
    pending = deque()
    epoch = 0

    def send(population):
        b, ks = [], []
        for g in population:
            k = key(g)
            if k in seen:
                continue
            seen.add(k)
            b.append(g)
            ks.append(k)
        h = submit(b) if submit is not None else _Done(engine.score_many(b))
        pending.append((epoch, b, ks, h))

    def take(ep, batch, bkeys, results, into_scored):
        nonlocal hof_changed
        for g, r, k in zip(batch, results, bkeys):
            families.add(family_signature(g))
            if k in fresh_keys and r["trades"] > 0 and r["fitness"] > -5:
                srs.append(r["sr_day"])
            if into_scored:
                scored.append((r["fitness"], g, r))
            if r["fitness"] > 0:
                hof_changed |= hof_add(k, (r["fitness"], g, r))

    def breed():
        if not scored:
            return [fresh() for _ in range(pop_n)]
        elite_n = max(2, pop_n // 10)
        parents = [x[1] for x in scored]
        fits = np.array([x[0] for x in scored])
        new = [p for p in parents[:elite_n]]

        def pick():
            idx = rng.integers(0, len(parents), size=3)
            return parents[idx[np.argmax(fits[idx])]]

        while len(new) < pop_n:
            r = rng.random()
            if r < 0.15:
                new.append(fresh())
            elif r < 0.45:
                a, b = pick(), pick()
                if a["tf"] == b["tf"] and a["dir"] == b["dir"]:
                    new.append(mutate(crossover(a, b, rng), rng, mutation_strength, timeframes, allow_short, names)
                               if rng.random() < 0.5 else crossover(a, b, rng))
                else:
                    new.append(mutate(a, rng, mutation_strength, timeframes, allow_short, names))
            else:
                new.append(mutate(pick(), rng, mutation_strength, timeframes, allow_short, names))
        return new

    hof_changed = False
    last_ck = time.time()
    try:
        while True:
            if not pending:
                send(pop)
            ep, batch, bkeys, h = pending.popleft()
            results = h.get()
            trials += len(batch)
            hof_changed = False
            take(ep, batch, bkeys, results, into_scored=(ep == epoch))
            gen += 1
            if checkpoint is not None and time.time() - last_ck >= checkpoint_every:
                try:      # save the best-so-far list, so a crash or a closed window loses nothing
                    checkpoint({"hof": list(hof.values()), "trials": trials, "srs": list(srs),
                                "families": set(families)})
                except Exception as e:
                    log(f"  (could not save search progress: {e})")
                last_ck = time.time()
            scored.sort(key=lambda x: -x[0])
            scored = scored[:pop_n]
            best = scored[0] if scored else None
            if stats is not None and scored:
                # best score so far (for the progress chart): one point per generation, thinned later
                bsf = max(scored[0][0], stats["curve"][-1][1] if stats.get("curve") else -9)
                stats.setdefault("curve", []).append([round((time.time() - t0) / 60, 2), round(float(bsf), 3)])
            rate = trials / max(time.time() - t0, 1e-9)
            if gen % 5 == 1 or (deadline and time.time() >= deadline):
                if best:
                    log(f"  [{label}] gen {gen:4d} | tested {trials:,} strategies ({rate:,.0f}/s) | "
                        f"best score {best[0]:.2f}, Sharpe {best[2].get('sharpe', 0):.2f}, trades {best[2]['trades']}")
            if deadline and time.time() >= deadline:
                break
            if max_trials and trials >= max_trials:
                break
            # ---------- stuck? ----------
            if best and best[0] > best_seen + 0.01:
                best_seen, last_gain_gen = best[0], gen
            if hof_changed or top_sig is None:      # the finalist list can only change when the hof does
                sig = tuple(sorted(id(x[1]) for x in pick_finalists(sorted(hof.values(), key=lambda x: -x[0]),
                                                                    n_final, min_trades=n_min_trades)))
            else:
                sig = top_sig
            if sig != top_sig:
                top_sig, top_changed_at = sig, time.time()
            if deadline and stuck_min > 0 and time.time() - top_changed_at > stuck_min * 60:
                log(f"  [{label}] no new finalist for {stuck_min:.0f} min - finishing early "
                    f"(after {restarts} fresh restarts)")
                break
            if restart_after > 0 and gen - last_gain_gen >= restart_after:
                # the population has converged on one idea: keep what we found (hall of fame)
                # and start a brand-new random population to explore somewhere else
                restarts += 1
                log(f"  [{label}] stuck for {restart_after} generations - restart #{restarts} with fresh ideas "
                    f"(best so far is kept)")
                epoch += 1                      # results still in flight from the old population
                pop = [fresh() for _ in range(pop_n)]   # only go into the hall of fame
                scored = []
                best_seen, last_gain_gen = -1e9, gen
                send(pop)
                continue
            # ---------- breed the next generation ----------
            pop = breed()
            send(pop)
            if pipelined and len(pending) < 2:
                pop = breed()
                send(pop)
    except KeyboardInterrupt:
        log(f"  [{label}] stopped by you (Ctrl+C) after {trials:,} strategies - "
            f"checking the best ones found so far ...")
        log("  (if Windows then asks 'Terminate batch job (Y/N)?': N = go on and start the bot, Y = close)")
        if stats is not None:
            stats["interrupted"] = True
    # results still in flight when the search stopped: keep any good ones
    while pending:
        ep, batch, bkeys, h = pending.popleft()
        results = h.get()
        trials += len(batch)
        take(ep, batch, bkeys, results, into_scored=False)
    ranked = sorted(hof.values(), key=lambda x: -x[0])
    if stats is not None:
        stats.update(restarts=restarts, generations=gen, seconds=round(time.time() - t0, 1))
    return ranked, trials, srs, families


def pick_finalists(ranked, n, per_family=2, min_trades=0):
    out, fam = [], {}
    for fit, g, r in ranked:
        if fit <= 0:
            break
        if r.get("trades", 0) < min_trades:      # would fail 'enough trades' anyway - don't waste a place
            continue
        f = family_signature(g)
        if fam.get(f, 0) >= per_family:
            continue
        fam[f] = fam.get(f, 0) + 1
        out.append((fit, g, r))
        if len(out) >= n:
            break
    return out


def estimate_independence(ctx, rng, timeframes, names, allow_short=False, n=240):
    """What fraction of randomly made strategies are genuinely different ideas?
    Correlation of their daily results -> 'effective number' (participation ratio of the
    correlation matrix, as used for multiple-testing corrections) / number sampled."""
    rows = []
    for _ in range(n * 3):
        if len(rows) >= n:
            break
        g = random_genome(rng, timeframes, allow_short, names)
        s0, re, _ = ctx.split[g["tf"]]
        try:
            _, tr, daily, _ = ctx.run(g, s0, re)
        except Exception:
            continue
        if len(tr) < 20 or np.std(daily) == 0:
            continue
        rows.append((g["tf"], daily, ctx.bench_daily(g["tf"], s0, re)))
    if len(rows) < 30:
        return None, 0
    # align on the shorter day count per timeframe (both cover the same days)
    L = min(min(len(d), len(b)) for _, d, b in rows)
    X = []
    for _, d, b in rows:
        d, b = d[:L], b[:L]
        # remove the part every long day-trade shares (the market's own move that day), so two
        # strategies only count as 'the same idea' if they are alike beyond that
        bb = b - b.mean()
        beta = float((bb * (d - d.mean())).sum() / max((bb * bb).sum(), 1e-18))
        X.append(d - beta * b)
    X = np.array(X)
    C = np.corrcoef(X)
    C = np.nan_to_num(C)
    ev = np.linalg.eigvalsh(C)
    pr = (ev.sum() ** 2) / max((ev ** 2).sum(), 1e-12)
    return float(min(1.0, pr / len(rows))), len(rows)


def pick_diverse(ctx, ranked, n, min_trades=0, max_corr=0.6, scan=400):
    """Finalists that are genuinely different strategies: skip any whose daily results are
    strongly correlated with one already picked (clones with a do-nothing extra condition)."""
    out, kept = [], []
    for fit, g, r in ranked[:scan]:
        if fit <= 0:
            break
        if r.get("trades", 0) < min_trades:
            continue
        s0, re, _ = ctx.split[g["tf"]]
        _, _, d, _ = ctx.run(g, s0, re)
        if np.std(d) == 0:
            continue
        if any(len(k) == len(d) and abs(np.corrcoef(k, d)[0, 1]) > max_corr for k in kept):
            continue
        kept.append(d)
        out.append((fit, g, r))
        if len(out) >= n:
            break
    return out


def _ck_file():
    from .config import DATA_DIR
    return DATA_DIR / "search_checkpoint.pkl"


def _save_checkpoint(state):
    import pickle
    f = _ck_file()
    tmp = f.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, f)                          # atomic: never a half-written file


def _clear_checkpoint():
    try:
        _ck_file().unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def _resume_checkpoint(registry, symbols, tfs, log=print):
    """Best strategies from a search that never finished (or [] if there is none)."""
    import pickle
    f = _ck_file()
    if not f.exists():
        return []
    try:
        with open(f, "rb") as fh:
            ck = pickle.load(fh)
    except Exception:
        _clear_checkpoint()
        return []
    if ck.get("symbols") != symbols or sorted(ck.get("tfs") or []) != sorted(tfs):
        _clear_checkpoint()                     # it was for a different set of symbols
        return []
    hof = sorted(ck.get("hof") or [], key=lambda x: -x[0])
    registry.add_trials(int(ck.get("trials") or 0), ck.get("srs") or [], ck.get("families") or set())
    _clear_checkpoint()
    log(f"  resuming a search that was cut off: {int(ck.get('trials') or 0):,} strategies it tested are counted, "
        f"and its {len(hof)} best are tested again first")
    return [g for _, g, _ in hof]


def _thin(curve, n):
    """Keep at most n points of a progress curve (always keeps the last one)."""
    if len(curve) <= n:
        return curve
    idx = np.unique(np.linspace(0, len(curve) - 1, n).astype(int))
    return [curve[i] for i in idx]


def discover(cfg, panels, registry, minutes=None, max_trials=None, seed=None, log=print, workers=None,
             extras=None, fomc_dates=(), cost=None):
    """Full discovery run: evolve -> gauntlet -> holdout exam -> register survivors."""
    rng = np.random.default_rng(seed)
    d = cfg["discovery"]
    log(f"Discovery starting: timeframes {list(panels)}, "
        f"{sum(len(p.symbols) for p in panels.values())} symbol-series, "
        f"budget {'%s min' % minutes if minutes else ''}{' %s trials' % max_trials if max_trials else ''}")
    import pandas as pd
    syms = list(next(iter(panels.values())).symbols) if panels else []
    run = {"started": pd.Timestamp.now(tz="UTC").isoformat(), "symbols": syms,
           "universe": "QQQ only" if syms == ["QQQ"] else f"{len(syms)} symbols",
           "budget_min": minutes, "cores": None}
    # a search that was cut off (crash, window closed, power) left its best-so-far list on disk:
    # count its tests (honest luck test) and start from its best strategies
    seeds = _resume_checkpoint(registry, syms, list(panels), log)
    engine = Engine(panels, cfg, workers=workers, log=log, extras=extras, fomc_dates=fomc_dates, cost=cost)
    run["cores"] = engine.n
    stats = {}

    def save_ck(state):
        _save_checkpoint(dict(state, symbols=syms, tfs=list(panels), saved=time.time()))

    try:
        ranked, trials, srs, fams = evolve(engine, rng, minutes=minutes, max_trials=max_trials,
                                           timeframes=list(panels), allow_short=cfg["market"]["allow_short"],
                                           log=log, skip_keys=registry.known_keys(), stats=stats,
                                           seeds=seeds or None, checkpoint=save_ck,
                                           checkpoint_every=float(d.get("checkpoint_seconds", 300)))
        total = registry.add_trials(trials, srs, fams)
        if d.get("measure_independence", False):
            # off by default: on pure-noise test markets it lowered the luck bar enough to let
            # a lucky strategy through, so the stricter count (every family = one idea) is kept
            try:
                ind, ns = estimate_independence(engine.ctx, rng, list(panels), engine.available_features(),
                                                cfg["market"]["allow_short"])
                if ind is not None:
                    ind = registry.update_independence(ind, ns)
                    run["independence"] = round(ind, 4)
            except Exception as e:
                log(f"  (could not measure idea independence: {e})")
        _clear_checkpoint()                    # its tests are counted now; the checks run next
        if stats.get("interrupted"):
            run["status"] = "stopped early"
        n_eff = registry.effective_trials()
        sr_var = registry.sr_variance()
        log(f"  tested {trials:,} new strategies this run ({total:,} in the AI's lifetime, "
            f"{n_eff:,} genuinely different ideas)")
        finalists = pick_diverse(engine.ctx, ranked, int(d["finalists"]), min_trades=int(d["min_trades"]),
                                 max_corr=float(d.get("finalist_max_corr", 0.6)))
        log(f"  {len(finalists)} finalists enter the validation gauntlet ...")
        run.update(trials=int(trials), lifetime=int(total), ideas=int(n_eff), finalists=len(finalists),
                   restarts=stats.get("restarts", 0), seconds=stats.get("seconds"),
                   rate=round(trials / max(stats.get("seconds") or 1, 1), 1),
                   best_score=round(float(ranked[0][0]), 3) if ranked else None,
                   curve=_thin(stats.get("curve", []), 80))
        passed = []
        fails = {}
        prob_ok = bool(cfg["validation"].get("probation", True))
        for fit, g, r in finalists:
            ok, rep = gauntlet(engine.ctx, g, n_eff, sr_var, rng, use_holdout=False, probation_ok=prob_ok)
            if ok:
                passed.append((fit, g, rep))
            else:
                failed = [k for k, v in rep["checks"].items() if not v["pass"]]
                fails[failed[0] if failed else "?"] = fails.get(failed[0] if failed else "?", 0) + 1
        if fails:
            log("  eliminated: " + ", ".join(f"{k}={v}" for k, v in sorted(fails.items(), key=lambda x: -x[1])))
        log(f"  {len(passed)} survived research checks")
        run.update(eliminated=fails, survived=len(passed), exam=[])
        # the one-time holdout exam: one strategy per idea (twins that trade identically
        # would only waste exam places)
        exam, fam_seen, twin_seen = [], set(), []
        for fit, g, rep in passed:
            fam = family_signature(g)
            r = rep.get("research", {})
            twin = (g["tf"], r.get("trades") or 0, r.get("sharpe") or 0.0)
            if fam in fam_seen or any(t[0] == twin[0] and abs(t[1] - twin[1]) <= 0.03 * max(t[1], 1)
                                      and abs(t[2] - twin[2]) < 0.1 for t in twin_seen):
                continue
            fam_seen.add(fam)
            twin_seen.append(twin)
            exam.append((fit, g, rep))
        log(f"  {min(len(exam), int(d['holdout_slots']))} different ideas take the one-time holdout exam")
        winners = []
        for fit, g, _ in exam[: int(d["holdout_slots"])]:
            ok, rep = gauntlet(engine.ctx, g, n_eff, sr_var, rng, use_holdout=True, probation_ok=prob_ok)
            h = rep.get("holdout", {})
            log(f"   {('PASS (probation)' if rep.get('probation') else 'PASS') if ok else 'fail'}  research Sharpe {rep['research']['sharpe']:.2f}, "
                f"PF {rep['research']['profit_factor']:.2f} | holdout Sharpe {h.get('sharpe', 0):.2f}, "
                f"return {h.get('total_return', 0):+.1%}, {h.get('trades', 0)} trades")
            run["exam"].append({"pass": bool(ok), "probation": bool(ok and rep.get("probation")), "research_sharpe": rep["research"]["sharpe"],
                                "pf": rep["research"]["profit_factor"], "holdout_sharpe": h.get("sharpe", 0),
                                "holdout_return": h.get("total_return", 0), "holdout_trades": h.get("trades", 0),
                                "tf": g["tf"]})
            if ok:
                winners.append((fit, g, rep))
        new_ids = []
        for fit, g, rep in winners:
            prob = bool(rep.get("probation"))
            sid = registry.add_strategy(g, rep, "reserve", fitness=fit, note="discovered (probation)" if prob else "discovered")
            registry.event(sid, "discovered",
                           (f"Passed every check except the luck test after {total:,} lifetime trials - starts on "
                            f"PROBATION (1/4 size) until live paper trades confirm it. " if prob else
                            f"Passed all checks after {total:,} lifetime trials. ")
                           + describe(g).replace("\n", " | "))
            new_ids.append(sid)
        log(f"Discovery finished: {len(new_ids)} new validated strategies.")
        run.update(validated=len(new_ids), finished=pd.Timestamp.now(tz="UTC").isoformat())
        try:
            registry.add_search_run(run)
        except Exception:
            pass
        return new_ids
    finally:
        engine.close()
