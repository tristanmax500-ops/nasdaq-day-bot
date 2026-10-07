"""Strategy DNA for day trading.

ENTRY  = all conditions true at a candle's close -> enter at the next candle's open
EXIT   = take-profit (always), stop-loss (always), optional trailing stop,
         optional exit condition, a time limit, and ALWAYS before the close.
"""
import copy
import hashlib
import json

import numpy as np

from .features import FEATURES, feature_label

HOLD_CHOICES = [2, 3, 4, 6, 8, 12, 18, 24, 36, 48, 78]
PER_DAY_CHOICES = [1, 2, 3, 5]
OPS = ["<", ">", "xa", "xb"]
BINARY = {"macro", "fomc", "earn", "earn_next"}


def _round(x, name):
    lo, hi = FEATURES[name][1], FEATURES[name][2]
    if name in BINARY:
        return 0.5
    step = (hi - lo) / 200.0
    return float(np.round(np.clip(x, lo, hi) / step) * step)


def random_cond(rng, names=None):
    names = names or list(FEATURES)
    name = names[rng.integers(len(names))]
    params, lo, hi, _ = FEATURES[name]
    p = params[rng.integers(len(params))]
    if name in BINARY:
        op = "<" if rng.random() < 0.5 else ">"
    else:
        op = OPS[rng.choice(4, p=[0.38, 0.38, 0.12, 0.12])]
    return {"f": name, "p": p, "op": op, "v": _round(rng.uniform(lo, hi), name)}


def random_genome(rng, timeframes, allow_short=False, names=None):
    n = rng.choice([1, 2, 3], p=[0.3, 0.45, 0.25])
    return {
        "tf": timeframes[rng.integers(len(timeframes))],
        "dir": "short" if (allow_short and rng.random() < 0.4) else "long",
        "entry": [random_cond(rng, names) for _ in range(n)],
        "exit": random_cond(rng, names) if rng.random() < 0.3 else None,
        "stop": float(np.round(rng.uniform(0.5, 4.0), 2)),
        "tp": float(np.round(rng.uniform(0.5, 8.0), 2)),
        "trail": float(np.round(rng.uniform(1.0, 5.0), 2)) if rng.random() < 0.2 else 0.0,
        "hold": int(HOLD_CHOICES[rng.integers(len(HOLD_CHOICES))]),
        "per_day": int(PER_DAY_CHOICES[rng.integers(len(PER_DAY_CHOICES))]),
    }


def _nudge(choices, x, rng):
    i = choices.index(x) if x in choices else len(choices) // 2
    return choices[int(np.clip(i + rng.choice([-1, 1]), 0, len(choices) - 1))]


def _copy(x):
    """Fast copy of a genome or a part of one (a genome is one level of dicts of plain values,
    plus the entry list and the exit condition)."""
    if isinstance(x, list):
        return [dict(c) for c in x]
    if isinstance(x, dict):
        if "entry" in x:
            g = dict(x)
            g["entry"] = [dict(c) for c in x["entry"]]
            if x.get("exit"):
                g["exit"] = dict(x["exit"])
            return g
        return dict(x)
    return x


def mutate(g, rng, strength=1.0, timeframes=None, allow_short=False, names=None):
    g = _copy(g)
    for _ in range(1 + int(rng.random() < 0.35)):
        r = rng.random()
        conds = g["entry"] + ([g["exit"]] if g["exit"] else [])
        if r < 0.36:
            c = conds[rng.integers(len(conds))]
            lo, hi = FEATURES[c["f"]][1], FEATURES[c["f"]][2]
            c["v"] = _round(c["v"] + rng.normal(0, 0.08 * strength) * (hi - lo), c["f"])
        elif r < 0.46:
            c = conds[rng.integers(len(conds))]
            c["p"] = _nudge(FEATURES[c["f"]][0], c["p"], rng)
        elif r < 0.51:
            c = conds[rng.integers(len(conds))]
            c["op"] = {"<": ">", ">": "<", "xa": "xb", "xb": "xa"}[c["op"]]
        elif r < 0.59:
            g["entry"][rng.integers(len(g["entry"]))] = random_cond(rng, names)
        elif r < 0.66:
            if len(g["entry"]) < 3 and (len(g["entry"]) == 1 or rng.random() < 0.5):
                g["entry"].append(random_cond(rng, names))
            elif len(g["entry"]) > 1:
                g["entry"].pop(rng.integers(len(g["entry"])))
        elif r < 0.70:
            g["exit"] = None if (g["exit"] and rng.random() < 0.5) else random_cond(rng, names)
        elif r < 0.78:
            g["stop"] = float(np.round(np.clip(g["stop"] * np.exp(rng.normal(0, 0.25 * strength)), 0.3, 6), 2))
        elif r < 0.86:
            g["tp"] = float(np.round(np.clip(g["tp"] * np.exp(rng.normal(0, 0.25 * strength)), 0.3, 12), 2))
        elif r < 0.89:
            g["trail"] = 0.0 if g["trail"] else float(np.round(rng.uniform(1.0, 5.0), 2))
        elif r < 0.94:
            g["hold"] = _nudge(HOLD_CHOICES, g["hold"], rng)
        elif r < 0.97:
            g["per_day"] = _nudge(PER_DAY_CHOICES, g["per_day"], rng)
        else:
            if timeframes and len(timeframes) > 1 and rng.random() < 0.7:
                g["tf"] = timeframes[rng.integers(len(timeframes))]
            elif allow_short:
                g["dir"] = "short" if g["dir"] == "long" else "long"
    return g


def crossover(a, b, rng):
    base, donor = (a, b) if rng.random() < 0.5 else (b, a)
    child = _copy(base)
    pool = _copy(a["entry"] + b["entry"])
    order = rng.permutation(len(pool))
    k = int(np.clip(rng.integers(1, 4), 1, len(pool)))
    child["entry"] = [pool[i] for i in order[:k]]
    for key_ in ("stop", "tp", "trail", "hold", "exit", "per_day"):
        if rng.random() < 0.5:
            child[key_] = _copy(donor[key_])
    return child


def key(g):
    return hashlib.sha1(json.dumps(g, sort_keys=True).encode()).hexdigest()[:16]


def family_signature(g):
    feats = sorted(f"{c['f']}{c['op'][0]}" for c in g["entry"])
    return g["tf"] + g["dir"] + "|".join(feats)


_OPS_TXT = {"<": "is below", ">": "is above", "xa": "crosses above", "xb": "crosses below"}


def describe_cond(c):
    if c["f"] in BINARY:
        return f"{feature_label(c['f'], c['p']).replace(' (0/1)', '')}: {'YES' if c['op'] == '>' else 'NO'}"
    return f"{feature_label(c['f'], c['p'])} {_OPS_TXT[c['op']]} {c['v']:g}"


def describe(g):
    side = "BUY" if g["dir"] == "long" else "SELL SHORT"
    lines = [f"{side} on {g['tf'].replace('Min', '-min')} candles when: "
             + "  AND  ".join(describe_cond(c) for c in g["entry"])]
    ex = [f"take-profit {g['tp']:g} ATR", f"stop-loss {g['stop']:g} ATR"]
    if g["trail"]:
        ex.append(f"trailing stop {g['trail']:g} ATR")
    if g["exit"]:
        ex.append("when " + describe_cond(g["exit"]))
    ex.append(f"after {g['hold']} candles")
    ex.append("always before the close")
    lines.append("EXIT: " + ", ".join(ex) + f"   (max {g['per_day']} trades per stock per day)")
    return "\n".join(lines)
