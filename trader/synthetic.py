"""Fake intraday markets where we KNOW the truth - used by the self-test."""
import datetime as dt

import numpy as np
import pandas as pd

from .data import Panel


def make_intraday(seed, n_sym=8, n_days=1100, tf_min=15, edge=True, change_from_day=None,
                  event_prob=0.2):
    """Random-walk stocks built minute by minute (so stops/targets behave realistically).

    Planted edge (if edge=True): after a sharp 2-candle drop (after the first 30 min),
    the stock drifts back up over the next 2 candles.
    From `change_from_day` on, the bounce FAILS on event days (macro=1) - it keeps
    falling instead. That is the 'market changed' scenario the bot must detect and fix.
    """
    rng = np.random.default_rng(seed)
    bpd = 390 // tf_min
    T = n_days * bpd
    sig_min = 0.0125 / np.sqrt(390)
    beta = rng.uniform(0.7, 1.3, n_sym)
    days = pd.bdate_range("2021-01-04", periods=n_days)
    is_event = rng.random(n_days) < event_prob
    o = np.zeros((T, n_sym)); h = np.zeros((T, n_sym)); l = np.zeros((T, n_sym)); c = np.zeros((T, n_sym))
    v = np.zeros((T, n_sym))
    price = np.log(rng.uniform(50, 400, n_sym))
    sig_bar = sig_min * np.sqrt(tf_min)
    drift_left = np.zeros(n_sym)
    drift_sign = np.ones(n_sym)
    ushape = 1.0 + 1.5 * (np.linspace(-1, 1, bpd) ** 2)
    t = 0
    hist = []
    for d in range(n_days):
        price = price + rng.normal(0, 0.004, n_sym)                 # overnight gap
        drift_left[:] = 0
        for b in range(bpd):
            m = rng.normal(0.0003 / 390, sig_min * 0.7, tf_min)
            steps = m[:, None] * beta[None, :] + rng.normal(0, sig_min * 0.7, (tf_min, n_sym))
            dr = np.where(drift_left > 0, drift_sign * 0.0009 / tf_min, 0.0)
            steps = steps + dr[None, :]
            path = price[None, :] + np.cumsum(steps, axis=0)
            o[t] = np.exp(price)
            c[t] = np.exp(path[-1])
            h[t] = np.maximum(np.exp(path.max(axis=0)), o[t])
            l[t] = np.minimum(np.exp(path.min(axis=0)), o[t])
            v[t] = rng.lognormal(12, 0.4, n_sym) * ushape[b]
            price = path[-1]
            drift_left = np.maximum(drift_left - 1, 0)
            hist.append(price.copy())
            if edge and b >= 2 and b < bpd - 3:
                drop = hist[-1] - hist[-3]
                hit = (drop < -2.0 * sig_bar * np.sqrt(2)) & (drift_left == 0)
                if hit.any():
                    flip = change_from_day is not None and d >= change_from_day and is_event[d]
                    drift_left[hit] = 2
                    drift_sign[hit] = -1.0 if flip else 1.0
            t += 1
    day = np.repeat(np.arange(n_days), bpd)
    mins = np.tile(np.arange(1, bpd + 1) * tf_min, n_days)
    eod = np.tile(np.r_[np.zeros(bpd - 1, bool), True], n_days)
    idx = []
    for d in days:
        op = pd.Timestamp(d.date()).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
        idx.extend(op + pd.Timedelta(minutes=tf_min * (k + 1)) for k in range(bpd))
    index = pd.DatetimeIndex(idx).tz_convert("UTC")
    syms = ["QQQ"] + ["SYN%d" % i for i in range(1, n_sym)]
    panel = Panel(f"{tf_min}Min", index, syms, o, h, l, c, v, day.astype(np.int64), mins.astype(np.int64),
                  eod, [x.date() for x in days], np.full(n_days, 390), benchmark="QQQ")
    extras = {
        "macro": np.repeat(is_event.astype(float)[day][:, None], n_sym, axis=1),
        "fomc": np.zeros((T, n_sym)),
        "earn": np.zeros((T, n_sym)),
        "earn_next": np.zeros((T, n_sym)),
        "sent": np.tanh(rng.normal(0, 0.5, (T, n_sym))),
        "newsv": rng.normal(0, 0.7, (T, n_sym)),
        "mkt_sent": np.repeat(np.tanh(rng.normal(0, 0.3, (T, 1))), n_sym, axis=1),
    }
    return panel, extras


def slice_extras(extras, n):
    return {k: v[:n] for k, v in extras.items()}
