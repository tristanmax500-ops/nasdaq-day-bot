"""Known intraday edges, written as strategy DNA.

Instead of millions of random indicator combinations, these are the handful of day-trading
setups that traders and published research keep coming back to. Each one is tried over a
small grid of sensible settings (a few thousand variants in total), so the luck test only has
to account for a few thousand tries instead of tens of millions - and the ideas started from
a reason, not from noise.

  orb        opening-range breakout (first 15/30/60 minutes)
  vwap_rev   stretched far from VWAP + short-term oversold/overbought -> snap back to VWAP
  vwap_trend trending day, price comes back to VWAP and crosses it again with the trend
  gap_fade   big opening gap, fade it in the first hour
  gap_go     big opening gap that breaks its opening range -> go with it
  late_mom   intraday momentum: the day's direction so far continues into the last half hour
  pd_break   breaks yesterday's high/low
  sweep_rev  liquidity sweep of yesterday's low/high, closes back inside -> reversal
  open_drive strong first hour -> continuation into midday
"""
import itertools

TEMPLATE_NAMES = ["orb", "vwap_rev", "vwap_trend", "gap_fade", "gap_go", "late_mom", "pd_break",
                  "sweep_rev", "open_drive"]


def _g(tf, d, entry, stop, tp, hold, per_day, exit_=None, trail=0.0, name=""):
    return {"tf": tf, "dir": d, "entry": entry, "exit": exit_, "stop": float(stop), "tp": float(tp),
            "trail": float(trail), "hold": int(hold), "per_day": int(per_day), "template": name}


def C(f, op, v, p=1):
    return {"f": f, "p": p, "op": op, "v": float(v)}


def _sides(allow_short):
    return [("long", 1)] + ([("short", -1)] if allow_short else [])


def generate(timeframes, allow_short=True, names=None):
    """Every template variant, as a list of genomes (each carries g['template'])."""
    names = names or TEMPLATE_NAMES
    out = []
    for tf in timeframes:
        bars_hr = {"5Min": 12, "15Min": 4}.get(tf, 12)
        holds_short = [bars_hr, 2 * bars_hr]           # 1-2 hours
        hold_day = 78 if tf == "5Min" else 26          # until the close
        for d, s in _sides(allow_short):
            up = s > 0
            if "orb" in names:
                for p, until, stop, tp, hold in itertools.product([15, 30, 60], [120, 390], [1.0, 1.5, 2.0],
                                                                  [1.5, 2.5, 4.0], [holds_short[1], hold_day]):
                    ent = [C("orb_hi" if up else "orb_lo", "xa" if up else "xb", 0.0, p), C("tod", "<", until)]
                    out.append(_g(tf, d, ent, stop, tp, hold, 1, name="orb"))
            if "vwap_rev" in names:
                for k, rp, rv, stop, tp in itertools.product([1.0, 1.5, 2.0, 2.5], [2, 3], [10, 20],
                                                             [1.0, 1.5, 2.5], [1.0, 2.0, 3.0]):
                    ent = [C("vwap", "<" if up else ">", -k if up else k),
                           C("rsi", "<" if up else ">", rv if up else 100 - rv, rp)]
                    out.append(_g(tf, d, ent, stop, tp, holds_short[1], 3,
                                  exit_=C("vwap", ">" if up else "<", 0.0), name="vwap_rev"))
            if "vwap_trend" in names:
                for dc, stop, tp, hold in itertools.product([0.3, 0.6, 1.0], [1.0, 1.5, 2.0], [1.5, 2.5, 4.0],
                                                            [holds_short[1], hold_day]):
                    ent = [C("dchg", ">" if up else "<", dc if up else -dc),
                           C("vwap", "xa" if up else "xb", 0.0), C("tod", ">", 30)]
                    out.append(_g(tf, d, ent, stop, tp, hold, 2, name="vwap_trend"))
            if "gap_fade" in names:
                for gp, until, stop, tp, hold in itertools.product([0.3, 0.6, 1.0], [30, 60], [1.0, 1.5, 2.5],
                                                                  [1.0, 2.0, 3.0], [holds_short[1], hold_day]):
                    # long fades a gap DOWN, short fades a gap UP
                    ent = [C("gap", "<" if up else ">", -gp if up else gp), C("tod", "<", until)]
                    out.append(_g(tf, d, ent, stop, tp, hold, 1, name="gap_fade"))
            if "gap_go" in names:
                for gp, p, stop, tp, hold in itertools.product([0.3, 0.6, 1.0], [15, 30], [1.0, 1.5, 2.0],
                                                              [1.5, 2.5, 4.0], [holds_short[1], hold_day]):
                    ent = [C("gap", ">" if up else "<", gp if up else -gp),
                           C("orb_hi" if up else "orb_lo", "xa" if up else "xb", 0.0, p)]
                    out.append(_g(tf, d, ent, stop, tp, hold, 1, name="gap_go"))
            if "late_mom" in names:
                for after, dc, stop, tp in itertools.product([300, 330, 350], [0.2, 0.4, 0.7, 1.0],
                                                            [1.0, 1.5, 2.5], [1.5, 3.0, 6.0]):
                    ent = [C("tod", ">", after), C("dchg", ">" if up else "<", dc if up else -dc)]
                    out.append(_g(tf, d, ent, stop, tp, hold_day, 1, name="late_mom"))
            if "pd_break" in names:
                for until, stop, tp, hold in itertools.product([120, 390], [1.0, 1.5, 2.0], [1.5, 2.5, 4.0],
                                                               [holds_short[1], hold_day]):
                    ent = [C("pdh" if up else "pdl", "xa" if up else "xb", 0.0), C("tod", "<", until)]
                    out.append(_g(tf, d, ent, stop, tp, hold, 1, name="pd_break"))
            if "sweep_rev" in names:
                for p, depth, stop, tp, hold in itertools.product([1, 12, 48], [0.0, 0.1], [0.75, 1.0, 1.5],
                                                                  [1.0, 2.0, 3.0], [holds_short[0], holds_short[1]]):
                    ent = [C("sweep_lo" if up else "sweep_hi", ">", depth, p)]
                    out.append(_g(tf, d, ent, stop, tp, hold, 2, name="sweep_rev"))
            if "open_drive" in names:
                for dc, stop, tp, hold in itertools.product([0.4, 0.7, 1.0], [1.0, 1.5, 2.0], [1.5, 2.5, 4.0],
                                                            [holds_short[1], hold_day]):
                    ent = [C("tod", ">", 55), C("tod", "<", 75), C("dchg", ">" if up else "<", dc if up else -dc)]
                    out.append(_g(tf, d, ent, stop, tp, hold, 1, name="open_drive"))
    return out
