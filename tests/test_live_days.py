"""Runs the real bot against a fake Alpaca for several full trading days and checks
every safety rule. Run:  python -m tests.test_live_days"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.fake_alpaca import FakeAlpaca, frames_from_panel, calendar_rows, synthetic_articles  # noqa
import trader.data as D  # noqa
import trader.events as E  # noqa
from trader.config import load_config  # noqa
from trader.registry import Registry  # noqa
from trader.news import NewsStore  # noqa
from trader.app import Bot  # noqa
from trader.synthetic import make_intraday  # noqa

NY = "America/New_York"


def main():
    tmp = Path(tempfile.mkdtemp(prefix="bot_live_test_"))
    D.CACHE = tmp
    E.CACHE_FILE = tmp / "events.json"
    panel, _ = make_intraday(5, n_sym=6, n_days=330, tf_min=5, edge=True)
    rng = np.random.default_rng(0)
    fake = FakeAlpaca(frames_from_panel(panel), calendar_rows(panel), synthetic_articles(panel, rng))
    cfg = load_config()
    cfg["db_path"] = str(tmp / "brain.db")
    cfg["market"].update(symbols=list(panel.symbols), benchmark="QQQ", timeframes=["5Min", "15Min"],
                         history_start=panel.day_dates[0].isoformat(), qqq_heavyweights=["SYN1", "SYN2"])
    cfg["news"]["history_start"] = panel.day_dates[0].isoformat()
    cfg["paper"].update(broker="alpaca", target_champions=2)
    cfg["auto"]["discovery_minutes_after_close"] = 0.2
    cfg["healing"]["repair_trials"] = 300
    cfg["discovery"]["workers"] = 1
    cfg["dashboard"]["status_folder"] = str(tmp)
    reg = Registry(cfg["db_path"], clock=lambda: fake.now.isoformat())
    bot = Bot(cfg=cfg, api=fake, registry=reg, news=NewsStore(tmp / "news.db"))
    bot.provider.now = lambda: fake.now
    # fake earnings calendar (no internet in tests)
    def fake_earn(self):
        self.earnings = {"SYN1": [(panel.day_dates[310], "amc")], "SYN3": [(panel.day_dates[312], "bmo")]}
        self.have_earnings = True
    E.EventCalendar._earnings = fake_earn

    day0 = 300
    o0 = pd.Timestamp(panel.day_dates[day0]).tz_localize(NY) + pd.Timedelta(hours=8)
    fake.advance(o0.tz_convert("UTC"))
    bot.download()
    p5 = bot.provider.panel("5Min")
    p15 = bot.provider.panel("15Min")
    print(f"downloaded: 5Min {p5.T} candles, 15Min {p15.T} candles, days {p5.n_days}")
    assert p5.T == 300 * 78, p5.T
    assert p15.T == 300 * 26
    # 15-min candles must equal aggregated 5-min candles
    j = 2
    assert np.allclose(p15.o[:, j], p5.o[::3, j]) and np.allclose(p15.c[:, j], p5.c[2::3, j])
    assert np.allclose(p15.h[:, j], np.maximum.reduce([p5.h[0::3, j], p5.h[1::3, j], p5.h[2::3, j]]))
    assert np.allclose(p15.v[:, j], p5.v[0::3, j] + p5.v[1::3, j] + p5.v[2::3, j])
    assert p5.eod.sum() == 300 and p5.mins[0] == 5 and p5.mins[77] == 390
    M = bot.market()
    for tf in M.extras:
        for k, v in M.extras[tf].items():
            fin = np.isfinite(v).mean()
            print(f"   feature {tf:<5} {k:<9} available {fin:.0%}")
    ex = M.extras["5Min"]
    d310 = list(p5.day_dates).index(panel.day_dates[310]) if panel.day_dates[310] in p5.day_dates else None

    # two hand-made champions that trade often (so the broker logic gets exercised)
    base_rep = {"checks": {}, "research": {}, "expected_trade_R": [0.05] * 200, "expected_max_dd": -0.03,
                "expected_exit_mix": {"stop-loss": 0.3, "take-profit": 0.3}, "research_regimes": {}}
    g1 = {"tf": "5Min", "dir": "long", "entry": [{"f": "mom", "p": 2, "op": "<", "v": -1.2}], "exit": None,
          "stop": 1.5, "tp": 2.0, "trail": 0.0, "hold": 12, "per_day": 3}
    g2 = {"tf": "15Min", "dir": "long", "entry": [{"f": "rsi", "p": 3, "op": "<", "v": 25.0}],
          "exit": {"f": "rsi", "p": 3, "op": ">", "v": 70.0}, "stop": 1.0, "tp": 3.0, "trail": 2.0, "hold": 8,
          "per_day": 2}
    reg.add_strategy(g1, base_rep, "champion", note="test")
    reg.add_strategy(g2, base_rep, "champion", note="test")

    problems = []
    n_steps = 0
    for d in range(day0, day0 + 4):
        op = pd.Timestamp(panel.day_dates[d]).tz_localize(NY) + pd.Timedelta(hours=9, minutes=30)
        cl = op + pd.Timedelta(hours=6, minutes=30)
        t = op + pd.Timedelta(seconds=10)
        while t < cl + pd.Timedelta(minutes=1):
            fake.advance(t.tz_convert("UTC"))
            if fake.clock()["is_open"]:
                bot.bar_cycle()
                n_steps += 1
            mins_left = (cl - t).total_seconds() / 60
            if mins_left <= 5 and fake.pos:
                problems.append(f"position open {mins_left:.0f} min before close: {fake.pos}")
            for o in fake.orders_:
                if o.get("order_class") == "bracket" and o.get("legs"):
                    tp, sl = o["legs"][0]["limit_price"], o["legs"][1]["stop_price"]
                    if not sl < tp:
                        problems.append(f"bad bracket {o}")
            # next wake: next 5-min boundary +10s, or the flatten time
            nxt = (t.floor("5min") + pd.Timedelta(minutes=5, seconds=10))
            flat = cl - pd.Timedelta(minutes=5) + pd.Timedelta(seconds=5)
            t = flat if t < flat < nxt else nxt
        fake.advance((cl + pd.Timedelta(minutes=30)).tz_convert("UTC"))
        if fake.pos:
            problems.append(f"overnight position after day {d}: {fake.pos}")
        bot.after_close()
        print(f"day {d}: equity ${fake.equity():,.0f}, brackets so far "
              f"{sum(1 for c in fake.calls if c[0] == 'bracket')}, closes {sum(1 for c in fake.calls if c[0] == 'close')}")
    # the Projects Dashboard feed
    js = (tmp / "nasdaq_bot_status.js").read_text()
    assert js.startswith("window.NASDAQ_BOT = ")
    st = json.loads(js[len("window.NASDAQ_BOT = "):].rstrip().rstrip(";"))
    print(f"dashboard feed: state={st['state']}, equity={st['account']['equity']}, strategies={len(st['strategies'])}, "
          f"events={len(st['events'])}, orders={len(st['recent_orders'])}, history days={len(st['equity_history'])}")
    assert st["account"]["equity"] > 0 and st["strategies"] and st["events"]
    cids = [o.get("client_order_id") for o in fake.orders_ if o.get("client_order_id")]
    if len(cids) != len(set(cids)):
        problems.append("duplicate client order ids")
    sl = reg.recent_slippage()
    print(f"bar cycles: {n_steps}, orders: {len(cids)}, fills measured: {len(sl)}, "
          f"median slippage {np.median(sl) if sl else float('nan'):.2f} bps")
    tv = list((Path(__file__).resolve().parent.parent / "tradingview").glob("bot_*.pine"))
    print("events:", [e["kind"] for e in reg.events(400)][:40])
    print("PROBLEMS:", problems if problems else "none")
    assert not problems
    assert len(cids) > 0, "no orders were placed"
    print("LIVE-DAY TEST PASSED")


if __name__ == "__main__":
    main()
