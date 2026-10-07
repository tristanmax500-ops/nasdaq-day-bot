"""Focused correctness tests. Run:  python -m tests.test_units"""
import datetime as dt
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trader.config import load_config  # noqa
from trader.data import build_panel, sessions_from_calendar  # noqa
from trader.news import NewsStore, score_text  # noqa
from trader.events import EventCalendar  # noqa
from trader import backtest as bt  # noqa
from trader.features import FeatureBook  # noqa

NY = "America/New_York"
PASSED = []


def ok(name):
    PASSED.append(name)
    print("  PASS", name)


def make_frames(dates, closes_by_day=None, half_days=(), syms=("QQQ", "AAA")):
    rows = []
    for d in dates:
        rows.append({"date": d, "open": "09:30", "close": "13:00" if d in half_days else "16:00"})
    sess = sessions_from_calendar(rows)
    frames = {}
    rng = np.random.default_rng(1)
    for s in syms:
        idx, vals = [], []
        for d in dates:
            o, c = sess.loc[pd.Timestamp(d).date(), "open"], sess.loc[pd.Timestamp(d).date(), "close"]
            # include some pre/post market bars that must be dropped
            for t in pd.date_range(o - pd.Timedelta(minutes=30), c + pd.Timedelta(minutes=30), freq="5min", inclusive="left"):
                idx.append(t)
                vals.append(100 + rng.normal())
        df = pd.DataFrame({"open": vals, "close": vals, "volume": 1000.0}, index=pd.DatetimeIndex(idx))
        df["high"] = df["open"] + 0.5
        df["low"] = df["open"] - 0.5
        frames[s] = df[["open", "high", "low", "close", "volume"]]
    return frames, sess, rows


def test_sessions():
    # includes a DST change (Mar 10 2024) and a half day (Jul 3 2024)
    dates = ["2024-03-07", "2024-03-08", "2024-03-11", "2024-03-12", "2024-07-02", "2024-07-03", "2024-07-05"]
    frames, sess, _ = make_frames(dates, half_days=("2024-07-03",))
    p = build_panel(frames, sess, "15Min", "QQQ", log=lambda m: None)
    per_day = np.bincount(p.day)
    assert list(per_day) == [26, 26, 26, 26, 26, 14, 26], per_day
    ny = p.index.tz_convert(NY)
    assert all(t.strftime("%H:%M") == "09:45" for t in ny[np.r_[0, np.flatnonzero(np.diff(p.day)) + 1]])
    assert ny[p.eod][5].strftime("%H:%M") == "13:00" and ny[p.eod][0].strftime("%H:%M") == "16:00"
    assert p.eod.sum() == 7 and p.day_len[5] == 210
    ok("sessions: regular hours only, DST, half-day close at 13:00")
    # rules on the half day: flatten at the open of the last candle (12:45), no entries after 12:40
    cfg = load_config()
    r = bt.Rules(cfg, p, fomc_dates={dt.date(2024, 3, 12)})
    half = np.flatnonzero(p.day == 5)
    times = [ny[i].strftime("%H:%M") for i in half]
    assert r.flat_next[half[times.index("12:45")]] and not r.flat_next[half[times.index("12:30")]]
    assert r.block_next[half[times.index("12:45")]] and r.block_next[half[times.index("12:30")]] is not None
    # FOMC day (Mar 12): no entries opening 13:45-14:30
    fom = np.flatnonzero(p.day == 3)
    ft = {ny[i].strftime("%H:%M"): i for i in fom}
    assert r.block_next[ft["13:45"]] and r.block_next[ft["14:15"]] and not r.block_next[ft["14:30"]]
    assert not r.block_next[ft["13:30"]]
    ok("rules: half-day flatten, Fed-day 13:45-14:30 entry blackout")
    # an unfinished candle must be dropped
    now = sess.iloc[-1]["open"] + pd.Timedelta(minutes=37)
    p2 = build_panel(frames, sess, "15Min", "QQQ", now=now, log=lambda m: None)
    last = p2.index[-1].tz_convert(NY).strftime("%H:%M")
    assert last == "10:00", last
    assert not p2.eod[-1]
    ok("no unfinished candles used (10:07 -> last candle is the 9:45-10:00 one)")


def test_news_no_lookahead():
    tmp = Path(tempfile.mkdtemp())
    ns = NewsStore(tmp / "n.db")
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-02", periods=30)]
    frames, sess, _ = make_frames(dates)
    p = build_panel(frames, sess, "15Min", "QQQ", log=lambda m: None)
    # filler history so the 21-day coverage rule is satisfied
    arts = [{"id": i, "created_at": (pd.Timestamp(dates[i]) + pd.Timedelta(hours=12)).tz_localize(NY).isoformat(),
             "headline": "market update", "symbols": ["QQQ"]} for i in range(25)]
    t_news = pd.Timestamp(f"{dates[27]} 11:07").tz_localize(NY)
    arts.append({"id": 999, "created_at": t_news.isoformat(), "headline": "AAA misses estimates, shares plunge",
                 "symbols": ["AAA"]})
    ns.add(arts)
    f = ns.features(p)
    j = p.symbols.index("AAA")
    ny = p.index.tz_convert(NY)
    before = (ny < t_news)
    after = (ny >= t_news) & (ny < t_news + pd.Timedelta(hours=24))
    assert np.nanmax(np.abs(f["sent"][before & np.isfinite(f["sent"][:, j]), j])) == 0.0
    assert (f["sent"][after, j] < 0).all()
    first_after = ny[after][0].strftime("%H:%M")
    assert first_after == "11:15", first_after
    assert score_text("NVDA beats estimates, shares jump") > 0 and score_text("AAPL faces antitrust probe") < 0
    ok("news: a headline at 11:07 first affects the 11:15 candle, never earlier")


def test_earnings_mapping():
    cfg = load_config()
    cfg["market"]["symbols"] = ["QQQ", "AAA", "BBB"]
    cfg["market"]["qqq_heavyweights"] = ["AAA"]
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-02", periods=10)]
    frames, sess, _ = make_frames(dates, syms=("QQQ", "AAA", "BBB"))
    p = build_panel(frames, sess, "15Min", "QQQ", log=lambda m: None)
    import trader.events as E
    E.CACHE_FILE = Path(tempfile.mkdtemp()) / "events.json"     # never touch a real cache in tests
    ev = EventCalendar(cfg, log=lambda m: None).load()
    ev.have_fred = False
    d = [x for x in p.day_dates]
    ev.earnings = {"AAA": [(d[3], "amc")], "BBB": [(d[6], "bmo")]}
    ev.have_earnings = True
    f = ev.features(p)
    a, b, q = p.symbols.index("AAA"), p.symbols.index("BBB"), p.symbols.index("QQQ")
    day_flag = lambda arr, j: [int(arr[p.day == k, j].max()) for k in range(p.n_days)]
    assert day_flag(f["earn"], a) == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]       # AMC day 3 -> reacts day 4
    assert day_flag(f["earn_next"], a) == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  # reports after close of day 3
    assert day_flag(f["earn"], b) == [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]       # BMO day 6 -> reacts day 6
    assert day_flag(f["earn_next"], b) == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  # due before next open
    assert day_flag(f["earn"], q) == day_flag(f["earn"], a)               # QQQ follows its heavyweights
    assert np.isnan(f["macro"]).all()                                      # no FRED key -> unavailable
    assert np.isfinite(f["fomc"]).all() and f["fomc"].max() == 0            # Jan 2024 test days: no Fed meeting
    assert len(ev.fomc_dates()) == 56 and dt.date(2024, 11, 7) in ev.fomc_dates()
    assert sum(1 for d in ev.fomc_dates() if d.year == 2023) == 8
    ok("earnings: after-close reports react next day, before-open the same day; QQQ follows heavyweights")


def test_backtest_rules():
    # one stock that falls steadily: every long trade loses -> max 2 losing trades per stock per day
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-02", periods=40)]
    frames, sess, _ = make_frames(dates)
    for s, df in frames.items():
        n = len(df)
        base = 100 * np.exp(-0.0004 * np.arange(n))
        df["open"] = base
        df["close"] = base * 0.9996
        df["high"] = base * 1.0001
        df["low"] = base * 0.9990
    p = build_panel(frames, sess, "5Min", "QQQ", log=lambda m: None)
    cfg = load_config()
    book = FeatureBook(p)
    rules = bt.Rules(cfg, p)
    g = {"tf": "5Min", "dir": "long", "entry": [{"f": "ibs", "p": 1, "op": ">", "v": -1.0}], "exit": None,
         "stop": 0.5, "tp": 5.0, "trail": 0.0, "hold": 50, "per_day": 5}
    rets, trades, final = bt.run(book, g, 0.0002, rules, 300, p.T)
    days = p.day[trades[:, 1].astype(int)]
    for s in range(len(p.symbols)):
        per = np.bincount(days[trades[:, 0] == s])
        assert per.max() <= 2, per
    assert (p.day[trades[:, 2].astype(int)] == p.day[trades[:, 1].astype(int)]).all()
    ok("backtest: stops after 2 losing trades per stock per day; never overnight")
    cfg["day_trading"]["max_open_positions"] = 1
    rules1 = bt.Rules(cfg, p)
    rets, trades, final = bt.run(book, g, 0.0002, rules1, 300, p.T)
    spans = [(int(a), int(b)) for a, b in trades[:, 1:3]]
    for i in range(300, p.T):
        assert sum(1 for a, b in spans if a <= i <= b) <= 1
    ok("backtest: max open positions respected")
    cfg["day_trading"]["max_daily_loss_pct"] = 0.05
    cfg["day_trading"]["max_open_positions"] = 6
    rules2 = bt.Rules(cfg, p)
    rets, trades, final = bt.run(book, g, 0.0002, rules2, 300, p.T)
    daily = bt.daily_returns(rets, p.day, 300, p.T)
    port = rets.sum(axis=1)
    halted_days = 0
    for d in np.unique(p.day[300:]):
        idx = np.flatnonzero(p.day == d)
        idx = idx[idx >= 300]
        cum = np.cumsum(port[idx])
        hit = np.flatnonzero(cum <= -0.0005)
        if not len(hit):
            continue
        halted_days += 1
        halt_bar = idx[hit[0]]
        later = trades[(p.day[trades[:, 1].astype(int)] == d) & (trades[:, 1] > halt_bar)]
        assert len(later) == 0, (d, later)
        still_open_after = trades[(trades[:, 1] <= halt_bar) & (trades[:, 2] > halt_bar + 1)]
        assert len(still_open_after) == 0, still_open_after
    assert halted_days > 0
    ok("backtest: daily loss limit closes positions and halts the day")


def test_broker_rules():
    from tests.fake_alpaca import FakeAlpaca
    from trader.registry import Registry
    from trader.broker import Broker
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-02", periods=3)]
    frames, sess, rows = make_frames(dates)
    fake = FakeAlpaca({s: f[(f.index.tz_convert(NY).time >= dt.time(9, 30)) & (f.index.tz_convert(NY).time < dt.time(16))]
                       for s, f in frames.items()}, rows)
    cfg = load_config()
    cfg["market"]["symbols"] = ["QQQ", "AAA"]
    tmp = Path(tempfile.mkdtemp())
    reg = Registry(tmp / "b.db", clock=lambda: fake.now.isoformat())
    g = {"tf": "5Min", "dir": "long", "entry": [], "exit": None, "stop": 1, "tp": 2, "trail": 0, "hold": 5, "per_day": 2}
    sid = reg.add_strategy(g, {}, "champion")
    br = Broker(cfg, fake, reg, log=lambda m: None)
    t0 = pd.Timestamp(f"{dates[1]} 10:00").tz_localize(NY).tz_convert("UTC") + pd.Timedelta(seconds=10)
    fake.advance(t0)
    px = fake._price("AAA")
    tgt = lambda last: {sid: {"stats": {"last_candle": last.isoformat()},
                              "targets": {"AAA": {"action": "BUY", "side": 1, "ref_price": px, "stop_atr": 1.0,
                                                  "tp_atr": 2.0, "atr": 0.5, "weight": 0.25}}}}
    # stale data -> no order
    br.sync(tgt(t0 - pd.Timedelta(minutes=30)))
    assert not fake.calls
    ok("broker: never trades on stale data")
    br.sync(tgt(t0 - pd.Timedelta(seconds=10)))
    assert fake.calls and fake.calls[-1][0] == "bracket" and "AAA" in fake.pos
    n = len(fake.calls)
    br.sync(tgt(t0 - pd.Timedelta(seconds=10)))            # same candle again -> no duplicate
    assert len(fake.calls) == n
    ok("broker: bracket order placed once per signal (no duplicates)")
    # a strategy on probation trades at 1/4 size
    full_qty = fake.pos["AAA"][0]
    reg2 = Registry(tmp / "b2.db", clock=lambda: fake.now.isoformat())
    sid2 = reg2.add_strategy(g, {"probation": True}, "champion")
    fake2 = FakeAlpaca({s: f[(f.index.tz_convert(NY).time >= dt.time(9, 30)) & (f.index.tz_convert(NY).time < dt.time(16))]
                        for s, f in frames.items()}, rows)
    fake2.advance(t0)
    br2 = Broker(cfg, fake2, reg2, log=lambda m: None)
    br2.sync({sid2: tgt(t0 - pd.Timedelta(seconds=10))[sid]})
    q2 = fake2.pos["AAA"][0]
    assert abs(q2 - full_qty * 0.25) <= 1, (q2, full_qty)
    ok(f"broker: a strategy on probation trades at 1/4 size ({q2} vs {full_qty} shares)")
    # a SHORT strategy: sell first with TP below / SL above, then the end-of-day flatten buys it back
    reg3 = Registry(tmp / "b3.db", clock=lambda: fake.now.isoformat())
    gs = dict(g, dir="short")
    sid3 = reg3.add_strategy(gs, {}, "champion")
    fake3 = FakeAlpaca({s: f[(f.index.tz_convert(NY).time >= dt.time(9, 30)) & (f.index.tz_convert(NY).time < dt.time(16))]
                        for s, f in frames.items()}, rows)
    fake3.advance(t0)
    px3 = fake3._price("AAA")
    br3 = Broker(cfg, fake3, reg3, log=lambda m: None)
    br3.sync({sid3: {"stats": {"last_candle": (t0 - pd.Timedelta(seconds=10)).isoformat()},
                     "targets": {"AAA": {"action": "SELL_SHORT", "side": -1, "ref_price": px3, "stop_atr": 1.0,
                                         "tp_atr": 2.0, "atr": 0.5, "weight": 0.25}}}})
    call = [c for c in fake3.calls if c[0] == "bracket"][-1]
    assert call[3] == "sell" and call[4] < px3 < call[5], call          # TP below, SL above
    assert fake3.pos["AAA"][0] < 0
    fake3.advance(pd.Timestamp(f"{dates[1]} 15:56").tz_localize(NY).tz_convert("UTC"))
    br3.sync({})
    assert "AAA" not in fake3.pos, fake3.pos
    ok("broker: short trades - sells first, TP below / SL above, bought back before the close")
    # end of day flatten (with one failed close that needs a retry)
    fake.fail_close_once.add("AAA")
    t1 = pd.Timestamp(f"{dates[1]} 15:56").tz_localize(NY).tz_convert("UTC")
    fake.advance(t1)
    br.sync({})
    assert not fake.pos, fake.pos
    ok("broker: flattens everything 5 minutes before the close (retries a failed close)")
    # leftover position from yesterday is closed at the next sync
    fake.pos["AAA"] = [10, 100.0]
    reg.set_owner("AAA", sid)
    reg.db.execute("UPDATE broker_owner SET since=?", ((t1 - pd.Timedelta(days=1)).isoformat(),))
    reg.db.commit()
    t2 = pd.Timestamp(f"{dates[2]} 09:45").tz_localize(NY).tz_convert("UTC")
    fake.advance(t2)
    br.sync({})
    assert "AAA" not in fake.pos
    ok("broker: positions left over from a previous day are closed")
    # daily loss limit
    fake.pos["AAA"] = [100, px]
    reg.set_owner("AAA", sid)                     # a position the bot opened today
    fake.last_equity = fake.equity() / (1 - 0.03)
    br.sync({})
    assert not fake.pos and reg.meta("halted_day") == fake.now.tz_convert(NY).date().isoformat()
    br.sync(tgt(fake.now - pd.Timedelta(seconds=10)))
    assert "AAA" not in fake.pos
    ok("broker: daily loss limit closes everything and blocks new trades for the day")
    # a position the bot did NOT open (e.g. another bot in the same account) is never touched
    fake.pos["QQQ"] = [7, 400.0]
    reg.set_meta("halted_day", None)
    fake.last_equity = fake.equity()
    t3 = pd.Timestamp(f"{dates[2]} 11:00").tz_localize(NY).tz_convert("UTC")
    fake.advance(t3)
    br.sync({})
    assert any(e["kind"] == "warning" for e in reg.events(50))
    fake.advance(pd.Timestamp(f"{dates[2]} 15:56").tz_localize(NY).tz_convert("UTC"))
    br.sync({})
    assert fake.pos.get("QQQ") == [7, 400.0], fake.pos
    ok("broker: never touches positions it did not open itself (warns instead)")
    # not the bot's symbol -> never touched
    fake.pos["SPY"] = [5, 400.0]
    fake.frames["SPY"] = fake.frames["AAA"]
    br.flatten_all("test")
    assert "SPY" in fake.pos
    ok("broker: never touches positions in symbols the bot doesn't trade")


if __name__ == "__main__":
    test_sessions()
    test_news_no_lookahead()
    test_earnings_mapping()
    test_backtest_rules()
    test_broker_rules()
    print(f"\nALL {len(PASSED)} UNIT TESTS PASSED")
