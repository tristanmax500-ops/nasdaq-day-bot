"""A fake Alpaca server for testing: serves bars/news/calendar from synthetic data and
simulates a paper account with bracket orders (fills at the next 5-min bar open,
TP/SL legs checked against each bar's high/low)."""
import itertools

import numpy as np
import pandas as pd

NY = "America/New_York"


class FakeAlpaca:
    def __init__(self, frames, sessions_rows, articles=(), equity=100000.0, iex_vol_ratio=30.0):
        self.frames = frames                    # {sym: df indexed by bar START, open..volume} 5-min
        self.rows = sessions_rows
        self.articles = sorted(articles, key=lambda a: a["created_at"])
        self.now = pd.Timestamp(frames[next(iter(frames))].index[0])
        self.cash = equity
        self.pos = {}                           # sym -> [qty, avg]
        self.orders_ = []
        self.ids = itertools.count(1)
        self.iex_ratio = iex_vol_ratio
        self.last_equity = equity
        self.calls = []
        self.fail_close_once = set()

    # ---------- helpers ----------
    def _session(self, d):
        for r in self.rows:
            if r["date"] == d:
                o = pd.Timestamp(f"{d} {r['open']}").tz_localize(NY).tz_convert("UTC")
                c = pd.Timestamp(f"{d} {r['close']}").tz_localize(NY).tz_convert("UTC")
                return o, c
        return None

    def _price(self, sym):
        df = self.frames[sym]
        done = df[df.index + pd.Timedelta(minutes=5) <= self.now]
        cur = df[(df.index <= self.now)]
        if len(cur) and cur.index[-1] + pd.Timedelta(minutes=5) > self.now:
            return float(cur["open"].iloc[-1]) if len(done) == 0 else float(done["close"].iloc[-1])
        return float(done["close"].iloc[-1])

    def equity(self):
        return self.cash + sum(q * self._price(s) for s, (q, a) in self.pos.items())

    # ---------- time ----------
    def advance(self, to):
        """Process every 5-min bar that STARTED in (self.now, to]."""
        to = pd.Timestamp(to)
        for sym, df in self.frames.items():
            bars = df[(df.index >= self.now) & (df.index < to)]
            for t, b in bars.iterrows():
                self._process_bar(sym, t, b)
        prev_day = self.now.tz_convert(NY).date()
        self.now = to
        if self.now.tz_convert(NY).date() != prev_day:
            self.last_equity = self.equity()

    def _process_bar(self, sym, t, b):
        for o in self.orders_:
            if o["symbol"] != sym:
                continue
            if o.get("parent") is None and o["status"] in ("new", "accepted") and o["order_class"] == "bracket" and pd.Timestamp(o["created"]) <= t:
                px = float(b["open"])
                self._fill(o, px)
                for leg in o["legs"]:
                    leg["status"] = "new"
        for o in self.orders_:
            if o["symbol"] != sym or o["status"] != "new" or o.get("parent") is None:
                continue
            parent = self._by_id(o["parent"])
            sibling = [l for l in parent["legs"] if l is not o][0]
            long = parent["side"] == "buy"
            if o["type"] == "stop":
                hit = (b["low"] <= o["stop_price"]) if long else (b["high"] >= o["stop_price"])
                px = min(b["open"], o["stop_price"]) if long else max(b["open"], o["stop_price"])
            else:
                hit = (b["high"] >= o["limit_price"]) if long else (b["low"] <= o["limit_price"])
                px = max(b["open"], o["limit_price"]) if long else min(b["open"], o["limit_price"])
            if hit and sibling["status"] == "new":
                self._fill(o, float(px))
                sibling["status"] = "canceled"

    def _fill(self, o, px):
        q = int(o["qty"]) * (1 if o["side"] == "buy" else -1)
        cur = self.pos.get(o["symbol"], [0, 0.0])
        newq = cur[0] + q
        self.cash -= q * px
        if newq == 0:
            self.pos.pop(o["symbol"], None)
        else:
            avg = px if cur[0] == 0 else cur[1]
            self.pos[o["symbol"]] = [newq, avg]
        o["status"] = "filled"
        o["filled_avg_price"] = str(px)

    def _by_id(self, i):
        return next(o for o in self.orders_ if o["id"] == i)

    # ---------- API surface used by the bot ----------
    def clock(self):
        d = self.now.tz_convert(NY).date().isoformat()
        s = self._session(d)
        is_open = bool(s and s[0] <= self.now < s[1])
        nxt_close = s[1] if s and self.now < s[1] else None
        nxt_open = None
        for r in self.rows:
            o, c = self._session(r["date"])
            if o > self.now and nxt_open is None:
                nxt_open = o
            if nxt_close is None and c > self.now:
                nxt_close = c
        return {"timestamp": self.now.isoformat(), "is_open": is_open,
                "next_open": (nxt_open or self.now + pd.Timedelta(days=1)).isoformat(),
                "next_close": (nxt_close or self.now + pd.Timedelta(days=1)).isoformat()}

    def calendar(self, start, end):
        return [r for r in self.rows if start <= r["date"] <= end]

    def account(self):
        return {"equity": str(self.equity()), "last_equity": str(self.last_equity),
                "buying_power": str(4 * self.equity()), "trading_blocked": False, "account_blocked": False}

    def positions(self):
        out = []
        for s, (q, a) in self.pos.items():
            px = self._price(s) if s in self.frames else a
            out.append({"symbol": s, "qty": str(q), "avg_entry_price": str(a), "side": "long" if q > 0 else "short",
                        "current_price": str(px), "unrealized_pl": str((px - a) * q),
                        "unrealized_plpc": str((px / a - 1) if a else 0)})
        return out

    def portfolio_history(self, period="3M"):
        self.hist = getattr(self, "hist", [])
        d = self.now.tz_convert(NY).normalize()
        if not self.hist or self.hist[-1][0] != d:
            self.hist.append((d, self.equity()))
        return {"timestamp": [int(t.timestamp()) for t, _ in self.hist], "equity": [e for _, e in self.hist]}

    def orders(self, status="open", after=None, limit=500, nested=False):
        out = []
        for o in self.orders_:
            if status == "open" and o["status"] in ("new", "accepted", "held", "partially_filled"):
                out.append(o)
        return out

    def order_by_client_id(self, cid):
        return next(o for o in self.orders_ if o.get("client_order_id") == cid)

    def submit_bracket(self, symbol, qty, side, take_profit, stop_loss, client_order_id):
        self.calls.append(("bracket", symbol, qty, side, take_profit, stop_loss))
        px = self._price(symbol)
        if side == "buy" and not (stop_loss < px < take_profit):
            raise RuntimeError("422 take_profit/stop_loss invalid vs base price")
        if any(o.get("client_order_id") == client_order_id for o in self.orders_):
            raise RuntimeError("422 client_order_id must be unique")
        pid = next(self.ids)
        exit_side = "sell" if side == "buy" else "buy"
        tp = {"id": next(self.ids), "symbol": symbol, "qty": qty, "side": exit_side, "type": "limit",
              "limit_price": float(take_profit), "status": "held", "parent": pid, "order_class": "bracket"}
        sl = {"id": next(self.ids), "symbol": symbol, "qty": qty, "side": exit_side, "type": "stop",
              "stop_price": float(stop_loss), "status": "held", "parent": pid, "order_class": "bracket"}
        o = {"id": pid, "client_order_id": client_order_id, "symbol": symbol, "qty": qty, "side": side,
             "type": "market", "order_class": "bracket", "status": "new", "created": self.now.isoformat(),
             "legs": [tp, sl], "parent": None}
        self.orders_ += [o, tp, sl]
        # market order fills right away at the current price plus 2 bps slippage
        self._fill(o, px * (1.0002 if side == "buy" else 0.9998))
        tp["status"] = sl["status"] = "new"
        return o

    def cancel_order(self, oid):
        o = self._by_id(oid)
        if o["status"] in ("new", "held", "accepted"):
            o["status"] = "canceled"
        if o.get("legs"):
            for l in o["legs"]:
                if l["status"] in ("new", "held"):
                    l["status"] = "canceled"
        return {}

    def cancel_all_orders(self):
        for o in self.orders_:
            if o["status"] in ("new", "held", "accepted"):
                o["status"] = "canceled"

    def close_position(self, symbol):
        if symbol not in self.pos:
            raise RuntimeError("404 position does not exist")
        if any(o["symbol"] == symbol and o["status"] == "new" and o.get("parent") for o in self.orders_):
            raise RuntimeError("403 insufficient qty available for order")
        if symbol in self.fail_close_once:
            self.fail_close_once.discard(symbol)
            raise RuntimeError("403 insufficient qty available for order (pending cancel)")
        q, a = self.pos[symbol]
        o = {"id": next(self.ids), "symbol": symbol, "qty": abs(q), "side": "sell" if q > 0 else "buy",
             "type": "market", "order_class": "simple", "status": "new", "legs": [], "parent": None}
        self._fill(o, self._price(symbol))
        self.orders_.append(o)
        self.calls.append(("close", symbol))
        return o

    def close_all_positions(self):
        for s in list(self.pos):
            self.close_position(s)

    def latest_trade(self, symbol, feed="iex"):
        return self._price(symbol)

    def bars(self, symbols, timeframe, start, end=None, feed="sip"):
        assert timeframe == "5Min"
        start = pd.Timestamp(start)
        start = start.tz_localize("UTC") if start.tzinfo is None else start
        end = pd.Timestamp(end) if end is not None else self.now
        end = end.tz_localize("UTC") if end.tzinfo is None else end
        end = min(end, self.now)
        out = {}
        for s in symbols:
            df = self.frames[s]
            part = df[(df.index >= start) & (df.index < end)].copy()
            # a still-forming bar is returned too (like the real API), with partial values
            if len(part) and part.index[-1] + pd.Timedelta(minutes=5) > self.now:
                part.iloc[-1, part.columns.get_loc("close")] = part["open"].iloc[-1]
            if feed == "iex":
                part["volume"] = part["volume"] / self.iex_ratio
            if len(part):
                out[s] = part
        return out

    def news(self, symbols, start, end=None):
        start = pd.Timestamp(start)
        start = start.tz_localize("UTC") if start.tzinfo is None else start
        for a in self.articles:
            t = pd.Timestamp(a["created_at"])
            if start <= t <= self.now:
                yield a


def frames_from_panel(panel):
    """Synthetic 5-min Panel -> Alpaca-style frames indexed by bar START."""
    frames = {}
    start = panel.index - pd.Timedelta(minutes=5)
    for j, s in enumerate(panel.symbols):
        frames[s] = pd.DataFrame({"open": panel.o[:, j], "high": panel.h[:, j], "low": panel.l[:, j],
                                  "close": panel.c[:, j], "volume": panel.v[:, j]}, index=start)
    return frames


def calendar_rows(panel):
    return [{"date": d.isoformat(), "open": "09:30", "close": "16:00"} for d in panel.day_dates]


def synthetic_articles(panel, rng, per_day=6):
    arts = []
    i = 1
    heads_pos = ["{s} beats estimates, shares jump", "{s} upgraded at major bank", "{s} record revenue"]
    heads_neg = ["{s} misses estimates, stock falls", "{s} faces antitrust probe", "{s} downgraded"]
    for d in panel.day_dates:
        for _ in range(per_day):
            s = panel.symbols[rng.integers(len(panel.symbols))]
            pos = rng.random() < 0.5
            h = (heads_pos if pos else heads_neg)[rng.integers(3)].format(s=s)
            t = pd.Timestamp(d).tz_localize(NY) + pd.Timedelta(hours=float(rng.uniform(6, 20)))
            arts.append({"id": i, "created_at": t.tz_convert("UTC").isoformat(), "headline": h,
                         "summary": "", "symbols": [s]})
            i += 1
    return arts
