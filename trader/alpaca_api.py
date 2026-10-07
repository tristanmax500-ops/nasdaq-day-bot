"""Minimal Alpaca REST client (paper trading + market data + news).

Only the PAPER trading URL is ever used - this code cannot touch a live account.
"""
import time

import pandas as pd
import requests

PAPER_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"


class AlpacaError(RuntimeError):
    pass


class Alpaca:
    def __init__(self, key, secret, log=print, session=None):
        if not key or not secret:
            raise AlpacaError("Alpaca keys missing. Put them in config.yaml (account: alpaca_key / alpaca_secret).")
        self.h = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self.log = log
        self.s = session or requests.Session()
        self._last = 0.0

    # ------------------------------------------------------------------
    def _req(self, method, url, params=None, json=None, retries=6):
        for attempt in range(retries):
            # stay well under the 200 requests/minute limit of the free plan
            wait = 0.32 - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self.s.request(method, url, headers=self.h, params=params, json=json, timeout=30)
            except requests.RequestException as e:
                if attempt == retries - 1:
                    raise AlpacaError(f"network error: {e}")
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(60, 2 ** (attempt + 1)))
                continue
            if r.status_code in (401, 403):
                raise AlpacaError(f"Alpaca refused the request ({r.status_code}): {r.text[:200]}. "
                                  "Check your API keys are PAPER keys and copied correctly.")
            if r.status_code >= 400:
                raise AlpacaError(f"{method} {url} -> {r.status_code}: {r.text[:300]}")
            if r.status_code == 204 or not r.content:
                return {}
            return r.json()
        raise AlpacaError(f"{method} {url} failed after {retries} attempts")

    # ---------------- trading (paper only) ----------------
    def account(self):
        return self._req("GET", f"{PAPER_URL}/v2/account")

    def clock(self):
        return self._req("GET", f"{PAPER_URL}/v2/clock")

    def calendar(self, start, end):
        return self._req("GET", f"{PAPER_URL}/v2/calendar", params={"start": start, "end": end})

    def positions(self):
        return self._req("GET", f"{PAPER_URL}/v2/positions")

    def orders(self, status="open", after=None, limit=500, nested=False):
        p = {"status": status, "limit": limit, "direction": "desc", "nested": str(nested).lower()}
        if after:
            p["after"] = after
        return self._req("GET", f"{PAPER_URL}/v2/orders", params=p)

    def submit_bracket(self, symbol, qty, side, take_profit, stop_loss, client_order_id):
        body = {
            "symbol": symbol, "qty": str(int(qty)), "side": side, "type": "market",
            # GTC legs: if the bot/computer dies, the stop-loss keeps protecting the position
            "time_in_force": "gtc", "order_class": "bracket",
            "take_profit": {"limit_price": f"{take_profit:.2f}"},
            "stop_loss": {"stop_price": f"{stop_loss:.2f}"},
            "client_order_id": client_order_id,
        }
        return self._req("POST", f"{PAPER_URL}/v2/orders", json=body)

    def order_by_client_id(self, client_id):
        return self._req("GET", f"{PAPER_URL}/v2/orders:by_client_order_id", params={"client_order_id": client_id})

    def cancel_order(self, order_id):
        return self._req("DELETE", f"{PAPER_URL}/v2/orders/{order_id}")

    def cancel_all_orders(self):
        return self._req("DELETE", f"{PAPER_URL}/v2/orders")

    def close_position(self, symbol):
        return self._req("DELETE", f"{PAPER_URL}/v2/positions/{symbol}")

    def close_all_positions(self):
        return self._req("DELETE", f"{PAPER_URL}/v2/positions", params={"cancel_orders": "true"})

    def portfolio_history(self, period="3M"):
        return self._req("GET", f"{PAPER_URL}/v2/account/portfolio/history",
                         params={"period": period, "timeframe": "1D"})

    def latest_trade(self, symbol, feed="iex"):
        r = self._req("GET", f"{DATA_URL}/v2/stocks/{symbol}/trades/latest", params={"feed": feed})
        return float(r["trade"]["p"])

    # ---------------- market data ----------------
    def bars(self, symbols, timeframe, start, end=None, feed="sip"):
        """All bars for symbols in [start, end). Returns {symbol: DataFrame(o,h,l,c,v) indexed by bar START (UTC)}."""
        params = {"symbols": ",".join(symbols), "timeframe": timeframe, "start": _iso(start),
                  "limit": 10000, "adjustment": "all", "feed": feed, "sort": "asc"}
        if end is not None:
            params["end"] = _iso(end)
        out = {s: [] for s in symbols}
        pages = 0
        while True:
            r = self._req("GET", f"{DATA_URL}/v2/stocks/bars", params=params)
            for sym, rows in (r.get("bars") or {}).items():
                out.setdefault(sym, []).extend(rows)
            pages += 1
            if pages % 25 == 0:
                self.log(f"    ... {pages} pages downloaded")
            tok = r.get("next_page_token")
            if not tok:
                break
            params["page_token"] = tok
        frames = {}
        for sym, rows in out.items():
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["t"] = pd.to_datetime(df["t"], utc=True)
            df = df.set_index("t")[["o", "h", "l", "c", "v"]].astype(float)
            df.columns = ["open", "high", "low", "close", "volume"]
            frames[sym] = df[~df.index.duplicated(keep="last")].sort_index()
        return frames

    def news(self, symbols, start, end=None):
        """Yields news articles (dicts) oldest first."""
        params = {"symbols": ",".join(symbols), "start": _iso(start), "limit": 50, "sort": "asc",
                  "include_content": "false"}
        if end is not None:
            params["end"] = _iso(end)
        while True:
            r = self._req("GET", f"{DATA_URL}/v1beta1/news", params=params)
            for a in r.get("news") or []:
                yield a
            tok = r.get("next_page_token")
            if not tok:
                break
            params["page_token"] = tok


def _iso(t):
    t = pd.Timestamp(t)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

