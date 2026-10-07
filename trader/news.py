"""Daily news -> numbers the strategies can use.

Headlines come from Alpaca's free news feed (Benzinga, history back to 2015).
Each headline gets a simple, transparent sentiment score from a finance word list
(edit the lists below if you like). Features, for every candle, use ONLY news
published before that candle closed - no peeking into the future:

  sent      this stock's news mood over the last 24h      (-1 very negative .. +1 very positive)
  newsv     how much more news than usual (buzz)           (0 = normal, +2 = ~7x normal)
  mkt_sent  mood of ALL Nasdaq news over the last 24h      (-1 .. +1)
"""
import re
import sqlite3
import time

import numpy as np
import pandas as pd

from .config import DATA_DIR

POSITIVE = set("""
beat beats tops topped surpass surpasses surpassed exceed exceeds exceeded record records
upgrade upgrades upgraded outperform overweight raises raised raise boost boosts boosted
jump jumps jumped soar soars soared surge surges surged rally rallies rallied gain gains
rise rises rising rose strong stronger strongest bullish growth grows expands expansion
profit profitable profits win wins won approval approved approves partnership partners
breakthrough optimistic upbeat rebound rebounds rebounded climb climbs climbed accelerate
accelerates accelerating buyback buybacks hike hikes upside momentum demand robust
""".split())

NEGATIVE = set("""
miss misses missed downgrade downgrades downgraded underperform underweight cut cuts lowers
lowered plunge plunges plunged plummet plummets sink sinks sank slump slumps slumped drop
drops dropped fall falls fell tumble tumbles tumbled slide slides slid weak weaker weakest
bearish loss losses lawsuit lawsuits sue sues sued probe probes investigation recall recalls
fraud fine fined antitrust ban bans banned delay delays delayed layoffs layoff warns warning
warned concern concerns decline declines declined crash crashes selloff disappoint disappoints
disappointing disappointed halt halts halted subpoena resign resigns resigned tariff tariffs
shortage glitch outage breach downside slowdown slowing slows bankruptcy default investigate
""".split())

_WORD = re.compile(r"[a-z]+")


def score_text(headline, summary=""):
    words = _WORD.findall((headline or "").lower())
    s = sum(1 for w in words if w in POSITIVE) - sum(1 for w in words if w in NEGATIVE)
    if summary:
        ws = _WORD.findall(summary.lower())
        s += 0.5 * (sum(1 for w in ws if w in POSITIVE) - sum(1 for w in ws if w in NEGATIVE))
    return float(np.clip(s, -3, 3))


class NewsStore:
    def __init__(self, path=None):
        self.path = str(path or DATA_DIR / "news.db")
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.execute("CREATE TABLE IF NOT EXISTS articles (id INTEGER PRIMARY KEY, t REAL, syms TEXT, "
                        "score REAL, headline TEXT)")
        self.db.execute("CREATE INDEX IF NOT EXISTS ix_t ON articles(t)")
        self.db.commit()
        self._cache = None

    def last_time(self):
        r = self.db.execute("SELECT MAX(t) FROM articles").fetchone()[0]
        return pd.Timestamp(r, unit="s", tz="UTC") if r else None

    def first_time(self):
        r = self.db.execute("SELECT MIN(t) FROM articles").fetchone()[0]
        return pd.Timestamp(r, unit="s", tz="UTC") if r else None

    def add(self, articles):
        rows = []
        for a in articles:
            try:
                t = pd.Timestamp(a["created_at"]).timestamp()
            except Exception:
                continue
            syms = "," + ",".join(a.get("symbols") or []) + ","
            rows.append((int(a["id"]), t, syms, score_text(a.get("headline", ""), a.get("summary", "")),
                         (a.get("headline") or "")[:300]))
        self.db.executemany("INSERT OR IGNORE INTO articles VALUES (?,?,?,?,?)", rows)
        self.db.commit()
        self._cache = None
        return len(rows)

    def update(self, api, symbols, history_start, log=print):
        """Download new articles (resumable - safe to stop and restart)."""
        last = self.last_time()
        start = (last - pd.Timedelta(hours=6)) if last is not None else pd.Timestamp(history_start, tz="UTC")
        if last is None:
            log(f"  downloading news since {history_start} (first time only - can take 20-60 min; "
                "safe to stop and resume later) ...")
        buf, n, t0 = [], 0, time.time()
        for a in api.news(symbols, start):
            buf.append(a)
            if len(buf) >= 500:
                n += self.add(buf)
                buf = []
                if time.time() - t0 > 30:
                    log(f"    ... {n:,} articles, now at {self.last_time():%Y-%m-%d}")
                    t0 = time.time()
        if buf:
            n += self.add(buf)
        return n

    def recent_headlines(self, symbol=None, hours=24, now=None, limit=20):
        now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC"))
        q = "SELECT t, syms, score, headline FROM articles WHERE t > ?"
        args = [(now - pd.Timedelta(hours=hours)).timestamp()]
        if symbol:
            q += " AND syms LIKE ?"
            args.append(f"%,{symbol},%")
        q += " ORDER BY t DESC LIMIT ?"
        args.append(limit)
        return self.db.execute(q, args).fetchall()

    def _load(self):
        if self._cache is None:
            rows = self.db.execute("SELECT t, syms, score FROM articles ORDER BY t").fetchall()
            t = np.array([r[0] for r in rows], dtype=float)
            syms = [r[1] for r in rows]
            sc = np.array([r[2] for r in rows], dtype=float)
            self._cache = (t, syms, sc)
        return self._cache

    def features(self, panel):
        """Returns dict of [T, S] arrays: sent, newsv, mkt_sent (NaN where no news history)."""
        T, S = panel.T, len(panel.symbols)
        out = {k: np.full((T, S), np.nan) for k in ("sent", "newsv", "mkt_sent")}
        t_all, syms, sc_all = self._load()
        if len(t_all) == 0:
            return out
        bt = np.asarray((panel.index - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1),
                        dtype=float)
        covered = bt >= t_all[0] + 21 * 86400
        day = 86400.0

        def window_sums(ta, vals, ends, length):
            cs = np.concatenate([[0.0], np.cumsum(vals)])
            i1 = np.searchsorted(ta, ends, side="right")
            i0 = np.searchsorted(ta, ends - length, side="right")
            return cs[i1] - cs[i0]

        # market-wide mood
        cnt = window_sums(t_all, np.ones_like(sc_all), bt, day)
        ssum = window_sums(t_all, sc_all, bt, day)
        mk = np.where(cnt > 0, np.tanh(ssum / np.maximum(cnt, 1)), 0.0)
        out["mkt_sent"][:] = np.where(covered, mk, np.nan)[:, None]
        for j, s in enumerate(panel.symbols):
            tag = f",{s},"
            m = np.fromiter((tag in x for x in syms), dtype=bool, count=len(syms))
            ta, sv = t_all[m], sc_all[m]
            if len(ta) == 0:
                out["sent"][:, j] = np.where(covered, 0.0, np.nan)
                out["newsv"][:, j] = np.where(covered, 0.0, np.nan)
                continue
            c24 = window_sums(ta, np.ones_like(sv), bt, day)
            s24 = window_sums(ta, sv, bt, day)
            base = window_sums(ta, np.ones_like(sv), bt - day, 20 * day) / 20.0
            out["sent"][:, j] = np.where(covered, np.tanh(s24 / 4.0), np.nan)
            out["newsv"][:, j] = np.where(covered, np.log1p(c24) - np.log1p(base), np.nan)
        return out
