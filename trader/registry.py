"""The bot's memory: every validated strategy, its paper trades, orders and history.

Stored in a single SQLite file (data/brain.db). Statuses:
  champion   - currently paper trading
  challenger - an improved version shadow-trading against its parent
  reserve    - passed validation, waiting for a free champion slot
  retired    - stopped (edge decayed, or replaced by a better version)
"""
import json
import sqlite3
import uuid
from datetime import datetime, timezone


SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    id TEXT PRIMARY KEY, family TEXT, parent TEXT, version INTEGER,
    status TEXT, created TEXT, paper_start TEXT, genome TEXT, report TEXT,
    fitness REAL, note TEXT, health TEXT, last_judged_trades INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS paper_trades (
    strategy TEXT, symbol TEXT, entry_time TEXT, exit_time TEXT,
    entry_px REAL, exit_px REAL, ret REAL, reason TEXT, regime TEXT, open INTEGER, rmult REAL, weight REAL);
CREATE TABLE IF NOT EXISTS paper_state (
    strategy TEXT PRIMARY KEY, updated TEXT, equity TEXT, targets TEXT, stats TEXT);
CREATE TABLE IF NOT EXISTS events (ts TEXT, strategy TEXT, kind TEXT, message TEXT);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS broker_owner (symbol TEXT PRIMARY KEY, strategy TEXT, since TEXT);
CREATE TABLE IF NOT EXISTS broker_orders (client_id TEXT PRIMARY KEY, strategy TEXT, symbol TEXT,
    side TEXT, qty INTEGER, ref_price REAL, stop REAL, tp REAL, submitted TEXT, bar_time TEXT,
    status TEXT, fill_price REAL, slippage_bps REAL);
"""


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Registry:
    def __init__(self, path, clock=None):
        self.path = str(path)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()
        self.clock = clock or utcnow   # self-test swaps in a simulated clock

    # ---------------- meta / counters ----------------
    def meta(self, k, default=None):
        r = self.db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return json.loads(r["v"]) if r else default

    def set_meta(self, k, v):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",
                        (k, json.dumps(v, default=lambda o: o.item() if hasattr(o, "item") else str(o))))
        self.db.commit()

    def add_trials(self, n, sr_values=None, families=None):
        total = self.meta("total_trials", 0) + int(n)
        self.set_meta("total_trials", total)
        if families:
            fam = set(self.meta("families", []))
            fam |= set(families)
            self.set_meta("families", sorted(fam))
        if sr_values is not None and len(sr_values):
            # running estimate of the variance of Sharpe ratios across all trials
            import numpy as np
            a = np.asarray(sr_values, dtype=float)
            a = a[np.isfinite(a)]
            st = self.meta("sr_stats", {"n": 0, "mean": 0.0, "m2": 0.0})
            n0, m0, s0 = st["n"], st["mean"], st["m2"]
            n1 = len(a)
            if n1:
                m1 = float(a.mean())
                v1 = float(a.var()) * n1
                n = n0 + n1
                delta = m1 - m0
                mean = m0 + delta * n1 / n
                m2 = s0 + v1 + delta * delta * n0 * n1 / n
                self.set_meta("sr_stats", {"n": n, "mean": mean, "m2": m2})
        return total

    def effective_trials(self):
        """Distinct strategy ideas tried so far (GA variations of one idea are
        highly correlated, so counting each one separately would be wrong)."""
        # families are counted by the building blocks they use, but many of them trade almost
        # identically (a 3rd condition that changes nothing makes a "new" family). The measured
        # independence factor (0..1, see evolve.estimate_independence) corrects for that.
        f = float(self.meta("independence", 1.0) or 1.0)
        return max(10, int(len(self.meta("families", [])) * min(1.0, max(f, 0.001))))

    def update_independence(self, f, n_samples):
        """Running average of how many really-different strategies there are per family tried."""
        st = self.meta("independence_stats", {"n": 0, "sum": 0.0})
        st = {"n": st["n"] + int(n_samples), "sum": st["sum"] + float(f) * int(n_samples)}
        self.set_meta("independence_stats", st)
        self.set_meta("independence", st["sum"] / st["n"])
        return st["sum"] / st["n"]

    def sr_variance(self):
        st = self.meta("sr_stats", {"n": 0, "mean": 0.0, "m2": 0.0})
        return st["m2"] / st["n"] if st["n"] > 1 else 1e-4

    # ---------------- strategies ----------------
    def add_strategy(self, genome, report, status, fitness=0.0, parent=None, family=None,
                     version=1, note=""):
        sid = uuid.uuid4().hex[:8]
        family = family or sid
        now = self.clock()
        self.db.execute(
            "INSERT INTO strategies (id,family,parent,version,status,created,paper_start,genome,report,fitness,note,health)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, family, parent, version, status, now, now if status in ("champion", "challenger") else None,
             json.dumps(genome), json.dumps(report), float(fitness), note, "new"))
        self.db.commit()
        return sid

    def _row(self, r):
        d = dict(r)
        d["genome"] = json.loads(d["genome"])
        d["report"] = json.loads(d["report"]) if d["report"] else {}
        return d

    def get(self, sid):
        r = self.db.execute("SELECT * FROM strategies WHERE id=?", (sid,)).fetchone()
        return self._row(r) if r else None

    def list(self, statuses=None):
        if statuses:
            q = "SELECT * FROM strategies WHERE status IN (%s) ORDER BY created" % ",".join("?" * len(statuses))
            rows = self.db.execute(q, tuple(statuses)).fetchall()
        else:
            rows = self.db.execute("SELECT * FROM strategies ORDER BY created").fetchall()
        return [self._row(r) for r in rows]

    def set_status(self, sid, status, note=None):
        s = self.get(sid)
        paper_start = s["paper_start"]
        if status in ("champion", "challenger") and not paper_start:
            paper_start = self.clock()
        self.db.execute("UPDATE strategies SET status=?, paper_start=?, note=COALESCE(?, note) WHERE id=?",
                        (status, paper_start, note, sid))
        self.db.commit()

    def set_probation(self, sid, on):
        s = self.get(sid)
        if not s:
            return
        rep = dict(s["report"])
        rep["probation"] = bool(on)
        self.db.execute("UPDATE strategies SET report=? WHERE id=?", (json.dumps(rep), sid))
        self.db.commit()

    def set_health(self, sid, health, judged_trades=None):
        if judged_trades is None:
            self.db.execute("UPDATE strategies SET health=? WHERE id=?", (health, sid))
        else:
            self.db.execute("UPDATE strategies SET health=?, last_judged_trades=? WHERE id=?",
                            (health, int(judged_trades), sid))
        self.db.commit()

    def known_keys(self):
        from .genome import key
        return {key(s["genome"]) for s in self.list()}

    # ---------------- paper trading ----------------
    def replace_paper_trades(self, sid, trades):
        self.db.execute("DELETE FROM paper_trades WHERE strategy=?", (sid,))
        self.db.executemany(
            "INSERT INTO paper_trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [(sid, t["symbol"], t["entry_time"], t["exit_time"], t["entry_px"], t["exit_px"],
              t["ret"], t["reason"], json.dumps(t["regime"]), int(t["open"]), t["R"], t["weight"])
             for t in trades])
        self.db.commit()

    def paper_trades(self, sid, closed_only=True):
        q = "SELECT * FROM paper_trades WHERE strategy=?" + (" AND open=0" if closed_only else "")
        rows = self.db.execute(q + " ORDER BY exit_time", (sid,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["regime"] = json.loads(d["regime"]) if d["regime"] else {}
            out.append(d)
        return out

    def set_paper_state(self, sid, equity, targets, stats):
        self.db.execute("INSERT OR REPLACE INTO paper_state VALUES (?,?,?,?,?)",
                        (sid, self.clock(), json.dumps(equity), json.dumps(targets), json.dumps(stats)))
        self.db.commit()

    def paper_state(self, sid):
        r = self.db.execute("SELECT * FROM paper_state WHERE strategy=?", (sid,)).fetchone()
        if not r:
            return None
        return {"updated": r["updated"], "equity": json.loads(r["equity"]),
                "targets": json.loads(r["targets"]), "stats": json.loads(r["stats"])}

    # ---------------- broker bookkeeping ----------------
    def owner(self, symbol):
        r = self.db.execute("SELECT strategy FROM broker_owner WHERE symbol=?", (symbol,)).fetchone()
        return r["strategy"] if r else None

    def owners(self):
        return {r["symbol"]: r["strategy"] for r in self.db.execute("SELECT * FROM broker_owner")}

    def set_owner(self, symbol, strategy):
        if strategy is None:
            self.db.execute("DELETE FROM broker_owner WHERE symbol=?", (symbol,))
        else:
            self.db.execute("INSERT OR REPLACE INTO broker_owner VALUES (?,?,?)", (symbol, strategy, self.clock()))
        self.db.commit()

    def order_logged(self, client_id):
        return self.db.execute("SELECT 1 FROM broker_orders WHERE client_id=?", (client_id,)).fetchone() is not None

    def log_order(self, client_id, strategy, symbol, side, qty, ref_price, stop, tp, bar_time, status):
        self.db.execute("INSERT OR REPLACE INTO broker_orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (client_id, strategy, symbol, side, int(qty), float(ref_price), float(stop), float(tp),
                         self.clock(), bar_time, status, None, None))
        self.db.commit()

    def unfilled_orders(self):
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM broker_orders WHERE fill_price IS NULL AND status='submitted'")]

    def set_fill(self, client_id, price, slip_bps, status="filled"):
        self.db.execute("UPDATE broker_orders SET fill_price=?, slippage_bps=?, status=? WHERE client_id=?",
                        (price, slip_bps, status, client_id))
        self.db.commit()

    def recent_slippage(self, n=200):
        rows = self.db.execute("SELECT slippage_bps FROM broker_orders WHERE slippage_bps IS NOT NULL "
                               "ORDER BY submitted DESC LIMIT ?", (n,)).fetchall()
        return [r[0] for r in rows]

    # ---------------- events ----------------
    # ---------------- strategy-search history (for the progress chart) ----------------
    def search_runs(self):
        return self.meta("search_runs", []) or []

    def add_search_run(self, run, keep=200):
        runs = self.search_runs()
        runs.append(run)
        runs.sort(key=lambda r: r.get("started") or "")
        self.set_meta("search_runs", runs[-keep:])

    def event(self, sid, kind, message):
        self.db.execute("INSERT INTO events VALUES (?,?,?,?)", (self.clock(), sid, kind, message))
        self.db.commit()

    def events(self, limit=100):
        rows = self.db.execute("SELECT * FROM events ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
