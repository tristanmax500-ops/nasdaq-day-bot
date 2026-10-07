"""Turns the champions' decisions into real Alpaca PAPER orders, with hard safety rules.

Every entry is a bracket order: market entry + take-profit limit + stop-loss stop,
all held by Alpaca's servers, so TP/SL work even if your computer goes to sleep.

Safety rules (enforced here, on top of the strategy's own rules):
  * flatten everything N minutes before the close (default 5) - never overnight
  * daily loss limit: account down X% today -> close everything, no new trades today
  * max open positions; one strategy per stock at a time
  * no new trades on stale data, in the last minutes of the day, or in the Fed blackout
  * every order is logged; duplicate orders for the same candle are impossible
It also measures real slippage on every fill and feeds it back into the cost model.
"""
import math
import time

import numpy as np
import pandas as pd

NY = "America/New_York"


class Broker:
    def __init__(self, cfg, api, registry, log=print):
        self.cfg = cfg
        self.api = api
        self.reg = registry
        self.log = log
        self.dt = cfg["day_trading"]

    # ------------------------------------------------------------------
    def flatten_all(self, why):
        """Cancel the bot's orders and close the bot's positions - only positions the bot
        opened itself (positions from you or another bot in the same account are left alone)."""
        mine = set(self.reg.owners())
        try:
            for o in self.api.orders("open", nested=False):
                if o.get("symbol") in mine:
                    try:
                        self.api.cancel_order(o["id"])
                    except Exception as e:
                        self.log(f"  ! cancel failed: {e}")
            for p in self.api.positions():
                if p["symbol"] in mine:
                    self._close_with_retry(p["symbol"])
        finally:
            for sym in list(self.reg.owners()):
                self.reg.set_owner(sym, None)
        self.reg.event(None, "flatten", f"Closed all positions: {why}")
        self.log(f"  FLATTEN ALL: {why}")

    def _cancel_symbol_orders(self, symbol):
        for o in self.api.orders("open", nested=False):
            if o.get("symbol") == symbol:
                try:
                    self.api.cancel_order(o["id"])
                except Exception as e:
                    self.log(f"  ! cancel {symbol} order failed: {e}")

    def _close_with_retry(self, symbol, tries=4):
        """Cancelling the TP/SL legs takes a moment to release the shares - retry the close."""
        for i in range(tries):
            try:
                self.api.close_position(symbol)
                return True
            except Exception as e:
                msg = str(e).lower()
                if "position does not exist" in msg or "404" in msg:
                    return True
                if i == tries - 1:
                    self.log(f"  ! could not close {symbol}: {e}")
                    self.reg.event(None, "error", f"could not close {symbol}: {str(e)[:200]}")
                    return False
                time.sleep(1.0 + i)

    def close_symbol(self, symbol, why):
        self._cancel_symbol_orders(symbol)
        self._close_with_retry(symbol)
        owner = self.reg.owner(symbol)
        self.reg.set_owner(symbol, None)
        self.reg.event(owner, "exit", f"Closed {symbol}: {why}")
        self.log(f"  close {symbol} ({why})")

    # ------------------------------------------------------------------
    def update_fills(self):
        """Record fill prices of our entry orders and learn real slippage."""
        for o in self.reg.unfilled_orders():
            try:
                r = self.api.order_by_client_id(o["client_id"])
            except Exception:
                continue
            st = r.get("status")
            if st == "filled" and r.get("filled_avg_price"):
                fill = float(r["filled_avg_price"])
                sign = 1 if o["side"] == "buy" else -1
                slip = sign * (fill / o["ref_price"] - 1) * 1e4        # + = paid more than expected
                self.reg.set_fill(o["client_id"], fill, slip)
            elif st in ("canceled", "expired", "rejected"):
                self.reg.set_fill(o["client_id"], None, None, status=st)

    def learn_costs(self):
        """Once a day: if real fills cost more than assumed, use the real number from now on."""
        self.update_fills()
        slips = self.reg.recent_slippage(200)
        if len(slips) >= 30:
            measured = float(np.median(slips))
            cfg_slip = float(self.cfg["costs"]["slippage_bps"])
            old = self.reg.meta("learned_slippage_bps")
            if measured > cfg_slip and (old is None or abs(measured - old) > 0.1):
                self.reg.set_meta("learned_slippage_bps", round(measured, 2))
                self.reg.event(None, "costs", f"Real fills cost {measured:.2f} bps vs {cfg_slip:.2f} assumed - "
                               "all backtests now use the real number.")

    # ------------------------------------------------------------------
    def sync(self, results, now=None):
        """results: {strategy_id: run_forward(...) output} for champions (latest candle)."""
        clock = self.api.clock()
        if not clock.get("is_open"):
            return
        now = pd.Timestamp(now) if now is not None else pd.Timestamp(clock["timestamp"]).tz_convert("UTC")
        close_t = pd.Timestamp(clock["next_close"]).tz_convert("UTC")
        mins_to_close = (close_t - now).total_seconds() / 60
        self.update_fills()
        acct = self.api.account()
        equity = float(acct["equity"])
        last_eq = float(acct.get("last_equity") or equity)
        day_pnl = equity / last_eq - 1 if last_eq > 0 else 0.0
        today = now.tz_convert(NY).date().isoformat()

        universe = set(self.cfg["market"]["symbols"])
        mine = set(self.reg.owners())
        if mins_to_close <= float(self.dt["flatten_minutes_before_close"]):
            if any(p["symbol"] in mine for p in self.api.positions()):
                self.flatten_all("end of day")
            return
        if day_pnl <= -float(self.dt["max_daily_loss_pct"]) / 100:
            if self.reg.meta("halted_day") != today:
                self.reg.set_meta("halted_day", today)
                self.flatten_all(f"daily loss limit hit ({day_pnl:+.2%})")
            return
        halted = self.reg.meta("halted_day") == today

        positions = {p["symbol"]: p for p in self.api.positions()}
        # leftovers from an earlier day (bot was off at the close) -> close them now
        owners_since = {r["symbol"]: r["since"] for r in self.reg.db.execute("SELECT * FROM broker_owner")}
        stale = [sym for sym in positions if sym in owners_since and
                 pd.Timestamp(owners_since[sym]).tz_convert(NY).date().isoformat() != today]
        foreign = sorted(sym for sym in positions if sym in universe and sym not in owners_since)
        if foreign and self.reg.meta("foreign_warned") != today:
            self.reg.set_meta("foreign_warned", today)
            self.reg.event(None, "warning", f"This Alpaca account holds {', '.join(foreign)} that the bot did not open. "
                           "It will not touch them and will not trade those symbols - but please give the bot "
                           "its own paper account so its results (and daily loss limit) are clean.")
        for sym in stale:
            self.close_symbol(sym, "left over from a previous day (bot was not running at the close)")
        if stale:
            positions = {p["symbol"]: p for p in self.api.positions()}
        open_orders = self.api.orders("open", nested=False)
        pending_syms = {o["symbol"] for o in open_orders if o.get("order_class") == "bracket"
                        and o.get("status") in ("new", "accepted", "pending_new", "partially_filled")}
        champs = {s["id"]: s for s in self.reg.list(["champion"])}

        # release stale ownership (bracket TP/SL already closed the position)
        for sym, owner in self.reg.owners().items():
            if sym not in positions and sym not in pending_syms:
                self.reg.set_owner(sym, None)
        owners = self.reg.owners()

        # exits decided by the strategy (exit signal, time limit, end of day, or sim says flat)
        for sym, owner in owners.items():
            if sym not in positions:
                continue
            res = results.get(owner)
            if owner not in champs:
                self.close_symbol(sym, "strategy no longer a champion")
                continue
            if res is None:
                continue
            tgt = res["targets"].get(sym)
            if tgt is None or tgt["action"] == "EXIT":
                self.close_symbol(sym, "strategy exit" if tgt else "strategy is flat")
        if halted:
            return

        # entries
        positions = {p["symbol"]: p for p in self.api.positions()}
        owners = self.reg.owners()
        n_open = len(positions) + len([s for s in pending_syms if s not in positions])
        per_strategy = equity / max(len(champs), 1)
        tfmin = {"5Min": 5, "15Min": 15, "30Min": 30, "1Hour": 60}
        for sid, res in results.items():
            if sid not in champs:
                continue
            g = champs[sid]["genome"]
            last = pd.Timestamp(res["stats"].get("last_candle"))
            if (now - last).total_seconds() > 60 * (tfmin[g["tf"]] + 3):
                continue          # stale data -> never trade on it
            for sym, tgt in res["targets"].items():
                if tgt["action"] not in ("BUY", "SELL_SHORT"):
                    continue
                if sym in owners or sym in positions or n_open >= int(self.dt["max_open_positions"]):
                    continue
                if mins_to_close < float(self.dt["no_entries_last_minutes"]):
                    continue
                cid = f"{sid}-{sym}-{int(last.timestamp())}"
                if self.reg.order_logged(cid):
                    continue
                try:
                    base = self.api.latest_trade(sym)
                except Exception:
                    base = tgt["ref_price"]
                d = tgt["side"]
                stop = round(base - d * tgt["stop_atr"] * tgt["atr"], 2)
                tp = round(base + d * tgt["tp_atr"] * tgt["atr"], 2)
                size = float(self.cfg["validation"].get("probation_size", 0.25)) \
                    if champs[sid]["report"].get("probation") else 1.0
                qty = math.floor(tgt["weight"] * per_strategy * size / base)
                if qty < 1 or (d > 0 and not (stop < base < tp)) or (d < 0 and not (tp < base < stop)):
                    continue
                side = "buy" if d > 0 else "sell"
                try:
                    self.api.submit_bracket(sym, qty, side, tp, stop, cid)
                    status = "submitted"
                    self.reg.set_owner(sym, sid)
                    owners[sym] = sid
                    n_open += 1
                    msg = f"{side.upper()} {qty} {sym} @ ~{base:.2f}  TP {tp:.2f}  SL {stop:.2f}"
                    self.reg.event(sid, "order", msg)
                    self.log(f"  order: {msg}  [{sid}]")
                except Exception as e:
                    status = "rejected"
                    self.reg.event(sid, "error", f"order for {sym} rejected: {str(e)[:200]}")
                    self.log(f"  ! order for {sym} rejected: {e}")
                self.reg.log_order(cid, sid, sym, side, qty, base, stop, tp, last.isoformat(), status)
