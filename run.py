"""Nasdaq day-trading bot - start here.

    python run.py              -> menu
    python run.py start        -> run the bot (trades while market is open, improves itself after the close)
    python run.py check        -> check API keys
    python run.py download     -> download/update prices, news and event calendar
    python run.py discover --minutes 60
    python run.py status | report | tradingview | selftest | bench
"""
import argparse
import sys
import webbrowser


def speed_warning():
    from trader.backtest import HAVE_NUMBA
    if not HAVE_NUMBA:
        print("\n  NOTE: the speed-up package 'numba' is not installed - testing runs ~40x slower.\n"
              "  Fix: run the setup file again, or:  pip install numba\n")


def trading_here(cfg):
    if cfg.get("trading_host") == "github":
        print("\n  Trading runs on GitHub now (your PC can be off) - this PC only searches for strategies.\n"
              "  Use SEARCH_AND_SEND_TO_GITHUB.bat. Two bots on one Alpaca account would fight each other.\n")
        return False
    return True


def send_to_github(bot):
    if bot.cfg.get("trading_host") == "github":
        from trader import cloud
        cloud.publish(bot.cfg, bot.reg, bot.log)


def ensure_dashboard_service(cfg):
    """Windows: make sure your Projects Dashboard's background service is running
    (it shows live data from both Alpaca bots). The first time, it also runs the
    dashboard's own setup: Desktop "Projects" icon + start with Windows."""
    import os
    import subprocess
    import urllib.request
    from pathlib import Path
    if os.name != "nt":
        return
    folder = Path(os.path.expanduser((cfg.get("dashboard") or {}).get("status_folder") or ""))
    if not (folder / "server.py").exists() or not (folder / "launch.vbs").exists():
        return
    try:
        urllib.request.urlopen("http://localhost:8765/ping.gif", timeout=2)
        return                                   # already running
    except Exception:
        pass
    try:
        if not (folder / "python_path.txt").exists() and (folder / "setup.ps1").exists():
            print("Setting up your Projects Dashboard (Desktop icon + start with Windows) ...")
            subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                            str(folder / "setup.ps1")], cwd=str(folder), timeout=120)
        subprocess.Popen(["wscript.exe", str(folder / "launch.vbs"), "/silent"], cwd=str(folder))
        print("Projects Dashboard service started: open the 'Projects' icon on your Desktop.")
    except Exception as e:
        print(f"  (could not start the Projects Dashboard service: {e})")


MODES = {   # symbols, biggest position %, assumed slippage per side (bps), label
    "qqq": ("[QQQ]", 100, 1, "QQQ only (the Nasdaq-100 fund)"),
    "stocks": ("[QQQ, NVDA, AAPL, MSFT, MU, AMD, AMZN, META, GOOGL, TSLA, INTC, AVGO, WMT, PLTR, CSCO, COST]", 25, 2,
               "QQQ + the 15 biggest Nasdaq stocks"),
}


def set_mode(which):
    """Switch what the bot trades by editing two lines of config.yaml (nothing else is touched)."""
    import re
    from pathlib import Path
    which = (which or "").lower()
    if which not in MODES:
        print("Use:  python run.py mode qqq   or   python run.py mode stocks")
        return 1
    syms, maxpos, slip, label = MODES[which]
    f = Path(__file__).resolve().parent / "config.yaml"
    text = f.read_text(encoding="utf-8")
    new, n1 = re.subn(r"(?m)^(\s*)symbols:\s*\[[^\]]*\]", lambda m: f"{m.group(1)}symbols: {syms}", text, count=1)
    new, n2 = re.subn(r"(?m)^(\s*)max_position_pct:.*$",
                      lambda m: f"{m.group(1)}max_position_pct: {maxpos:<15}# biggest single position, % of the account",
                      new, count=1)
    new, n3 = re.subn(r"(?m)^(\s*)slippage_bps:.*$",
                      lambda m: f"{m.group(1)}slippage_bps: {slip:<5}# per buy and per sell, in 0.01% (bot raises it if real fills are worse)",
                      new, count=1)
    if not (n1 and n2 and n3):
        print("Could not find the 'symbols' / 'max_position_pct' / 'slippage_bps' lines in config.yaml - nothing changed.")
        return 1
    if new == text:
        print(f"Already set to: {label}")
        return 0
    f.write_text(new, encoding="utf-8")
    # strategies were proven on the old set of symbols - don't let them trade the new one unproven
    from trader.config import load_config
    from trader.registry import Registry
    reg = Registry(load_config()["db_path"])
    n = 0
    for st in ("champion", "challenger", "reserve"):
        for s in reg.list([st]):
            reg.set_status(s["id"], "retired", note=f"switched to {label}")
            reg.event(s["id"], "retired", f"Retired: you switched the bot to trade {label}; "
                                         f"it was proven on different symbols.")
            n += 1
    print(f"Done. The bot now trades: {label}" + (f" ({n} strategies retired - they were proven on "
                                                     f"the old symbols)" if n else ""))
    print("Start the bot again with 3_SEARCH_THEN_START_WINDOWS.bat")
    return 0


def open_file(path):
    try:
        webbrowser.open(path.resolve().as_uri())
    except Exception:
        pass
    print(f"Opened: {path}")


def menu():
    from trader.app import Bot
    speed_warning()
    bot = Bot()
    ensure_dashboard_service(bot.cfg)
    while True:
        print("""
=================================================
      NASDAQ DAY-TRADING BOT   (paper money only)
=================================================
 1) Check my setup (Alpaca + FRED keys)
 2) Download / update data (prices, news, events)
 3) Find new strategies
 4) START THE BOT  (trades + learns automatically)
 5) Show status
 6) Open dashboard
 7) Export strategies to TradingView
 8) Self-test (proves it works, no internet needed)
 0) Quit
""")
        c = input("Choose a number: ").strip()
        try:
            if c == "1":
                bot.check_setup()
            elif c == "2":
                bot.download()
            elif c == "3":
                m = input("How many minutes should it search? [45]: ").strip() or "45"
                bot.download()
                bot.discover(minutes=float(m))
                send_to_github(bot)
            elif c == "4":
                if trading_here(bot.cfg):
                    bot.auto()
            elif c == "5":
                bot.status()
            elif c == "6":
                open_file(bot.report(force=True))
            elif c == "7":
                files = bot.export_tradingview()
                print(f"Wrote {len(files)} TradingView script(s) to the 'tradingview' folder."
                      if files else "No strategies trading yet - nothing to export.")
                for f in files:
                    print("  ", f.name)
            elif c == "8":
                from trader.selftest import main as st
                st()
            elif c == "0":
                return
        except KeyboardInterrupt:
            print("\nStopped. (Open Alpaca positions keep their take-profit and stop-loss orders.)")
        except Exception as e:
            print(f"\n! Something went wrong: {e}\n  Full details are in logs/bot.log")
            import logging
            import traceback
            logging.getLogger("nasdaq_bot").error(traceback.format_exc())


def main():
    if len(sys.argv) == 1:
        return menu()
    ap = argparse.ArgumentParser(description="Nasdaq day-trading bot")
    ap.add_argument("command", choices=["start", "check", "download", "discover", "heal", "status",
                                        "report", "tradingview", "selftest", "bench", "mode", "session", "publish"])
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--write", default=None)
    ap.add_argument("which", nargs="?", default=None, help="for 'mode': qqq or stocks")
    a = ap.parse_args()
    if a.command == "mode":
        return set_mode(a.which)
    speed_warning()
    if a.command == "selftest":
        from trader.selftest import main as st
        return st()
    from trader.app import Bot
    bot = Bot()
    if a.command in ("start", "discover"):
        ensure_dashboard_service(bot.cfg)
    if a.command == "start":
        if not trading_here(bot.cfg):
            return
        bot.auto()
    elif a.command == "session":
        bot.session()
    elif a.command == "check":
        bot.check_setup()
    elif a.command == "download":
        bot.download()
    elif a.command == "discover":
        bot.download()
        bot.discover(minutes=a.minutes or (None if a.trials else bot.cfg["discovery"]["default_minutes"]),
                     trials=a.trials, seed=a.seed)
        send_to_github(bot)
    elif a.command == "publish":
        from trader import cloud
        cloud.publish(bot.cfg, bot.reg, bot.log)
    elif a.command == "heal":
        bot.heal(seed=a.seed)
        bot.report(force=True)
    elif a.command == "status":
        bot.status()
    elif a.command == "report":
        open_file(bot.report(force=True))
    elif a.command == "bench":
        bot.bench(write=a.write)
    elif a.command == "tradingview":
        for f in bot.export_tradingview():
            print(f)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. (Open Alpaca positions keep their take-profit and stop-loss orders.)")
