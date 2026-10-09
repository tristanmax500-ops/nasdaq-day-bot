"""Loads config.yaml, fills in defaults, and checks values are sane."""
import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
LOG_DIR = ROOT / "logs"
TV_DIR = ROOT / "tradingview"

DEFAULTS = {
    "account": {"alpaca_key": "", "alpaca_secret": "", "fred_api_key": ""},
    "market": {
        "symbols": ["QQQ", "NVDA", "AAPL", "MSFT", "MU", "AMD", "AMZN", "META", "GOOGL",
                    "TSLA", "INTC", "AVGO", "WMT", "PLTR", "CSCO", "COST"],
        "benchmark": "QQQ",
        "timeframes": ["5Min", "15Min"],
        "history_start": "2021-01-01",
        "allow_short": False,
        "qqq_heavyweights": ["NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL"],
        # not traded - only read for news mood and earnings dates that move QQQ
        "context_symbols": ["NVDA", "AAPL", "MSFT", "MU", "AMD", "AMZN", "META", "GOOGL",
                            "TSLA", "INTC", "AVGO", "WMT", "PLTR", "CSCO", "COST"],
    },
    "day_trading": {
        "no_entries_last_minutes": 20, "flatten_minutes_before_close": 5,
        "risk_per_trade_pct": 0.5, "max_position_pct": 25, "max_open_positions": 6,
        "max_losses_per_symbol_day": 2, "max_daily_loss_pct": 2.0, "fomc_blackout": True,
    },
    "costs": {"slippage_bps": 2, "commission_bps": 0},
    "discovery": {"population": 300, "default_minutes": 45, "workers": 0, "holdout_fraction": 0.25,
                  "folds": 4, "min_trades": 150, "min_trades_per_day": 0.0, "mode": "templates+evolve", "finalists": 30, "holdout_slots": 5},
    "validation": {
        "min_folds_profitable": 3, "min_symbols_profitable": 0.55, "dsr_min": 0.95,
        "robustness_tests": 20, "robustness_min_profitable": 0.70, "random_entry_pctile": 0.95,
        "min_alpha_t": 1.0, "holdout_min_sharpe": 0.5, "holdout_min_ratio": 0.0, "holdout_min_trades": 30,
        "probation": True, "probation_size": 0.25, "probation_trades": 40,
        "holdout_random_pctile": 0.90, "min_profit_factor": 1.15,
    },
    "paper": {"broker": "alpaca", "target_champions": 3, "max_challengers": 3},
    "healing": {
        "min_trades_to_judge": 30, "degrade_pvalue": 0.05, "drawdown_break_mult": 1.5,
        "loss_streak_trigger": 6, "improve_every_trades": 40, "repair_trials": 3000,
        "promote_min_trades": 25,
    },
    "news": {"enabled": True, "history_start": "2021-01-01"},
    "data": {"plan": "free"},
    "auto": {"discovery_minutes_after_close": 60, "search_after_close": True},
    # "github" = trading runs on GitHub Actions; this PC only searches and sends strategies there
    "trading_host": "pc",
    "github": {"repo": ""},
    "dashboard": {"status_folder": "~/Documents/Projects Dashboard"},
    "db_path": str(DATA_DIR / "brain.db"),
}

TF_MINUTES = {"5Min": 5, "15Min": 15, "30Min": 30, "1Hour": 60}


def _merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.yaml"
    user = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, user)
    m = cfg["market"]
    m["symbols"] = list(dict.fromkeys(s.strip().upper() for s in m["symbols"]))
    m["benchmark"] = m["benchmark"].strip().upper()
    if m["benchmark"] not in m["symbols"]:
        m["symbols"].insert(0, m["benchmark"])
    bad = [t for t in m["timeframes"] if t not in TF_MINUTES]
    if bad:
        raise ValueError(f"config.yaml: unsupported timeframes {bad}. Use any of {list(TF_MINUTES)}")
    # environment variables override the file (handy, and keeps keys out of the file)
    a = cfg["account"]
    a["alpaca_key"] = os.environ.get("ALPACA_KEY", a.get("alpaca_key") or "").strip()
    a["alpaca_secret"] = os.environ.get("ALPACA_SECRET", a.get("alpaca_secret") or "").strip()
    a["fred_api_key"] = os.environ.get("FRED_API_KEY", a.get("fred_api_key") or "").strip()
    if os.environ.get("NASDAQ_STATUS_DIR"):          # GitHub: where the dashboard status file goes
        cfg.setdefault("dashboard", {})["status_folder"] = os.environ["NASDAQ_STATUS_DIR"]
        os.makedirs(os.environ["NASDAQ_STATUS_DIR"], exist_ok=True)
    for d in (DATA_DIR, REPORT_DIR, LOG_DIR, TV_DIR, DATA_DIR / "cache"):
        os.makedirs(d, exist_ok=True)
    return cfg


def watch_symbols(cfg):
    """Everything the bot reads news and earnings for: what it trades + QQQ's big companies."""
    m = cfg["market"]
    out = list(m["symbols"]) + list(m.get("qqq_heavyweights") or []) + list(m.get("context_symbols") or [])
    return list(dict.fromkeys(s.strip().upper() for s in out))


def cost_per_side(cfg, registry=None):
    """Slippage + commission per side. If the bot has measured worse real slippage
    from Alpaca fills, the measured value is used instead (never a better one)."""
    c = cfg["costs"]
    slip = float(c["slippage_bps"])
    if registry is not None:
        learned = registry.meta("learned_slippage_bps")
        if learned is not None:
            slip = max(slip, float(learned))
    return (slip + float(c["commission_bps"])) / 10000.0
