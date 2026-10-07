"""Builds reports/dashboard.html (open in any browser; refreshes itself every minute)."""
import html
from datetime import datetime

from .config import REPORT_DIR
from .genome import describe


def _svg(curve, w=300, h=64):
    if not curve or len(curve) < 2:
        return '<div class="muted small">no paper-trading days yet</div>'
    ys = [p[1] for p in curve]
    lo, hi = min(ys + [1.0]), max(ys + [1.0])
    rng = (hi - lo) or 1e-9
    pts = " ".join(f"{i / (len(ys) - 1) * w:.1f},{h - (y - lo) / rng * h:.1f}" for i, y in enumerate(ys))
    base = h - (1.0 - lo) / rng * h
    col = "var(--up)" if ys[-1] >= 1 else "var(--down)"
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" class="spark">'
            f'<line x1="0" x2="{w}" y1="{base:.1f}" y2="{base:.1f}" class="base"/>'
            f'<polyline points="{pts}" style="stroke:{col}"/></svg>')


def _ny(ts):
    try:
        import pandas as pd
        return pd.Timestamp(ts).tz_convert("America/New_York").strftime("%b %d %H:%M ET")
    except Exception:
        return str(ts)[:16]


def _pct(x):
    return f"{x:+.2%}" if isinstance(x, (int, float)) else "-"


def _usd(x):
    return f"${x:+,.0f}" if isinstance(x, (int, float)) else "-"


def _card(s, reg):
    st = reg.paper_state(s["id"]) or {}
    ps = st.get("stats", {})
    rep = s["report"]
    r, hd = rep.get("research", {}), rep.get("holdout", {})
    acts = []
    for sym, t in (st.get("targets") or {}).items():
        if t["action"] in ("BUY", "SELL_SHORT"):
            acts.append(f"<b>{sym}</b> {t['action'].replace('_', ' ')} next candle (TP {t['tp']}, SL {t['stop']})")
        elif t["action"] == "HOLD":
            acts.append(f"<b>{sym}</b> holding (TP {t.get('tp')}, SL {t.get('stop')})")
        elif t["action"] == "EXIT":
            acts.append(f"<b>{sym}</b> exit next candle")
    checks = "".join(
        f'<span class="chip {"ok" if c["pass"] else "bad"}" title="{html.escape(c["detail"])}">'
        f'{html.escape(n.replace("_", " "))}</span>' for n, c in rep.get("checks", {}).items())
    trades = reg.paper_trades(s["id"])[-8:][::-1]
    rows = "".join(
        f"<tr><td>{html.escape(t['symbol'])}</td><td class='muted'>{_ny(t['entry_time'])}</td>"
        f"<td>{html.escape(t['reason'])}</td><td class='{'up' if t['ret'] > 0 else 'down'}'>{(t['rmult'] or 0):+.2f}R</td></tr>"
        for t in trades)
    return f"""
<div class="card">
  <div class="row"><span class="tag {s['status']}">{s['status']}</span><b>{s['id']}</b>
    <span class="muted">v{s['version']}</span><span class="health">{html.escape(str(s.get('health') or ''))}</span></div>
  <pre>{html.escape(describe(s['genome']))}</pre>
  <div class="grid">
    <div><div class="lbl">Paper trading (live, new data)</div><b class="{'up' if ps.get('pnl_usd', 0) >= 0 else 'down'}">{_usd(ps.get('pnl_usd', 0))}</b>
      ({_pct(ps.get('total_return', 0))}) · {ps.get('trades', 0)} trades · win {ps.get('win_rate', 0):.0%} · PF {ps.get('profit_factor', 0):.2f} · avg {ps.get('avg_R', 0):+.2f}R · max DD {_pct(ps.get('max_dd', 0))}</div>
    <div><div class="lbl">Backtest</div>Sharpe {r.get('sharpe', 0):.2f} · PF {r.get('profit_factor', 0):.2f} · win {r.get('win_rate', 0):.0%} · {r.get('trades_per_day', 0)} trades/day · max DD {_pct(r.get('max_dd'))}</div>
    <div><div class="lbl">Never-seen-data exam</div>{('Sharpe %.2f · return %s · %d trades' % (hd.get('sharpe', 0), _pct(hd.get('total_return')), hd.get('trades', 0))) if hd else 'n/a (rewritten version - proving itself live)'}</div>
  </div>
  {_svg(st.get('equity'))}
  <div class="small">{' · '.join(acts) if acts else '<span class="muted">flat - waiting for a setup</span>'}</div>
  {'<table class="small">' + rows + '</table>' if rows else ''}
  <div class="checks">{checks}</div>
</div>"""


def build(cfg, registry, path=None, extra_html=""):
    path = path or (REPORT_DIR / "dashboard.html")
    allst = registry.list()
    by = {}
    for s in allst:
        by.setdefault(s["status"], []).append(s)
    total_pnl = sum(((registry.paper_state(s["id"]) or {}).get("stats", {}) or {}).get("pnl_usd", 0) or 0
                    for s in by.get("champion", []))
    evs = registry.events(100)
    ev_rows = "".join(
        f"<tr><td class='muted'>{html.escape(e['ts'][:16].replace('T', ' '))}</td>"
        f"<td><span class='tag {html.escape(e['kind'])}'>{html.escape(e['kind'])}</span></td>"
        f"<td>{html.escape(e['strategy'] or '')}</td><td>{html.escape(e['message'])}</td></tr>" for e in evs)
    sections = ""
    for status, title in [("champion", "Trading now"), ("challenger", "Rewritten versions proving themselves"),
                          ("reserve", "Validated, waiting for a slot")]:
        items = by.get(status, [])
        sections += f"<h2>{title} <span class='muted'>({len(items)})</span></h2>"
        sections += "".join(_card(s, registry) for s in items) or "<p class='muted'>none</p>"
    slip = registry.meta("learned_slippage_bps")
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Nasdaq Bot</title>
<meta http-equiv="refresh" content="60">
<style>
:root{{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#77756f;--line:#e4e2dc;--up:#1f8a5b;--down:#c2413b;--accent:#3b5bdb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161615;--card:#1f1f1d;--ink:#ecebe6;--muted:#9a978f;--line:#33322f;--up:#45c08a;--down:#ef6a63;--accent:#7d97ff}}}}
body{{background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0 auto;padding:24px 16px;max-width:1100px}}
h1{{margin:0 0 4px}} h2{{margin:28px 0 10px;font-size:18px}} .muted{{color:var(--muted)}} .small{{font-size:12px}}
.up{{color:var(--up)}} .down{{color:var(--down)}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}} .kpi b{{font-size:22px;display:block}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:12px}}
.row{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}} .health{{margin-left:auto;color:var(--muted)}}
pre{{white-space:pre-wrap;background:var(--bg);border-radius:6px;padding:8px 10px;font-size:12.5px;margin:10px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;font-size:13px}}
.lbl{{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}}
.spark{{width:100%;height:64px;margin:8px 0}} .spark polyline{{fill:none;stroke-width:1.6}} .spark .base{{stroke:var(--line);stroke-dasharray:3 3}}
.tag,.chip{{font-size:11px;padding:2px 7px;border-radius:99px;border:1px solid var(--line);white-space:nowrap}}
.tag.champion,.tag.promoted,.tag.deployed,.tag.order{{background:var(--up);color:#fff;border:0}}
.tag.challenger,.tag.rewrite,.tag.weakspot,.tag.streak{{background:var(--accent);color:#fff;border:0}}
.tag.degraded,.tag.broken,.tag.retired,.tag.error,.tag.flatten{{background:var(--down);color:#fff;border:0}}
.chip.ok{{color:var(--up)}} .chip.bad{{color:var(--down)}} .checks{{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}}
table{{width:100%;border-collapse:collapse;font-size:13px}} td{{border-top:1px solid var(--line);padding:5px 8px;vertical-align:top}}
.wrap{{overflow-x:auto}}
</style></head><body>
<h1>Nasdaq Day-Trading Bot</h1>
<div class="muted">Updated {datetime.now().strftime('%Y-%m-%d %H:%M')} · broker: {cfg['paper']['broker']} (paper money only) · trades: {', '.join(cfg['market']['symbols'])}</div>
<div class="kpis">
 <div class="kpi"><span class="muted small">Paper P&amp;L (current strategies)</span><b class="{'up' if total_pnl >= 0 else 'down'}">{_usd(total_pnl)}</b></div>
 <div class="kpi"><span class="muted small">Strategies tested (lifetime)</span><b>{registry.meta('total_trials', 0):,}</b></div>
 <div class="kpi"><span class="muted small">Trading now</span><b>{len(by.get('champion', []))}</b></div>
 <div class="kpi"><span class="muted small">Rewrites being tested</span><b>{len(by.get('challenger', []))}</b></div>
 <div class="kpi"><span class="muted small">Real slippage per side</span><b>{f'{slip:.2f} bps' if slip is not None else 'measuring'}</b></div>
</div>
{extra_html}
{sections}
<h2>What the bot has been doing</h2><div class="wrap"><table>{ev_rows or '<tr><td class="muted">nothing yet</td></tr>'}</table></div>
<p class="muted small">Paper trading only. Simulated and past results do not guarantee future results. Not financial advice.</p>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path
