"""Export a strategy as a TradingView Pine Script (v5) so you can SEE its trades on
your own TradingView charts.

How to use: TradingView -> open a chart of e.g. NVDA on the same candle size as the
strategy (5m / 15m), regular trading hours -> Pine Editor -> paste the file -> Add to chart.

Notes
  * News and event conditions (CPI/Fed days, earnings, news mood) don't exist on
    TradingView; they are switched off in the script and listed at the top.
  * Account-wide rules (daily loss limit, max open positions across stocks) are
    not simulated by TradingView - only the per-stock rules are.
  * Indicator start-up values differ slightly, so the first days on a chart may differ.
"""
from .config import TF_MINUTES
from .genome import BINARY, describe

UNSUPPORTED = {"macro", "fomc", "earn", "earn_next", "sent", "newsv", "mkt_sent",
               # need multi-day or multi-candle memory that this simple export doesn't rebuild
               "fvg_bull", "fvg_bear", "dloc", "pd_ret", "vol_reg", "wk_trend"}


def _feat(name, p, bpd):
    """Pine expression for a feature."""
    if name == "rsi":
        return f"ta.rsi(close, {p})"
    if name == "zscore":
        return f"(close - ta.sma(close, {p})) / ta.stdev(close, {p}, false)"
    if name == "mom":
        return f"math.log(close / close[{p}]) / (ta.stdev(lr, 100, false) * math.sqrt({p}))"
    if name == "trend":
        return f"(close - ta.sma(close, {p})) / atr14"
    if name == "macd":
        return f"(ta.ema(close, {p}) - ta.ema(close, {p * 4})) / atr14"
    if name == "stoch":
        return f"100 * (close - ta.lowest(low, {p})) / (ta.highest(high, {p}) - ta.lowest(low, {p}))"
    if name == "ibs":
        return "(close - low) / (high - low)"
    if name == "volz":
        return "volz"
    if name == "volat":
        return f"ta.stdev(lr, {p}, false) / ta.stdev(lr, {int(20 * bpd)}, false)"
    if name == "tod":
        return "tod"
    if name == "vwap":
        return "(close - ta.vwap(hlc3)) / atr14"
    if name == "orb_hi":
        return f"(tod >= {p} ? (close - orh{p}) / atr14 : na)"
    if name == "orb_lo":
        return f"(tod >= {p} ? (close - orl{p}) / atr14 : na)"
    if name == "dchg":
        return "math.log(close / dayOpen) / dvol"
    if name == "gap":
        return "math.log(dayOpen / prevClose) / dvol"
    if name == "pdh":
        return "(close - prevHigh) / atr14"
    if name == "pdl":
        return "(close - prevLow) / atr14"
    if name == "sweep_lo":
        ref = "prevLow" if p == 1 else f"ta.lowest(low, {p})[1]"
        return f"((low < {ref} and close > {ref}) ? ({ref} - low) / atr14 : 0.0)"
    if name == "sweep_hi":
        ref = "prevHigh" if p == 1 else f"ta.highest(high, {p})[1]"
        return f"((high > {ref} and close < {ref}) ? (high - {ref}) / atr14 : 0.0)"
    if name == "swing_hi":
        return f"(close - ta.highest(high, {p})[1]) / atr14"
    if name == "swing_lo":
        return f"(close - ta.lowest(low, {p})[1]) / atr14"
    if name == "body":
        return f"(close - open[{p - 1}]) / atr14"
    if name == "flow":
        return (f"(math.sum(((close - low) - (high - close)) / math.max(high - low, syminfo.mintick) * volume, {p})"
                f" / math.sum(volume, {p}))")
    if name == "mkt":
        return "qqqDchg"
    if name == "mkt_vwap":
        return "qqqVwap"
    raise KeyError(name)


def _cond(c, bpd, idx):
    """Returns (boolean variable name, declaration lines) - every ta.* call runs on every candle."""
    if c["f"] in UNSUPPORTED:
        return "true", None
    x = _feat(c["f"], c["p"], bpd)
    v = c["v"]
    test = {"<": f"f{idx} < {v}", ">": f"f{idx} > {v}",
            "xa": f"ta.crossover(f{idx}, {v})", "xb": f"ta.crossunder(f{idx}, {v})"}[c["op"]]
    return f"c{idx}", f"f{idx} = {x}\nc{idx} = {test}"


def to_pine(g, sid, version, bench="QQQ"):
    tfm = TF_MINUTES[g["tf"]]
    bpd = 390 / tfm
    decls, ent = [], []
    skipped = []
    for i, c in enumerate(g["entry"]):
        expr, decl = _cond(c, bpd, i)
        if decl is None:
            skipped.append(c["f"])
        else:
            decls.append(decl)
        ent.append(expr)
    ex_expr = "false"
    if g.get("exit"):
        ex_expr, decl = _cond(g["exit"], bpd, 9)
        if decl is None:
            skipped.append(g["exit"]["f"])
            ex_expr = "false"
        else:
            decls.append(decl)
    orb = sorted({c["p"] for c in g["entry"] + ([g["exit"]] if g.get("exit") else []) if c["f"] in ("orb_hi", "orb_lo")})
    orb_lines = []
    for p in orb:
        orb_lines += [f"var float orh{p} = na", f"var float orl{p} = na",
                      f"orh{p} := newDay ? high : (tod <= {p} ? math.max(orh{p}, high) : orh{p})",
                      f"orl{p} := newDay ? low : (tod <= {p} ? math.min(orl{p}, low) : orl{p})"]
    long = g["dir"] == "long"
    header = "\n".join("// " + ln for ln in describe(g).splitlines())
    skip_note = ("// NOTE: these conditions use news/event data TradingView doesn't have and are IGNORED here: "
                 + ", ".join(sorted(set(skipped)))) if skipped else "// All conditions are reproduced."
    nl = "\n"
    return f"""//@version=5
// Nasdaq day-trading bot - strategy {sid} v{version}
{header}
{skip_note}
// Use on a {tfm}-minute chart, regular trading hours.
strategy("Bot {sid} v{version}", overlay=true, pyramiding=0, initial_capital=100000,
     default_qty_type=strategy.percent_of_equity, default_qty_value=25,
     commission_type=strategy.commission.percent, commission_value=0.02, slippage=0)

// ---------- shared helpers ----------
atr14 = ta.rma(ta.tr(true), 14)
lr = math.log(close / close[1])
tod = (hour(time_close, "America/New_York") * 60 + minute(time_close, "America/New_York")) - 570
newDay = ta.change(time("D")) != 0
var float dayOpen = na
dayOpen := newDay ? open : dayOpen
dvol = request.security(syminfo.tickerid, "D", ta.stdev(math.log(close / open), 20, false)[1], lookahead=barmerge.lookahead_on)
prevClose = request.security(syminfo.tickerid, "D", close[1], lookahead=barmerge.lookahead_on)
prevHigh = request.security(syminfo.tickerid, "D", high[1], lookahead=barmerge.lookahead_on)
prevLow = request.security(syminfo.tickerid, "D", low[1], lookahead=barmerge.lookahead_on)
qqqClose = request.security("NASDAQ:{bench}", timeframe.period, close)
qqqDayOpen = request.security("NASDAQ:{bench}", "D", open, lookahead=barmerge.lookahead_on)
qqqDvol = request.security("NASDAQ:{bench}", "D", ta.stdev(math.log(close / open), 20, false)[1], lookahead=barmerge.lookahead_on)
qqqDchg = math.log(qqqClose / qqqDayOpen) / qqqDvol
qqqVwap = request.security("NASDAQ:{bench}", timeframe.period, (close - ta.vwap(hlc3)) / ta.rma(ta.tr(true), 14))

// relative volume vs the same time of day over the previous 20 sessions
var int bod = 0
bod := newDay ? 0 : bod + 1
var matrix<float> volHist = matrix.new<float>(100, 20, na)
lv = math.log(1 + volume)
float vsum = 0.0
int vn = 0
for k = 0 to 19
    x = matrix.get(volHist, math.min(bod, 99), k)
    if not na(x)
        vsum += x
        vn += 1
volz = vn >= 5 ? lv - vsum / vn : na
for k = 19 to 1
    matrix.set(volHist, math.min(bod, 99), k, matrix.get(volHist, math.min(bod, 99), k - 1))
matrix.set(volHist, math.min(bod, 99), 0, lv)
{nl.join(orb_lines)}

// ---------- strategy conditions ----------
{nl.join(decls)}
entrySignal = {" and ".join(ent)}
exitSignal = {ex_expr}

// ---------- day-trading rules ----------
minsLeft = 390 - tod
flatNext = minsLeft <= {tfm}
blockNext = flatNext or minsLeft < 20
var int tradesToday = 0
var int lossesToday = 0
if newDay
    tradesToday := 0
    lossesToday := 0
if strategy.closedtrades > strategy.closedtrades[1]
    if strategy.closedtrades.profit(strategy.closedtrades - 1) < 0
        lossesToday += 1

var float entryAtr = na
var float best = na
inPos = strategy.position_size != 0
if not inPos and entrySignal and not blockNext and tradesToday < {g['per_day']} and lossesToday < 2
    entryAtr := atr14
    strategy.entry("E", {"strategy.long" if long else "strategy.short"})
    tradesToday += 1

if inPos
    ep = strategy.position_avg_price
    best := na(best) ? ep : {"math.max" if long else "math.min"}(best, close)
    stopPx = ep {"-" if long else "+"} {g['stop']} * entryAtr
    trail = {g['trail']}
    if trail > 0
        stopPx := {"math.max" if long else "math.min"}(stopPx, best {"-" if long else "+"} trail * atr14[1])
    tpPx = ep {"+" if long else "-"} {g['tp']} * entryAtr
    strategy.exit("X", "E", stop=stopPx, limit=tpPx)
    barsIn = bar_index - strategy.opentrades.entry_bar_index(0) + 1
    if flatNext or exitSignal or barsIn >= {g['hold']}
        strategy.close("E", comment=flatNext ? "end of day" : (exitSignal ? "exit signal" : "time limit"))
else
    best := na
"""


def export(registry, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for f in folder.glob("bot_*.pine"):
        f.unlink()
    out = []
    bench = "QQQ"
    for s in registry.list(["champion", "challenger"]):
        path = folder / f"bot_{s['status']}_{s['id']}_v{s['version']}.pine"
        path.write_text(to_pine(s["genome"], s["id"], s["version"], bench), encoding="utf-8")
        out.append(path)
    return out
