"""Swing agent — phase 3: each strategy alone, intraday-timed vs daily-only twin.

Usage (from the repo root; needs phases 1–2):
    apps/api/.venv/bin/python scripts/swing_agent_strategies.py run        # 4 strategies × 2 variants
    apps/api/.venv/bin/python scripts/swing_agent_strategies.py run --only pullback
    apps/api/.venv/bin/python scripts/swing_agent_strategies.py chart

One continuous run over the research window (2017-01 → 2023-12, 2016 is
indicator warm-up); metrics are split into train (2017–21) and validation
(2022–23). Nothing here is tuned; every run is appended to experiments.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("WARMUP_DISABLED", "1")
sys.path.insert(0, "apps/api")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.swing_agent import books as bk, config, engine, experiments, metrics as mx  # noqa: E402
from app.swing_agent import strategies as sg  # noqa: E402
from app.swing_agent.data import MarketData  # noqa: E402

CFG = config.load()
ROOT = config.root()
OUT = ROOT / "strategies"
OUT.mkdir(parents=True, exist_ok=True)
SEG = {"train": config.segment("train"), "validation": config.segment("validation")}


def say(*a: object) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def seg_metrics(res: dict, seg: str, bench: pd.Series) -> dict:
    lo, hi = SEG[seg]
    eq = res["equity"]["equity"]
    e = eq[(eq.index >= lo) & (eq.index <= hi)]
    prev = eq[eq.index < lo]
    if len(prev):                       # start the segment from the previous close
        e = pd.concat([prev.iloc[-1:], e])
    t = res["trades"]
    t = t[(t["entry_date"] >= lo) & (t["entry_date"] <= hi)] if len(t) else t
    ex = res["equity"]["exposure"] / res["equity"]["equity"]
    b = bench[(bench.index >= e.index[0]) & (bench.index <= hi)]
    return {"curve": mx.curve(e), "trades": mx.trades(t), "yearly": mx.yearly(e),
            "avg_exposure_pct": round(100 * float(ex[(ex.index >= lo) & (ex.index <= hi)].mean()), 1),
            "spy": mx.curve(b), "spy_yearly": mx.yearly(b)}


def step_run(only: list[str] | None) -> None:
    md = MarketData(segment="research")
    days, n_slots, idx = bk.session_axis(md.calendar, pd.Timestamp(CFG["data"]["history_start"]).date(),
                                         SEG["validation"][1].date())
    t0 = time.time()
    books = bk.build_all(md, CFG, days, idx, progress=lambda i, n: say(f"books {i}/{n}"))
    say(f"{len(books)} books in {time.time() - t0:.0f}s")
    ctx = bk.context(CFG, md.calendar, days, n_slots, idx, ROOT)
    start_si = next(i for i, d in enumerate(days) if d >= SEG["train"][0])
    end_si = len(days) - 1
    fx = bk.eurusd_at(days[start_si - 1], CFG["execution"]["eurusd_fallback"])
    capital = CFG["execution"]["initial_capital_eur"] * fx
    say(f"capital: €{CFG['execution']['initial_capital_eur']:,} × EURUSD {fx:.4f} = ${capital:,.0f}")
    spy = md.daily("SPY")
    bench = (spy["tr_close"] / spy["tr_close"].loc[days[start_si - 1]]) * capital
    summary = {"capital_usd": capital, "eurusd": fx, "runs": {}}
    for name in only or sg.NAMES:
        for variant, col in (("intraday", "entry"), ("daily_only", "setup_entry")):
            t1 = time.time()
            res = engine.run(books, ctx, {name: col}, CFG["execution"], CFG["costs"], capital, start_si, end_si)
            tag = f"{name}__{variant}"
            res["trades"].to_parquet(OUT / f"{tag}_trades.parquet")
            res["equity"].to_parquet(OUT / f"{tag}_equity.parquet")
            out = {seg: seg_metrics(res, seg, bench) for seg in SEG}
            if variant == "intraday" and len(res["trades"]):
                tr = res["trades"]
                for seg in SEG:
                    lo, hi = SEG[seg]
                    ts = tr[(tr["entry_date"] >= lo) & (tr["entry_date"] <= hi)]
                    out[seg]["by_market_regime"] = mx.by_group(ts, ["market_regime"])
                    out[seg]["by_micro"] = mx.by_group(ts, ["micro"])
                    out[seg]["by_market_x_micro"] = mx.by_group(ts, ["market_regime", "micro"], min_n=10)
                    out[seg]["by_stock_regime"] = mx.by_group(ts, ["stock_regime"])
            summary["runs"][tag] = out
            for seg in SEG:
                m = out[seg]
                experiments.log(3, "strategy_alone", tag, {"strategy": CFG["strategy"][name], "variant": variant,
                                                           "execution": CFG["execution"], "costs": CFG["costs"]},
                                seg, {"curve": m["curve"], "trades": {k: v for k, v in m["trades"].items()
                                                                      if k != "exit_reasons"}})
                c, tm = m["curve"], m["trades"]
                say(f"{tag:38s} [{seg:10s}] CAGR {c.get('cagr_pct')}% Sharpe {c.get('sharpe')} "
                    f"MDD {c.get('max_dd_pct')}% | n {tm.get('n')} win {tm.get('win_rate')} "
                    f"PF {tm.get('profit_factor')} R {tm.get('mean_r')} (t {tm.get('t_mean_r')}) "
                    f"costs {tm.get('costs_pct_of_gross_profit')}% | SPY {m['spy'].get('cagr_pct')}% "
                    f"Sharpe {m['spy'].get('sharpe')} ({time.time() - t1:.0f}s)")
    prev = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() and only else {}
    if prev:
        prev["runs"].update(summary["runs"])
        summary = prev | {"runs": prev["runs"]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))


def jittered(books: dict, rng: np.random.Generator, eps: float = 1e-5) -> dict:
    """Prices nudged by ±0.001% (far below a tick) and exact score ties broken
    at random: a run on these books is a run on 'the same' data. The spread
    of results across such runs is the path dependence of a 5-slot portfolio."""
    import dataclasses

    out = {}
    for k, b in books.items():
        z = 1 + eps * rng.standard_normal(len(b.o))
        sig = {n: {**d, "score": d["score"] + 1e-9 * rng.standard_normal(len(b.o))} for n, d in b.sig.items()}
        out[k] = dataclasses.replace(b, o=b.o * z, h=b.h * z, l=b.l * z, c=b.c * z, sig=sig)
    return out


def step_jitter(n: int, only: list[str] | None) -> None:
    md = MarketData(segment="research")
    days, n_slots, idx = bk.session_axis(md.calendar, pd.Timestamp(CFG["data"]["history_start"]).date(),
                                         SEG["validation"][1].date())
    books = bk.build_all(md, CFG, days, idx)
    ctx = bk.context(CFG, md.calendar, days, n_slots, idx, ROOT)
    start_si = next(i for i, d in enumerate(days) if d >= SEG["train"][0])
    summary = json.loads((OUT / "summary.json").read_text())
    capital = summary["capital_usd"]
    spy = md.daily("SPY")
    bench = (spy["tr_close"] / spy["tr_close"].loc[days[start_si - 1]]) * capital
    rng = np.random.default_rng(CFG["seed"])
    rows = []
    for j in range(n):
        jb = jittered(books, rng)
        for name in only or sg.NAMES:
            for variant, col in (("intraday", "entry"), ("daily_only", "setup_entry")):
                res = engine.run(jb, ctx, {name: col}, CFG["execution"], CFG["costs"], capital, start_si,
                                 len(days) - 1)
                for seg in SEG:
                    m = seg_metrics(res, seg, bench)
                    rows.append({"run": j, "strategy": name, "variant": variant, "segment": seg,
                                 "cagr": m["curve"].get("cagr_pct"), "sharpe": m["curve"].get("sharpe"),
                                 "max_dd": m["curve"].get("max_dd_pct"), "n": m["trades"].get("n"),
                                 "mean_r": m["trades"].get("mean_r"), "t": m["trades"].get("t_mean_r")})
        say(f"jitter {j + 1}/{n}")
    t = pd.DataFrame(rows)
    t.to_parquet(OUT / "jitter.parquet")
    agg = t.groupby(["strategy", "variant", "segment"]).agg(
        cagr_med=("cagr", "median"), cagr_min=("cagr", "min"), cagr_max=("cagr", "max"),
        sharpe_med=("sharpe", "median"), sharpe_min=("sharpe", "min"), sharpe_max=("sharpe", "max"),
        mean_r_med=("mean_r", "median"), t_med=("t", "median"), n_med=("n", "median")).round(3)
    agg.to_csv(OUT / "jitter_summary.csv")
    for (sname, var), g in t.groupby(["strategy", "variant"]):
        for seg, gg in g.groupby("segment"):
            experiments.log(3, "strategy_alone_jitter", f"{sname}__{var}", {"runs": n, "eps": 1e-5}, seg,
                            {"cagr_median": float(gg["cagr"].median()), "sharpe_median": float(gg["sharpe"].median()),
                             "mean_r_median": float(gg["mean_r"].median())})
    print(agg.to_string())


CHART = """<!doctype html><html lang="pt"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Estratégias sozinhas</title>
<style>
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--page:#f9f9f7;--text-primary:#0b0b0b;--text-secondary:#52514e;--muted:#898781;
 --grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--band:rgba(11,11,11,.06)}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;
 --text-primary:#fff;--text-secondary:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--band:rgba(255,255,255,.07)}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;--text-primary:#fff;--text-secondary:#c3c2b7;
 --grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--band:rgba(255,255,255,.07)}
body{margin:0;background:var(--page)}
.viz-root{font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--text-primary);background:var(--page);padding:16px;max-width:1100px;margin:0 auto}
.card{background:var(--surface-1);border:1px solid var(--ring);border-radius:12px;padding:16px}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--text-secondary);margin:0 0 12px}
.bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:8px}
button{font:inherit;border:1px solid var(--ring);background:var(--surface-1);color:var(--text-primary);border-radius:8px;padding:4px 10px;cursor:pointer}
button[aria-pressed=true]{background:var(--text-primary);color:var(--surface-1)}
.legend{display:flex;flex-wrap:wrap;gap:14px;color:var(--text-secondary)}.chip{display:inline-flex;align-items:center;gap:6px}
.sw{width:14px;height:3px;border-radius:2px}
svg{display:block;width:100%;height:auto;overflow:visible}.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.gl{stroke:var(--grid)}.lbl{fill:var(--text-secondary);font-size:12px}
.tip{position:fixed;pointer-events:none;background:var(--surface-1);border:1px solid var(--ring);border-radius:8px;padding:8px 10px;font-size:12px;
 box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;min-width:190px}.tip .r{display:flex;justify-content:space-between;gap:12px;color:var(--text-secondary)}
.tip .r span:last-child{color:var(--text-primary);font-variant-numeric:tabular-nums}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;margin-top:8px;font-size:13px}td,th{padding:4px 8px;text-align:right;border-bottom:1px solid var(--grid)}
th:first-child,td:first-child{text-align:left}details{margin-top:12px;color:var(--text-secondary)}
</style></head><body><div class="viz-root"><div class="card">
<h1>Cada estratégia sozinha vs SPY — capital de 10 000 € (em USD), 2017–2023</h1>
<p class="sub">Índice 100 no início. Faixas cinzentas: regime diário de alta volatilidade. Treino 2017–2021, validação 2022–2023. Custos incluídos; sem afinação.</p>
<div class="bar"><span>Entrada:</span><button id="b-intraday" aria-pressed="true">gatilho 15m</button><button id="b-daily_only" aria-pressed="false">só diário (10:00)</button></div>
<div class="legend" id="legend"></div>
<svg id="chart" viewBox="0 0 1000 520" role="img" aria-label="Curvas de capital e drawdown por estratégia"></svg>
<details open><summary>Tabela: treino | validação</summary><table id="tbl"></table></details>
</div><div class="tip" id="tip"></div></div>
<script>
const D = __DATA__;
const NAMES = {pullback:"Pullback",vwap_reversion:"Reversão VWAP",compression_breakout:"Breakout compressão",gap_go:"Gap and go"};
const KEYS = ["pullback","vwap_reversion","compression_breakout","gap_go"], COL = {pullback:"--s1",vwap_reversion:"--s2",compression_breakout:"--s3",gap_go:"--s4"};
const css = n => getComputedStyle(document.querySelector('.viz-root')).getPropertyValue(n).trim();
const svg = document.getElementById('chart'), NS='http://www.w3.org/2000/svg';
const W=1000, L=48, R=150, P1={y:20,h:300}, P2={y:360,h:130};
const t=D.dates.map(d=>Date.parse(d+'T00:00:00Z')), t0=t[0], t1=t[t.length-1], X=v=>L+(v-t0)/(t1-t0)*(W-L-R);
function el(n,a){const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);svg.appendChild(e);return e}
let variant='intraday';
function draw(){
 svg.innerHTML='';
 const S = KEYS.map(k=>({k, v:D.series[k+'__'+variant]})), spy=D.spy;
 let lo=Infinity, hi=-Infinity; [...S.map(s=>s.v),spy].forEach(a=>a.forEach(v=>{if(v!=null){lo=Math.min(lo,v);hi=Math.max(hi,v)}}));
 const ly=v=>P1.y+P1.h-(Math.log(v)-Math.log(lo*0.97))/(Math.log(hi*1.03)-Math.log(lo*0.97))*P1.h;
 const dy=v=>P2.y+(-v)/40*P2.h;
 D.bands.forEach(([a,b])=>{const x0=X(Date.parse(a+'T00:00:00Z')),x1=X(Date.parse(b+'T00:00:00Z'));[P1,P2].forEach(p=>el('rect',{x:x0,y:p.y,width:Math.max(1,x1-x0),height:p.h,fill:css('--band')}))});
 [80,100,125,150,200,250].forEach(v=>{if(v<lo*0.97||v>hi*1.03)return;el('line',{x1:L,x2:W-R,y1:ly(v),y2:ly(v),class:'gl'});const tx=el('text',{x:L-6,y:ly(v)+4,'text-anchor':'end',class:'tick'});tx.textContent=v});
 [0,-10,-20,-30,-40].forEach(v=>{el('line',{x1:L,x2:W-R,y1:dy(v),y2:dy(v),class:'gl'});const tx=el('text',{x:L-6,y:dy(v)+4,'text-anchor':'end',class:'tick'});tx.textContent=v+'%'});
 const tl=el('text',{x:L,y:P1.y-6,class:'lbl'});tl.textContent='Capital (índice 100, escala log)';
 const tl2=el('text',{x:L,y:P2.y-8,class:'lbl'});tl2.textContent='Drawdown';
 for(let yr=2017;yr<=2024;yr++){const x=X(Date.UTC(yr,0,1));if(x<L||x>W-R)continue;const tx=el('text',{x,y:P2.y+P2.h+16,'text-anchor':'middle',class:'tick'});tx.textContent=yr}
 const xv=X(Date.UTC(2022,0,1));el('line',{x1:xv,x2:xv,y1:P1.y,y2:P2.y+P2.h,stroke:css('--text-primary'),'stroke-width':1.5});
 const tv=el('text',{x:xv+5,y:P1.y+12,class:'lbl'});tv.textContent='validação →';
 const path=(a,f)=>a.map((v,i)=>v==null?'':(i&&a[i-1]!=null?'L':'M')+X(t[i]).toFixed(1)+','+f(v).toFixed(1)).join('');
 const ddOf=a=>{let m=-Infinity;return a.map(v=>{if(v==null)return null;m=Math.max(m,v);return 100*(v/m-1)})};
 el('path',{d:path(spy,ly),fill:'none',stroke:css('--text-primary'),'stroke-width':1.5,'stroke-dasharray':'5 4'});
 el('path',{d:path(ddOf(spy),dy),fill:'none',stroke:css('--text-primary'),'stroke-width':1,'stroke-dasharray':'5 4'});
 const ends=[];
 S.forEach(s=>{el('path',{d:path(s.v,ly),fill:'none',stroke:css(COL[s.k]),'stroke-width':2,'stroke-linejoin':'round'});
  el('path',{d:path(ddOf(s.v),dy),fill:'none',stroke:css(COL[s.k]),'stroke-width':1.5});ends.push([s.v[s.v.length-1],NAMES[s.k],COL[s.k]])});
 ends.push([spy[spy.length-1],'SPY','--text-primary']);
 ends.sort((a,b)=>b[0]-a[0]); let last=-1e9;
 ends.forEach(([v,n,c])=>{let y=Math.max(ly(v)+4,last+14);last=y;const tx=el('text',{x:W-R+6,y,class:'lbl'});tx.textContent=n+' '+v.toFixed(0);tx.style.fill=css('--text-secondary')});
 document.getElementById('legend').innerHTML=KEYS.map(k=>`<span class="chip"><span class="sw" style="background:var(${COL[k]})"></span>${NAMES[k]}</span>`).join('')+
  '<span class="chip"><span class="sw" style="background:var(--text-primary)"></span>SPY (tracejado)</span>';
 const cross=el('line',{y1:P1.y,y2:P2.y+P2.h,stroke:css('--text-secondary'),visibility:'hidden'});
 const hit=el('rect',{x:L,y:P1.y,width:W-L-R,height:P2.y+P2.h-P1.y,fill:'transparent'}), tip=document.getElementById('tip');
 hit.onmousemove=ev=>{const b=svg.getBoundingClientRect(),vx=(ev.clientX-b.left)/b.width*W,tv=t0+(vx-L)/(W-L-R)*(t1-t0);
  let i=0,j=t.length-1;while(j-i>1){const m=(i+j)>>1;if(t[m]<tv)i=m;else j=m} i=(tv-t[i]<t[j]-tv)?i:j; const x=X(t[i]);
  cross.setAttribute('x1',x);cross.setAttribute('x2',x);cross.setAttribute('visibility','visible');
  tip.innerHTML=`<b>${D.dates[i]}</b>`+S.map(s=>`<div class="r"><span>${NAMES[s.k]}</span><span>${s.v[i]!=null?s.v[i].toFixed(1):'—'}</span></div>`).join('')+
   `<div class="r"><span>SPY</span><span>${spy[i].toFixed(1)}</span></div>`;tip.style.display='block';
  tip.style.left=(ev.clientX+210>innerWidth?ev.clientX-214:ev.clientX+14)+'px';tip.style.top=(ev.clientY+12)+'px'};
 hit.onmouseleave=()=>{tip.style.display='none';cross.setAttribute('visibility','hidden')};
 const m=D.metrics, f=(x,d=1)=>x==null?'—':(+x).toFixed(d);
 document.getElementById('tbl').innerHTML='<tr><th>Estratégia</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Trades</th><th>R/trade (t)</th><th>Custos % lucro bruto</th></tr>'+
  [...KEYS.map(k=>[NAMES[k],m[k+'__'+variant]]),['SPY',m.spy]].map(([n,v])=>`<tr><td>${n}</td>`+
  `<td>${f(v.train.cagr)}% | ${f(v.validation.cagr)}%</td><td>${f(v.train.sharpe,2)} | ${f(v.validation.sharpe,2)}</td><td>${f(v.train.mdd)}% | ${f(v.validation.mdd)}%</td>`+
  `<td>${v.train.n??'—'} | ${v.validation.n??'—'}</td><td>${v.train.r!=null?f(v.train.r,2)+' ('+f(v.train.t,1)+')':'—'} | ${v.validation.r!=null?f(v.validation.r,2)+' ('+f(v.validation.t,1)+')':'—'}</td>`+
  `<td>${v.train.costs!=null?f(v.train.costs,0)+'%':'—'} | ${v.validation.costs!=null?f(v.validation.costs,0)+'%':'—'}</td></tr>`).join('');
}
['intraday','daily_only'].forEach(v=>document.getElementById('b-'+v).onclick=()=>{variant=v;
 document.querySelectorAll('.bar button').forEach(b=>b.setAttribute('aria-pressed',b.id==='b-'+v));draw()});
draw();
</script></body></html>"""


def step_chart() -> None:
    summary = json.loads((OUT / "summary.json").read_text())
    md = MarketData(segment="research")
    series, metrics_ = {}, {}
    dates = None
    for name in sg.NAMES:
        for v in ("intraday", "daily_only"):
            e = pd.read_parquet(OUT / f"{name}__{v}_equity.parquet")["equity"]
            dates = e.index if dates is None else dates
            series[f"{name}__{v}"] = (100 * e / summary["capital_usd"]).round(2).tolist()
            r = summary["runs"][f"{name}__{v}"]
            metrics_[f"{name}__{v}"] = {s: {"cagr": r[s]["curve"].get("cagr_pct"), "sharpe": r[s]["curve"].get("sharpe"),
                                            "mdd": r[s]["curve"].get("max_dd_pct"), "n": r[s]["trades"].get("n"),
                                            "r": r[s]["trades"].get("mean_r"), "t": r[s]["trades"].get("t_mean_r"),
                                            "costs": r[s]["trades"].get("costs_pct_of_gross_profit")} for s in SEG}
    spy = md.daily("SPY")["tr_close"].reindex(dates)
    r0 = summary["runs"]["pullback__intraday"]
    metrics_["spy"] = {s: {"cagr": r0[s]["spy"].get("cagr_pct"), "sharpe": r0[s]["spy"].get("sharpe"),
                           "mdd": r0[s]["spy"].get("max_dd_pct")} for s in SEG}
    prev_close = md.daily("SPY")["tr_close"].loc[:dates[0]].iloc[-2]
    reg = pd.read_parquet(ROOT / "regimes" / "market_daily.parquet")["regime"].reindex(dates)
    hv = (reg == "high_vol").astype(int)
    grp = (hv != hv.shift()).cumsum()
    bands = [[str(g.index[0].date()), str(g.index[-1].date())] for _, g in hv.groupby(grp) if g.iloc[0] == 1]
    data = {"dates": [str(d.date()) for d in dates], "series": series,
            "spy": (100 * spy / prev_close).round(2).tolist(), "bands": bands, "metrics": metrics_}
    path = OUT / "strategies_chart.html"
    path.write_text(CHART.replace("__DATA__", json.dumps(data)))
    say(f"chart → {path}")


def step_intraday(only: list[str] | None) -> None:
    """Intraday-only hypotheses H1–H4: one run each at normal costs and one at 2× costs."""
    names = tuple(only or sg.INTRADAY_NAMES)
    md = MarketData(segment="research")
    days, n_slots, idx = bk.session_axis(md.calendar, pd.Timestamp(CFG["data"]["history_start"]).date(),
                                         SEG["validation"][1].date())
    books = bk.build_all(md, CFG, days, idx, names=names)
    ctx = bk.context(CFG, md.calendar, days, n_slots, idx, ROOT)
    start_si = next(i for i, d in enumerate(days) if d >= SEG["train"][0])
    fx = bk.eurusd_at(days[start_si - 1], CFG["execution"]["eurusd_fallback"])
    capital = CFG["execution"]["initial_capital_eur"] * fx
    spy = md.daily("SPY")
    bench = (spy["tr_close"] / spy["tr_close"].loc[days[start_si - 1]]) * capital
    ex = CFG["execution"] | {"earnings_block_sessions": -1}   # flat at the close: no report is held
    cost2 = {k: (v * 2 if k.startswith(("half_spread", "slippage")) else v) for k, v in CFG["costs"].items()}
    out = {"capital_usd": capital, "runs": {}}
    for name in names:
        for variant, c in (("costs_1x", CFG["costs"]), ("costs_2x", cost2)):
            res = engine.run(books, ctx, {name: "entry"}, ex, c, capital, start_si, len(days) - 1)
            tag = f"{name}__{variant}"
            res["trades"].to_parquet(OUT / f"{tag}_trades.parquet")
            res["equity"].to_parquet(OUT / f"{tag}_equity.parquet")
            out["runs"][tag] = {seg: seg_metrics(res, seg, bench) for seg in SEG}
            tr = res["trades"]
            for seg in SEG:
                lo, hi = SEG[seg]
                ts = tr[(tr["entry_date"] >= lo) & (tr["entry_date"] <= hi)] if len(tr) else tr
                out["runs"][tag][seg]["by_market_regime"] = mx.by_group(ts, ["market_regime"])
                m = out["runs"][tag][seg]
                experiments.log(3, "intraday_hypothesis", tag, {"strategy": CFG["strategy"][name], "costs": c},
                                seg, {"curve": m["curve"], "trades": {k: v for k, v in m["trades"].items()
                                                                      if k != "exit_reasons"}})
                cv, tm = m["curve"], m["trades"]
                say(f"{tag:30s} [{seg:10s}] CAGR {cv.get('cagr_pct')}% Sharpe {cv.get('sharpe')} "
                    f"MDD {cv.get('max_dd_pct')}% | n {tm.get('n')} win {tm.get('win_rate')} "
                    f"PF {tm.get('profit_factor')} R {tm.get('mean_r')} (t {tm.get('t_mean_r')}) "
                    f"costs/trade {tm.get('costs_r_per_trade')}R expo {m['avg_exposure_pct']}%")
    (OUT / "summary_intraday.json").write_text(json.dumps(out, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=["run", "jitter", "chart", "intraday"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    for s in a.steps:
        if s == "run":
            step_run(a.only)
        elif s == "jitter":
            step_jitter(a.n, a.only)
        elif s == "chart":
            step_chart()
        elif s == "intraday":
            step_intraday(a.only)


if __name__ == "__main__":
    main()
