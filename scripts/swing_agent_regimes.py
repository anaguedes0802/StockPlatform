"""Swing agent — phase 2: regimes (daily market, daily per stock, intraday micro).

Usage (from the repo root; needs phase 1's data):
    apps/api/.venv/bin/python scripts/swing_agent_regimes.py daily        # breadth + market + stock regimes, baseline eval
    apps/api/.venv/bin/python scripts/swing_agent_regimes.py sensitivity  # one-at-a-time parameter variants
    apps/api/.venv/bin/python scripts/swing_agent_regimes.py hmm          # Markov-switching comparison
    apps/api/.venv/bin/python scripts/swing_agent_regimes.py intraday     # micro regimes + eval
    apps/api/.venv/bin/python scripts/swing_agent_regimes.py charts report
    apps/api/.venv/bin/python scripts/swing_agent_regimes.py all

Research window only (train + validation). Every evaluated variant is
appended to apps/api/artifacts/swing_agent/experiments.jsonl.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import date

os.environ.setdefault("WARMUP_DISABLED", "1")
sys.path.insert(0, "apps/api")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.swing_agent import config, experiments, regime_daily as rd, regime_eval as ev  # noqa: E402
from app.swing_agent import regime_intraday as ri  # noqa: E402
from app.swing_agent.data import MarketData  # noqa: E402
from app.swing_agent.universe import Membership  # noqa: E402

CFG = config.load()
ROOT = config.root()
OUT = ROOT / "regimes"
OUT.mkdir(parents=True, exist_ok=True)
SEG = {"train": config.segment("train"), "validation": config.segment("validation")}


def say(*a: object) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))


def md() -> MarketData:
    return MarketData(segment="research")


def clip(s: pd.Series | pd.DataFrame, seg: str):
    lo, hi = SEG[seg]
    return s[(s.index >= lo) & (s.index <= hi)]


# ---------------------------------------------------------------------------
# daily
# ---------------------------------------------------------------------------

def load_breadth(m: MarketData) -> pd.DataFrame:
    p = OUT / "breadth.parquet"
    if p.exists():
        return pd.read_parquet(p)
    mem = Membership.load(ROOT / "membership" / "sp500_hist.csv")
    ivs = mem.intervals(date.fromisoformat(CFG["data"]["history_start"]))
    daily = {}
    for iv in ivs:
        try:
            df = m.daily(iv.key)
        except KeyError:
            continue
        if not df.empty:
            daily[iv.key] = df
    sessions = rd.date_range_index(m.calendar, date.fromisoformat(CFG["data"]["history_start"]),
                                   SEG["validation"][1].date())
    b = rd.breadth(ivs, daily, sessions, CFG["regime"]["daily"]["breadth_ma"])
    b.to_parquet(p)
    return b


def market(m: MarketData, p: dict) -> pd.DataFrame:
    bench = m.daily(p["benchmark"])
    vix = m.daily("^VIX")
    return rd.market_regime(bench, vix, load_breadth(m), m.calendar, p)


def daily_metrics(reg: pd.DataFrame, bench: pd.DataFrame, seg: str) -> dict:
    h = CFG["regime"]["eval"]["forward_days"]
    r = clip(reg["regime"], seg)
    fd = ev.forward_daily(clip(bench, seg), r, h)
    px = clip(bench["close"] * bench["tr_factor"], seg)
    eps = ev.drawdowns(px, CFG["regime"]["eval"]["drawdown_min"])
    det = ev.detection(eps, r, px)
    lags = [d["lag_sessions"] for d in det if "lag_sessions" in d]
    return {"shape": ev.shape(r), "forward": ev.forward_table(fd, h),
            "vol_separation": ev.vol_separation(fd), "drawdowns": det,
            "mean_lag_sessions": round(float(np.mean(lags)), 1) if lags else None}


def step_daily() -> None:
    m = md()
    p = CFG["regime"]["daily"]
    reg = market(m, p)
    reg.to_parquet(OUT / "market_daily.parquet")
    bench = m.daily(p["benchmark"])
    res = {seg: daily_metrics(reg, bench, seg) for seg in SEG}
    jdump(OUT / "market_daily_eval.json", res)
    for seg, r in res.items():
        experiments.log(2, "market_regime_rules", "baseline", p, seg,
                        {k: r[k] for k in ("vol_separation", "mean_lag_sessions")}
                        | {"switches_per_year": r["shape"]["switches_per_year"],
                           "share": r["shape"]["share"]})
        say(f"market regime [{seg}]: share {r['shape']['share']}, "
            f"{r['shape']['switches_per_year']} switches/yr, vol sep {r['vol_separation']}, "
            f"lag {r['mean_lag_sessions']}")
    # per-stock daily regimes for the universe + reference ETFs
    pit = pd.read_parquet(ROOT / "universe_pit.parquet")
    keys = sorted(set(pit["key"]) | set(CFG["universe"]["references"]) | {"IWM"})
    sp = CFG["regime"]["stock"]
    (OUT / "stock_daily").mkdir(parents=True, exist_ok=True)
    shares = []
    for k in keys:
        df = m.daily(k)
        if df.empty:
            continue
        sr = rd.stock_regime(df, sp)
        sr.to_parquet(OUT / "stock_daily" / f"{k}.parquet")
        months = set(pit.loc[pit["key"] == k, "rebalance"].dt.to_period("M"))
        inu = sr[sr.index.to_period("M").isin(months)] if months else sr
        shares.append(inu["regime"].dropna())
    allr = pd.concat(shares)
    st_shape = {"share_in_universe_months": allr.value_counts(normalize=True).round(4).to_dict(),
                "switches_per_stock_year": round(float(np.mean([
                    ((s != s.shift()).sum() - 1) / (len(s) / 252) for s in shares if len(s) > 60])), 2)}
    jdump(OUT / "stock_daily_eval.json", st_shape)
    experiments.log(2, "stock_regime_rules", "baseline", sp, "research", st_shape)
    say(f"stock regimes: {len(shares)} keys, {st_shape}")


# ---------------------------------------------------------------------------
# sensitivity
# ---------------------------------------------------------------------------

VARIANTS = [
    ("v1_priority", {"structure": "priority"}),
    ("vix_22_18", {"vix_high_enter": 22.0, "vix_high_exit": 18.0}),
    ("vix_28_23", {"vix_high_enter": 28.0, "vix_high_exit": 23.0}),
    ("adx_15", {"adx_trend": 15}),
    ("adx_25", {"adx_trend": 25}),
    ("slope_5d", {"slope_fast_days": 5}),
    ("slope_20d", {"slope_fast_days": 20}),
    ("confirm_1", {"confirm_days": 1}),
    ("confirm_3", {"confirm_days": 3}),
    ("no_breadth", {"breadth_bull_min": 0.0}),
    ("breadth_60", {"breadth_bull_min": 0.60}),
    ("sma_40_150", {"sma_fast": 40, "sma_slow": 150}),
    ("sma_60_250", {"sma_fast": 60, "sma_slow": 250}),
    ("atr_rank_95", {"atr_rank_high": 0.95}),
]


def step_sensitivity() -> None:
    m = md()
    base_p = CFG["regime"]["daily"]
    base = pd.read_parquet(OUT / "market_daily.parquet")["regime"]
    bench = m.daily(base_p["benchmark"])
    rows = []
    for name, over in VARIANTS:
        p = copy.deepcopy(base_p) | over
        reg = market(m, p)
        row = {"variant": name, **over}
        for seg in SEG:
            r = daily_metrics(reg, bench, seg)
            row[f"{seg}_agreement"] = ev.agreement(clip(base, seg), clip(reg["regime"], seg))
            row[f"{seg}_switches_yr"] = r["shape"]["switches_per_year"]
            row[f"{seg}_vol_sep"] = r["vol_separation"]
            row[f"{seg}_lag"] = r["mean_lag_sessions"]
            row[f"{seg}_bull_share"] = r["shape"]["share"].get("bull")
            experiments.log(2, "market_regime_rules", name, p, seg,
                            {"agreement": row[f"{seg}_agreement"], "vol_separation": r["vol_separation"],
                             "mean_lag_sessions": r["mean_lag_sessions"],
                             "switches_per_year": r["shape"]["switches_per_year"]})
        rows.append(row)
        say(f"sensitivity {name}: agree train {row['train_agreement']}, val {row['validation_agreement']}")
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "sensitivity.csv", index=False)
    jdump(OUT / "sensitivity.json", t.to_dict("records"))


# ---------------------------------------------------------------------------
# HMM comparison
# ---------------------------------------------------------------------------

def step_hmm() -> None:
    from app.swing_agent import regime_hmm

    m = md()
    p = CFG["regime"]["daily"]
    bench = m.daily(p["benchmark"])
    rules = pd.read_parquet(OUT / "market_daily.parquet")["regime"]
    ret = (bench["close"] * bench["tr_factor"]).pct_change()
    out = {"rules": {seg: daily_metrics(pd.DataFrame({"regime": rules}), bench, seg) for seg in SEG}}
    for k in (2, 3):
        h = regime_hmm.fit_filter(ret, SEG["train"][1], k=k, seed=CFG["seed"])
        h["regime"].to_frame().assign(p_risk_off=h["p_risk_off"]).to_parquet(OUT / f"hmm{k}.parquet")
        out[f"hmm{k}"] = {"params": h["params"]}
        for seg in SEG:
            r = daily_metrics(pd.DataFrame({"regime": h["regime"]}), bench, seg)
            out[f"hmm{k}"][seg] = r
            experiments.log(2, "market_regime_hmm", f"markov_switching_k{k}", {"k": k, "fit_end": str(SEG["train"][1].date())},
                            seg, {"vol_separation": r["vol_separation"], "mean_lag_sessions": r["mean_lag_sessions"],
                                  "switches_per_year": r["shape"]["switches_per_year"]})
    # pre-declared rule (SWING_AGENT.md § 2.0), on validation
    rv = out["rules"]["validation"]
    verdict = {}
    for k in (2, 3):
        hv = out[f"hmm{k}"]["validation"]
        c1 = (hv["vol_separation"] or 0) > (rv["vol_separation"] or 0)
        c2 = hv["mean_lag_sessions"] is not None and rv["mean_lag_sessions"] is not None and \
            hv["mean_lag_sessions"] < rv["mean_lag_sessions"]
        c3 = hv["shape"]["switches_per_year"] <= 1.5 * rv["shape"]["switches_per_year"]
        verdict[f"hmm{k}"] = {"vol_sep_better": c1, "lag_shorter": c2, "switches_ok": c3,
                              "adopt": bool(c1 and c2 and c3)}
    out["verdict"] = verdict
    jdump(OUT / "hmm_eval.json", out)
    for k in (2, 3):
        hv = out[f"hmm{k}"]["validation"]
        say(f"hmm{k} validation: vol sep {hv['vol_separation']} (rules {rv['vol_separation']}), "
            f"lag {hv['mean_lag_sessions']} (rules {rv['mean_lag_sessions']}), switches/yr "
            f"{hv['shape']['switches_per_year']} (rules {rv['shape']['switches_per_year']}) → {verdict[f'hmm{k}']}")


# ---------------------------------------------------------------------------
# intraday
# ---------------------------------------------------------------------------

def step_intraday() -> None:
    m = md()
    p = CFG["regime"]["intraday"]
    pit = pd.read_parquet(ROOT / "universe_pit.parquet")
    keys = sorted(set(pit["key"]) | set(CFG["universe"]["references"]))
    (OUT / "intraday").mkdir(parents=True, exist_ok=True)
    fw = []
    t0 = time.time()
    for i, k in enumerate(keys, 1):
        m15 = m.m15(k)
        if m15.empty:
            continue
        mr = ri.micro_regime(m15, m.daily(k), p)
        mr[["session", "slot", "t_close", "close", "vwap_s", "dist", "gap", "relvol", "range_ratio",
            "side_share", "atr_prev", "micro", "vol_state"]].to_parquet(OUT / "intraday" / f"{k}.parquet")
        last = m15.groupby("session")["close"].last() * m15.groupby("session")["tr_factor"].last()
        f = ev.forward_intraday(mr, last)
        if k in CFG["universe"]["references"]:
            f["group"] = k
        else:
            months = set(pit.loc[pit["key"] == k, "rebalance"].dt.to_period("M"))
            f = f[pd.DatetimeIndex(f["session"]).to_period("M").isin(months)]
            f["group"] = "stocks"
        fw.append(f.assign(key=k))
        if i % 20 == 0:
            say(f"intraday: {i}/{len(keys)} ({time.time() - t0:.0f}s)")
    allf = pd.concat(fw, ignore_index=True)
    stocks_rel = ev.add_relative(allf[allf["group"] == "stocks"])
    refs = allf[allf["group"] != "stocks"].assign(fwd_rel_atr=np.nan)
    refs["signed_fwd_atr"] = refs["fwd_atr"] * refs["dir"]
    refs["signed_rel_atr"] = np.nan
    allf = pd.concat([stocks_rel, refs], ignore_index=True)
    allf.to_parquet(OUT / "intraday_forward.parquet")
    res = {}
    for seg in SEG:
        lo, hi = SEG[seg]
        x = allf[(allf["session"] >= lo) & (allf["session"] <= hi)]
        res[seg] = {}
        for grp, g in x.groupby("group"):
            res[seg][grp] = {
                "by_label": ev.clustered(g, "signed_fwd_atr", "micro"),
                "by_label_vs_market": ev.clustered(g, "signed_rel_atr", "micro") if grp == "stocks" else None,
                # |move| to the close vs the average |move| at the same hour (the
                # time left in the session would otherwise drive the comparison)
                "by_vol_state_abs_move_vs_same_hour": ev.clustered(
                    g.assign(abs_rel=g["fwd_atr"].abs() / g.groupby("slot")["fwd_atr"]
                             .transform(lambda v: v.abs().mean())), "abs_rel", "vol_state"),
            }
        stocks = x[x["group"] == "stocks"]
        res[seg]["stocks_shape"] = {
            "share": stocks["micro"].value_counts(normalize=True).round(4).to_dict(),
            "hourly_label_changes_per_session": round(float(
                stocks.groupby(["key", "session"])["micro"].apply(lambda s: (s != s.shift()).sum() - 1).mean()), 2),
        }
        experiments.log(2, "micro_regime_rules", "baseline", p, seg,
                        {"stocks_vs_market": {lab: (v["mean_per_day"], v["t_clustered"]) for lab, v in
                                              res[seg]["stocks"]["by_label_vs_market"].items()}})
    jdump(OUT / "intraday_eval.json", res)
    for seg in SEG:
        for which in ("by_label", "by_label_vs_market"):
            say(f"intraday [{seg}] stocks {which}:", {k: (v["mean_per_day"], v["t_clustered"], v["share"])
                                                    for k, v in res[seg]["stocks"][which].items()})


# ---------------------------------------------------------------------------
# chart
# ---------------------------------------------------------------------------

CHART_HTML = """<!doctype html>
<html lang="pt"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Regimes de mercado</title>
<style>
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--page:#f9f9f7;--text-primary:#0b0b0b;--text-secondary:#52514e;
 --muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
 --bull:#2a78d6;--high_vol:#eb6834;--range:#1baf7a;--bear:#eda100;--line:#0b0b0b;--band-op:.30}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface-1:#1a1a19;
 --page:#0d0d0d;--text-primary:#fff;--text-secondary:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
 --bull:#3987e5;--high_vol:#d95926;--range:#199e70;--bear:#c98500;--line:#fff;--band-op:.36}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;--text-primary:#fff;
 --text-secondary:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
 --bull:#3987e5;--high_vol:#d95926;--range:#199e70;--bear:#c98500;--line:#fff;--band-op:.36}
body{margin:0;background:var(--page)}
.viz-root{font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--text-primary);background:var(--page);
 padding:16px;max-width:1100px;margin:0 auto}
.card{background:var(--surface-1);border:1px solid var(--ring);border-radius:12px;padding:16px}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--text-secondary);margin:0 0 12px}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:8px 0 4px;color:var(--text-secondary)}
.chip{display:inline-flex;align-items:center;gap:6px}.sw{width:12px;height:12px;border-radius:3px}
svg{display:block;width:100%;height:auto;overflow:visible}
.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}.gl{stroke:var(--grid);stroke-width:1}
.ax{stroke:var(--axis)}.lbl{fill:var(--text-secondary);font-size:12px}
.tip{position:fixed;pointer-events:none;background:var(--surface-1);border:1px solid var(--ring);border-radius:8px;
 padding:8px 10px;font-size:12px;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;min-width:170px}
.tip b{font-weight:600}.tip .r{display:flex;justify-content:space-between;gap:12px;color:var(--text-secondary)}
.tip .r span:last-child{color:var(--text-primary);font-variant-numeric:tabular-nums}
details{margin-top:12px;color:var(--text-secondary)}table{border-collapse:collapse;font-variant-numeric:tabular-nums;margin-top:8px}
td,th{padding:4px 10px;text-align:right;border-bottom:1px solid var(--grid)}th:first-child,td:first-child{text-align:left}
</style></head><body><div class="viz-root"><div class="card">
<h1>Regime de mercado diário — S&amp;P 500 (SPY), 2016–2023</h1>
<p class="sub">Calculado no fecho de cada dia e usado a partir do dia seguinte. 2016 é aquecimento; treino 2017–2021; validação 2022–2023. O teste final (2024+) não foi lido.</p>
<div class="legend" id="legend"></div>
<svg id="chart" viewBox="0 0 1000 600" role="img" aria-label="Preço do SPY com regimes coloridos, VIX e amplitude"></svg>
<details><summary>Tabela: % do tempo em cada regime por ano</summary><table id="tbl"></table></details>
</div><div class="tip" id="tip"></div></div>
<script>
const D = __DATA__;
const LAB = {bull:"Alta", range:"Lateral", bear:"Baixa", high_vol:"Alta volatilidade"};
const ORDER = ["bull","high_vol","range","bear"];
const css = n => getComputedStyle(document.querySelector('.viz-root')).getPropertyValue(n).trim();
const svg = document.getElementById('chart'), NS = 'http://www.w3.org/2000/svg';
const W = 1000, L = 48, R = 104, P = [{y:44,h:280,key:'spy'},{y:356,h:96,key:'vix'},{y:484,h:86,key:'breadth'}];
const t = D.map(d => new Date(d.d+'T00:00:00Z').getTime()), t0 = t[0], t1 = t[t.length-1];
const X = v => L + (v - t0) / (t1 - t0) * (W - L - R);
function el(n, a, p){const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);(p||svg).appendChild(e);return e}
function scale(vals, lo, hi, log){const f=log?Math.log:(v=>v);const a=f(lo),b=f(hi);return (p,v)=>p.y+p.h-(f(v)-a)/(b-a)*p.h}
// legend with share of time
const share = {}; D.forEach(d=>{ if(d.r) share[d.r]=(share[d.r]||0)+1 });
const n = Object.values(share).reduce((a,b)=>a+b,0);
document.getElementById('legend').innerHTML = ORDER.map(k=>`<span class="chip"><span class="sw" style="background:var(--${k})"></span>${LAB[k]} · ${(100*(share[k]||0)/n).toFixed(0)}%</span>`).join('')
 + '<span class="chip"><span class="sw" style="background:var(--line);height:2px"></span>SPY</span>';
// regime bands across all panels
let i0 = 0;
for (let i=1;i<=D.length;i++){ if(i===D.length || D[i].r!==D[i0].r){ const r=D[i0].r; if(r){
  const x0=X(t[i0]), x1=X(t[Math.min(i,D.length-1)]);
  P.forEach(p=>el('rect',{x:x0,y:p.y,width:Math.max(0.5,x1-x0),height:p.h,fill:`var(--${r})`,'fill-opacity':'var(--band-op)'}).style.fillOpacity=css('--band-op'));
 } i0=i; } }
// panels
function panel(p, key, lo, hi, log, ticks, fmt, refs){
  const y = scale(null, lo, hi, log);
  ticks.forEach(v=>{el('line',{x1:L,x2:W-R,y1:y(p,v),y2:y(p,v),class:'gl'});const tx=el('text',{x:L-6,y:y(p,v)+4,'text-anchor':'end',class:'tick'});tx.textContent=fmt(v)});
  (refs||[]).forEach(([v,txt],k)=>{el('line',{x1:L,x2:W-R,y1:y(p,v),y2:y(p,v),stroke:css('--text-secondary'),'stroke-dasharray':'4 4','stroke-width':1});
    const tx=el('text',{x:W-R+6,y:y(p,v)+(k?10:-2),class:'tick'});tx.textContent=txt});
  let dpath=''; D.forEach((d,i)=>{ if(d[key]==null) return; dpath += (dpath?'L':'M')+X(t[i]).toFixed(1)+','+y(p,d[key]).toFixed(1) });
  el('path',{d:dpath,fill:'none',stroke:css('--line'),'stroke-width':key==='spy'?2:1.5,'stroke-linejoin':'round'});
  return y;
}
const spyv = D.map(d=>d.spy).filter(v=>v), vixv = D.map(d=>d.vix).filter(v=>v);
const ys = panel(P[0],'spy',Math.min(...spyv)*0.97,Math.max(...spyv)*1.03,true,[200,250,300,350,400,450],v=>v);
const yv = panel(P[1],'vix',8,Math.min(85,Math.max(...vixv)+2),false,[10,20,40,60,80],v=>v,[[25,'25: entra alta vol.'],[20,'20: sai']]);
const yb = panel(P[2],'breadth',0,100,false,[0,50,100],v=>v+'%',[[50,'50%: mínimo p/ alta']]);
[['SPY, ajustado a dividendos (escala log)',P[0]],['VIX',P[1]],['Amplitude: % do S&P 500 acima da média de 50 dias',P[2]]].forEach(([s,p])=>{const tx=el('text',{x:L,y:p.y-8,class:'lbl'});tx.textContent=s});
// x axis years + segment markers
for(let yr=2016; yr<=2024; yr++){const v=Date.UTC(yr,0,1); if(v<t0||v>t1) continue; const x=X(v);
 el('line',{x1:x,x2:x,y1:P[2].y+P[2].h,y2:P[2].y+P[2].h+5,class:'ax'}); const tx=el('text',{x:x,y:P[2].y+P[2].h+18,'text-anchor':'middle',class:'tick'}); tx.textContent=yr}
[[t0,'aquecimento'],[Date.UTC(2017,0,1),'treino'],[Date.UTC(2022,0,1),'validação']].forEach(([v,s],k)=>{const x=X(v);
 if(k) el('line',{x1:x,x2:x,y1:16,y2:P[2].y+P[2].h,stroke:css('--text-primary'),'stroke-width':1.5});
 const tx=el('text',{x:x+(k?6:0),y:14,class:'lbl'}); tx.textContent=s+(k?' →':'')});
// hover
const cross = el('line',{x1:0,x2:0,y1:P[0].y,y2:P[2].y+P[2].h,stroke:css('--text-secondary'),'stroke-width':1,visibility:'hidden'});
const dot = el('circle',{r:4,fill:css('--surface-1'),stroke:css('--line'),'stroke-width':2,visibility:'hidden'});
const tip = document.getElementById('tip');
const hit = el('rect',{x:L,y:P[0].y,width:W-L-R,height:P[2].y+P[2].h-P[0].y,fill:'transparent'});
hit.addEventListener('mousemove',ev=>{ const b=svg.getBoundingClientRect(); const vx=(ev.clientX-b.left)/b.width*W;
 const tv=t0+(vx-L)/(W-L-R)*(t1-t0); let lo=0,hi=t.length-1; while(hi-lo>1){const m=(lo+hi)>>1; if(t[m]<tv) lo=m; else hi=m}
 const i = (tv-t[lo] < t[hi]-tv) ? lo : hi, d=D[i], x=X(t[i]);
 cross.setAttribute('x1',x);cross.setAttribute('x2',x);cross.setAttribute('visibility','visible');
 if(d.spy){dot.setAttribute('cx',x);dot.setAttribute('cy',ys(P[0],d.spy));dot.setAttribute('visibility','visible')}
 tip.innerHTML=`<b>${d.d}</b><div class="r"><span>Regime</span><span>${d.r?LAB[d.r]:'—'}</span></div>`+
  `<div class="r"><span>Tendência</span><span>${d.tr?LAB[d.tr]:'—'}</span></div><div class="r"><span>Volatilidade</span><span>${d.vs==='high'?'alta':'normal'}</span></div>`+
  `<div class="r"><span>SPY</span><span>${d.spy?d.spy.toFixed(2):'—'}</span></div><div class="r"><span>VIX</span><span>${d.vix?d.vix.toFixed(1):'—'}</span></div>`+
  `<div class="r"><span>Amplitude</span><span>${d.breadth!=null?d.breadth.toFixed(0)+'%':'—'}</span></div>`;
 tip.style.display='block'; const tx=ev.clientX+14+180>innerWidth?ev.clientX-194:ev.clientX+14; tip.style.left=tx+'px'; tip.style.top=(ev.clientY+12)+'px'});
hit.addEventListener('mouseleave',()=>{tip.style.display='none';cross.setAttribute('visibility','hidden');dot.setAttribute('visibility','hidden')});
// table
const yrs = {}; D.forEach(d=>{ if(!d.r) return; const y=d.d.slice(0,4); (yrs[y]=yrs[y]||{n:0}); yrs[y][d.r]=(yrs[y][d.r]||0)+1; yrs[y].n++ });
document.getElementById('tbl').innerHTML = '<tr><th>Ano</th>'+ORDER.map(k=>`<th>${LAB[k]}</th>`).join('')+'</tr>'+
 Object.entries(yrs).map(([y,v])=>`<tr><td>${y}</td>`+ORDER.map(k=>`<td>${(100*(v[k]||0)/v.n).toFixed(0)}%</td>`).join('')+'</tr>').join('');
</script></body></html>"""


def step_charts() -> None:
    r = pd.read_parquet(OUT / "market_daily.parquet")
    r = r[r.index <= SEG["validation"][1]]
    rows = [{"d": str(i.date()), "spy": None if pd.isna(x.close) else round(float(x.close), 2),
             "vix": None if pd.isna(x.vix) else round(float(x.vix), 2),
             "breadth": None if pd.isna(x.breadth) else round(100 * float(x.breadth), 1),
             "r": x.regime, "tr": x.trend, "vs": x.vol_state} for i, x in r.iterrows()]
    path = OUT / "regime_chart.html"
    path.write_text(CHART_HTML.replace("__DATA__", json.dumps(rows)))
    say(f"chart → {path}")


STEPS = {"daily": step_daily, "sensitivity": step_sensitivity, "hmm": step_hmm,
         "intraday": step_intraday, "charts": step_charts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=[*STEPS, "all"])
    a = ap.parse_args()
    for s in (list(STEPS) if a.steps == ["all"] else a.steps):
        say(f"== {s}")
        STEPS[s]()


if __name__ == "__main__":
    main()
