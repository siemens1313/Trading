"""Self-contained HTML tearsheet (no external assets): stat tiles, equity, drawdown, exposure,
alpha-weight history, and tables. Charts are inline SVG drawn by a small script with a crosshair."""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

from .backtest import BacktestResult


def _weekly(s: pd.Series | pd.DataFrame):
    return s.resample("W-FRI").last().dropna(how="all")


def write_tearsheet(res: BacktestResult, path: str | Path, bench: str, research: pd.DataFrame | None = None,
                    title: str = "quantbot tearsheet") -> Path:
    m, b = res.metrics, res.benchmark_metrics
    eq = _weekly(pd.concat([res.equity, res.benchmark], axis=1))
    dd = eq / eq.cummax() - 1
    ex = _weekly(res.diag[["gross"]].astype(float))
    aw = _weekly(res.alpha_weights.astype(float))
    data = {
        "dates": [d.strftime("%Y-%m-%d") for d in eq.index],
        "equity": [[round(x, 2) for x in eq.iloc[:, 0]], [round(x, 2) for x in eq.iloc[:, 1]]],
        "dd": [[round(x, 4) for x in dd.iloc[:, 0]], [round(x, 4) for x in dd.iloc[:, 1]]],
        "exp_dates": [d.strftime("%Y-%m-%d") for d in ex.index],
        "gross": [round(x, 4) for x in ex["gross"]],
        "aw_dates": [d.strftime("%Y-%m-%d") for d in aw.index],
        "aw": {k: [round(x, 4) for x in aw[k]] for k in aw.columns},
        "bench": bench,
    }
    tiles = [
        ("CAGR", f"{m['cagr']:+.1%}", f"{bench} {b['cagr']:+.1%}"),
        ("Sharpe", f"{m['sharpe']:.2f}", f"95% CI {m['sharpe_ci'][0]:.2f} to {m['sharpe_ci'][1]:.2f}"),
        ("Max drawdown", f"{m['max_drawdown']:.1%}", f"{bench} {b['max_drawdown']:.1%}"),
        ("Beta to " + bench, f"{m['beta']:.2f}", f"avg gross {m['avg_gross']:.0%}"),
        ("Cost drag", f"{m['cost_drag_per_year']:.2%}/yr", f"turnover {m['turnover_per_year']:.1f}x/yr"),
    ]
    rows = [
        ("Total return", f"{m['total_return']:+.1%}", f"{b['total_return']:+.1%}"),
        ("CAGR", f"{m['cagr']:+.2%}", f"{b['cagr']:+.2%}"),
        ("Volatility", f"{m['ann_vol']:.2%}", f"{b['ann_vol']:.2%}"),
        ("Sharpe (rf=0)", f"{m['sharpe']:.2f}", f"{b['sharpe']:.2f}"),
        ("Sharpe 95% CI", f"{m['sharpe_ci'][0]:.2f} to {m['sharpe_ci'][1]:.2f}", f"{b['sharpe_ci'][0]:.2f} to {b['sharpe_ci'][1]:.2f}"),
        ("P(Sharpe > 0)", f"{m['psr']:.0%}", f"{b['psr']:.0%}"),
        ("Sortino", f"{m['sortino']:.2f}", f"{b['sortino']:.2f}"),
        ("Max drawdown", f"{m['max_drawdown']:.1%}", f"{b['max_drawdown']:.1%}"),
        ("Calmar", f"{m['calmar']:.2f}", f"{b['calmar']:.2f}"),
        ("Worst day", f"{m['worst_day']:.2%}", f"{b['worst_day']:.2%}"),
        ("Trades", f"{m['trades']:,}", ""),
    ]
    e = html.escape
    tile_html = "".join(f'<div class="tile"><div class="tl">{e(a)}</div><div class="tv">{e(v)}</div>'
                        f'<div class="ts">{e(s)}</div></div>' for a, v, s in tiles)
    table_html = "".join(f"<tr><th>{e(a)}</th><td>{e(x)}</td><td>{e(y)}</td></tr>" for a, x, y in rows)
    research_html = ""
    if research is not None and len(research):
        cols = list(research.columns)
        head = "".join(f"<th>{e(str(c))}</th>" for c in ["alpha"] + cols)
        body = "".join("<tr><th>" + e(str(i)) + "</th>" + "".join(f"<td>{e(str(v))}</td>" for v in r) + "</tr>"
                       for i, r in research.iterrows())
        research_html = (f'<section><h2>Alpha research</h2><p class="note">Full-sample, in-sample statistics '
                         f'(descriptive; the live combiner only uses past data).</p>'
                         f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></section>')
    last_w = res.weights.iloc[-1]
    last_w = last_w[last_w.abs() > 1e-4].sort_values(ascending=False)
    alloc = "".join(f"<tr><th>{e(s)}</th><td>{w:.1%}</td></tr>" for s, w in last_w.items())
    alloc += f"<tr><th>Cash</th><td>{1 - last_w.sum():.1%}</td></tr>"
    aw_table = aw.iloc[::13].iloc[::-1]
    aw_rows = "".join("<tr><th>" + d.strftime("%Y-%m-%d") + "</th>" + "".join(f"<td>{v:.0%}</td>" for v in r) + "</tr>"
                      for d, r in aw_table.iterrows())
    aw_head = "".join(f"<th>{e(c)}</th>" for c in aw.columns)

    page = TEMPLATE.replace("__TITLE__", e(title)).replace("__PERIOD__", f"{m['start']} to {m['end']}") \
        .replace("__TILES__", tile_html).replace("__TABLE__", table_html).replace("__BENCH__", e(bench)) \
        .replace("__RESEARCH__", research_html).replace("__ALLOC__", alloc).replace("__ALLOCDATE__", str(res.weights.index[-1].date())) \
        .replace("__AWHEAD__", aw_head).replace("__AWROWS__", aw_rows) \
        .replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page)
    return path


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--bench:#898781}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;
--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px 48px}h1{font-size:22px;margin:0 0 2px}h2{font-size:15px;margin:0 0 10px}
.sub{color:var(--ink2);margin:0 0 20px}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:16px}
.tile,section{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tl{color:var(--ink2);font-size:12px}.tv{font-size:24px;font-weight:600;margin:2px 0}.ts{color:var(--muted);font-size:12px}
section{margin-bottom:14px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:760px){.grid2{grid-template-columns:1fr}}
.grid2 section{margin-bottom:0}.chart{position:relative;width:100%}.chart svg{display:block;width:100%;overflow:visible}
.legend{display:flex;flex-wrap:wrap;gap:14px;color:var(--ink2);font-size:12px;margin:-4px 0 8px}.legend span{display:inline-flex;align-items:center;gap:6px}
.key{width:14px;height:2px;border-radius:1px;display:inline-block}.keyb{width:10px;height:10px;border-radius:2px;display:inline-block}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:8px 10px;
font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.12);display:none;white-space:nowrap;z-index:2}.tip .d{color:var(--muted);margin-bottom:4px}
.tip .r{display:flex;align-items:center;gap:8px}.tip b{font-variant-numeric:tabular-nums}.tip .n{color:var(--ink2)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:5px 8px;border-bottom:1px solid var(--grid)}
th:first-child{text-align:left;font-weight:500;color:var(--ink2)}thead th{color:var(--muted);font-weight:500}.scroll{overflow-x:auto}
.note{color:var(--muted);font-size:12px;margin:-6px 0 10px}details summary{cursor:pointer;color:var(--ink2);margin-top:8px}
.ax{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
</style></head><body><main>
<h1>__TITLE__</h1><p class="sub">Backtest __PERIOD__ · fills at next open after half-spread, slippage and impact · vs __BENCH__ buy and hold</p>
<div class="tiles">__TILES__</div>
<section><h2>Growth of $100k</h2><div class="legend"><span><i class="key" style="background:var(--s1)"></i>Strategy</span>
<span><i class="key" style="background:var(--bench)"></i>__BENCH__ buy and hold</span></div><div class="chart" id="c-eq"></div></section>
<div class="grid2">
<section><h2>Drawdown</h2><div class="legend"><span><i class="key" style="background:var(--s1)"></i>Strategy</span>
<span><i class="key" style="background:var(--bench)"></i>__BENCH__</span></div><div class="chart" id="c-dd"></div></section>
<section><h2>Gross exposure</h2><p class="note">Sum of position weights; the rest is cash</p><div class="chart" id="c-gx"></div></section>
</div>
<section><h2>Alpha weights</h2><p class="note">How much of the forecast each signal drove, set by its realized out-of-sample IC</p>
<div class="legend" id="aw-legend"></div><div class="chart" id="c-aw"></div>
<details><summary>Table view</summary><div class="scroll"><table><thead><tr><th>Week</th>__AWHEAD__</tr></thead><tbody>__AWROWS__</tbody></table></div></details></section>
<div class="grid2">
<section><h2>Performance</h2><table><thead><tr><th></th><th>Strategy</th><th>__BENCH__</th></tr></thead><tbody>__TABLE__</tbody></table></section>
<section><h2>Allocation on __ALLOCDATE__</h2><table><tbody>__ALLOC__</tbody></table></section>
</div>
__RESEARCH__
</main>
<script>
const D=__DATA__;
const NS="http://www.w3.org/2000/svg";
const pct=(v,d=0)=>(v*100).toFixed(d)+"%", usd=v=>"$"+Math.round(v).toLocaleString();
function el(t,a,p){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function ticks(lo,hi,n){const span=hi-lo||1,step0=span/n,mag=Math.pow(10,Math.floor(Math.log10(step0)));
 const step=[1,2,2.5,5,10].map(m=>m*mag).find(s=>span/s<=n)||10*mag;const out=[];for(let v=Math.ceil(lo/step)*step;v<=hi+1e-12;v+=step)out.push(+v.toFixed(10));return out}
function chart(id,{dates,series,fmt,stack=false,height=240,zero=false,area=false}){
 const box=document.getElementById(id);box.innerHTML="";const W=box.clientWidth||600,H=height,m={l:52,r:12,t:8,b:24};
 const iw=W-m.l-m.r,ih=H-m.t-m.b,n=dates.length;let cum=null;
 const ser=series.map(s=>({...s}));
 if(stack){cum=new Array(n).fill(0);ser.forEach(s=>{s.lo=cum.slice();s.hi=s.values.map((v,i)=>cum[i]+=v)})}
 let lo=Infinity,hi=-Infinity;ser.forEach(s=>(stack?s.hi:s.values).forEach(v=>{if(v<lo)lo=v;if(v>hi)hi=v}));
 if(stack||zero){lo=Math.min(0,lo);hi=Math.max(0,hi)}const pad=(hi-lo)*0.05;if(!stack&&!zero)lo-=pad;hi+=pad;if(stack)hi=Math.max(hi,1);
 const X=i=>m.l+(n<2?0:i/(n-1)*iw),Y=v=>m.t+ih-(v-lo)/(hi-lo)*ih;
 const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,height:H,role:"img"},box);
 ticks(lo,hi,4).forEach(v=>{el("line",{x1:m.l,x2:W-m.r,y1:Y(v),y2:Y(v),stroke:"var(--grid)","stroke-width":1},svg);
  el("text",{x:m.l-8,y:Y(v)+4,"text-anchor":"end",class:"ax"},svg).textContent=fmt(v)});
 let lastX=-1e9;const years=[...new Set(dates.map(d=>d.slice(0,4)))];years.forEach(y=>{const i=dates.findIndex(d=>d.startsWith(y));
  if(i<0||X(i)>W-m.r-20||X(i)-lastX<40)return;lastX=X(i);
  el("text",{x:X(i),y:H-6,"text-anchor":"middle",class:"ax"},svg).textContent=y});
 el("line",{x1:m.l,x2:W-m.r,y1:Y(Math.max(lo,0)),y2:Y(Math.max(lo,0)),stroke:"var(--axis)","stroke-width":1},svg);
 const path=(vals)=>vals.map((v,i)=>(i?"L":"M")+X(i).toFixed(1)+","+Y(v).toFixed(1)).join("");
 ser.forEach(s=>{if(stack){const top=s.hi.map((v,i)=>X(i).toFixed(1)+","+Y(v).toFixed(1)),bot=s.lo.map((v,i)=>X(i).toFixed(1)+","+Y(v).toFixed(1)).reverse();
   el("path",{d:"M"+top.join("L")+"L"+bot.join("L")+"Z",fill:s.color,stroke:"var(--surface)","stroke-width":1},svg)}
  else{if(area||s.area)el("path",{d:path(s.values)+`L${X(n-1)},${Y(Math.max(lo,0))}L${X(0)},${Y(Math.max(lo,0))}Z`,fill:s.color,"fill-opacity":.1},svg);
   el("path",{d:path(s.values),fill:"none",stroke:s.color,"stroke-width":2,"stroke-linejoin":"round","stroke-linecap":"round"},svg)}});
 const cross=el("line",{y1:m.t,y2:m.t+ih,stroke:"var(--axis)","stroke-width":1,visibility:"hidden"},svg);
 const dots=ser.map(s=>stack?null:el("circle",{r:4,fill:s.color,stroke:"var(--surface)","stroke-width":2,visibility:"hidden"},svg));
 const tip=document.createElement("div");tip.className="tip";box.appendChild(tip);
 const hit=el("rect",{x:m.l,y:m.t,width:iw,height:ih,fill:"transparent",tabindex:0},svg);
 function show(i){const x=X(i);cross.setAttribute("x1",x);cross.setAttribute("x2",x);cross.setAttribute("visibility","visible");
  tip.replaceChildren();const d=document.createElement("div");d.className="d";d.textContent=dates[i];tip.appendChild(d);
  ser.slice().reverse().forEach((s,k)=>{const r=document.createElement("div");r.className="r";const key=document.createElement("i");key.className="key";
   key.style.background=s.color;const v=document.createElement("b");v.textContent=fmt(s.values[i],1);const nm=document.createElement("span");
   nm.className="n";nm.textContent=s.name;r.append(key,v,nm);tip.appendChild(r)});
  ser.forEach((s,k)=>{if(dots[k]){dots[k].setAttribute("cx",x);dots[k].setAttribute("cy",Y(s.values[i]));dots[k].setAttribute("visibility","visible")}});
  tip.style.display="block";const tw=tip.offsetWidth,px=x/W*box.clientWidth;tip.style.left=Math.min(Math.max(px+12,0),box.clientWidth-tw)+"px";
  tip.style.top=(px+12+tw>box.clientWidth?8:8)+"px";if(px+12+tw>box.clientWidth)tip.style.left=Math.max(px-tw-12,0)+"px"}
 function hide(){cross.setAttribute("visibility","hidden");dots.forEach(d=>d&&d.setAttribute("visibility","hidden"));tip.style.display="none"}
 let cur=n-1;
 hit.addEventListener("pointermove",ev=>{const r=svg.getBoundingClientRect(),x=(ev.clientX-r.left)/r.width*W;cur=Math.round(Math.min(Math.max((x-m.l)/iw,0),1)*(n-1));show(cur)});
 hit.addEventListener("pointerleave",hide);hit.addEventListener("blur",hide);hit.addEventListener("focus",()=>show(cur));
 hit.addEventListener("keydown",ev=>{if(ev.key==="ArrowLeft")cur=Math.max(0,cur-1);else if(ev.key==="ArrowRight")cur=Math.min(n-1,cur+1);else return;ev.preventDefault();show(cur)});
}
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const AWC=["--s1","--s2","--s3","--s4"];
function render(){
 chart("c-eq",{dates:D.dates,fmt:(v)=>usd(v),height:280,series:[{name:D.bench,values:D.equity[1],color:css("--bench")},{name:"Strategy",values:D.equity[0],color:css("--s1")}]});
 chart("c-dd",{dates:D.dates,fmt:(v,d)=>pct(v,d||0),zero:true,series:[{name:D.bench,values:D.dd[1],color:css("--bench")},{name:"Strategy",values:D.dd[0],color:css("--s1"),area:true}]});
 chart("c-gx",{dates:D.exp_dates,fmt:(v,d)=>pct(v,d||0),zero:true,series:[{name:"Gross",values:D.gross,color:css("--s1"),area:true}]});
 const names=Object.keys(D.aw);chart("c-aw",{dates:D.aw_dates,fmt:(v,d)=>pct(v,d||0),stack:true,height:200,
  series:names.map((k,i)=>({name:k,values:D.aw[k],color:css(AWC[i%4])}))});
 const lg=document.getElementById("aw-legend");lg.replaceChildren();names.forEach((k,i)=>{const s=document.createElement("span");
  const b=document.createElement("i");b.className="keyb";b.style.background=css(AWC[i%4]);s.append(b,document.createTextNode(k));lg.appendChild(s)});
}
render();let rt;addEventListener("resize",()=>{clearTimeout(rt);rt=setTimeout(render,120)});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",render);
new MutationObserver(render).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
</script></body></html>
"""
