
let CUR_KEY=null;
function rankList(model,noneProb,k){
  return META.hazard_classes.map(c=>({c,...model[c],active:model[c].prob>k*noneProb,
    ratio:noneProb>0?model[c].prob/noneProb:0})).sort((a,b)=>b.prob-a.prob);
}
// 1h-KNN V1 distribution from the most-local k (e.g. k15)
function knnV1(rec){
  if(!rec.knn) return null;
  const kk='k60'
  return {k:kk, v1:rec.knn[kk].v1_distance_weighted||{}, conf:rec.knn[kk].v1_confidence};
}
// Blend the model's none-relative ratios with the 1h-KNN's none-relative ratios.
// Both are "x none" so the 1:1 average is meaningful; threshold k still applies.
// Also computes blo = the confidence-discounted LOWER BOUND the lights threshold on:
//   modelConf = 1 - clamp((prob - lo)/prob, 0, 1)   (proportional MC-dropout confidence)
//   knnConf   = v1_confidence
//   blo       = br * (modelConf * knnConf)           (conjunctive: both must be sure)
function blended(model,noneProb,knn,W){
  const kv=knn?knn.v1:null, kn=kv?(kv[META.none_label]||0):0;
  const knnConf=knn?(knn.conf||0):null;
  const out=META.hazard_classes.map(c=>{
    const rr=noneProb>0?model[c].prob/noneProb:0;                 // model x none
    let kr=0;
    if(kv) kr=Math.min((kv[c]||0)/Math.max(kn,0.02),5);           // knn x none (capped)
    const w=kv?W:0;                                               // no knn -> model only
    const br=(1-w)*rr+w*kr;
    // confidence-discounted lower bound
    const p=model[c].prob, lo=model[c].lo;
    const modelConf=(p>0&&lo!=null)?(1-clamp((p-lo)/p,0,1)):0.5;
    const jointConf=(knnConf!=null)?(modelConf*knnConf):modelConf;
    const blo=br*jointConf;
    return {c,prob:model[c].prob,lo:model[c].lo,hi:model[c].hi,rr,kr,br,
            modelConf,knnConf,jointConf,blo};
  }).sort((a,b)=>b.br-a.br);
  return out;
}
function bestRatio(rows){ return rows.length?Math.max(...rows.map(r=>r.br)):0; }

const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function brightnessFor(ratio,k){ return clamp((ratio-k)/(1.0-k),0,1); }

// progress of one class toward its thresholds, measured on the MEAN (br) so the gauge
// core sweeps one continuous quantity. Three equal segments anchored at the ladder:
//   0 .. watch    -> 0.00 .. 0.33   (blue->green)
//   watch .. adv  -> 0.33 .. 0.66   (green->yellow)
//   adv .. warn   -> 0.66 .. 1.00   (yellow->red)
//   past warn     -> >1 (a touch, for latched red)
function classProgress(br,t){
  if(!t||br<=0) return 0;
  const w=t.watch, a=(t.advisory!=null?t.advisory:(t.watch+t.warn)/2), r=t.warn;
  if(br<w)  return (1/3)*(br/Math.max(w,1e-6));
  if(br<a)  return 1/3+(1/3)*((br-w)/Math.max(a-w,1e-6));
  if(br<r)  return 2/3+(1/3)*((br-a)/Math.max(r-a,1e-6));
  return 1.0+Math.min((br-r)/Math.max(r,0.2),0.3);
}
// CORE: SMOOTH blue(212)->green(140)->yellow(52)->red(0) over progress 0..1,
// stops at 1/3 (watch), 2/3 (advisory), 1 (warn). Latched red past warn.
function coreColor(p){
  if(p>=1.0) return "hsl(0,80%,55%)";                            // latched red over warn
  p=clamp(p,0,1);
  let h;
  if(p<1/3)      h=212+(140-212)*(p/(1/3));                       // blue -> green
  else if(p<2/3) h=140+(52-140)*((p-1/3)/(1/3));                  // green -> yellow
  else           h=52+(0-52)*((p-2/3)/(1/3));                     // yellow -> red
  return `hsl(${h.toFixed(0)},74%,${(36+p*16).toFixed(0)}%)`;
}
// HALO color tracks the same headline-class progress (kept for breadth tint); blue->red
function haloColor(m){ const h=212+(0-212)*clamp(m,0,1); return `hsl(${h.toFixed(0)},70%,52%)`; }
// per-class brightness proxy: that class's own joint confidence (model CI x KNN)
function headlineConfFor(r){ return clamp(r.jointConf!=null?r.jointConf:0.5,0,1); }
// gauge: core color = headline (worst) class progress on the watch/adv/warn ramp;
// halo SIZE/BRIGHTNESS = that SAME headline class's joint confidence -> "how close is the
// current-worst threat, and how sure are we about it".
function gaugeStyle(peak,headlineConf){
  const core=coreColor(peak), halo=haloColor(peak);
  const conf=clamp(headlineConf,0,1);
  const haloBlur=(5+conf*36).toFixed(0), haloSpread=(1+conf*10).toFixed(1);   // size <- confidence
  const haloAlpha=(0.25+conf*0.5).toFixed(2);                                 // brightness <- confidence
  const coreGlow=(4+clamp(peak,0,1)*12).toFixed(0);
  return `background:radial-gradient(circle at 38% 34%, ${core}, hsl(0,0%,12%));border-color:${core};`
       + `box-shadow:0 0 ${coreGlow}px 1px ${core}, 0 0 ${haloBlur}px ${haloSpread}px ${halo};opacity:${(0.7+conf*0.3).toFixed(2)}`;
}
// model CI half-width -> 0..1 confidence (tight band = confident). scale ~0.25 = fully uncertain.
function ciConfidence(lo,hi){ if(lo==null||hi==null) return 0.5; return clamp(1-((hi-lo)/2)/0.25,0,1); }
// TIER_STYLE: yellow watch, orange advisory, red warning
const TIER_STYLE={
  warn:    {rgb:"255,77,77",  light:"#ff8a8a", dark:"var(--red)",    label:"WARNING",  col:"var(--red)"},
  advisory:{rgb:"255,138,32", light:"#ffb066", dark:"#ff8a20",       label:"ADVISORY", col:"#ff8a20"},
  watch:   {rgb:"255,200,40", light:"#ffe27a", dark:"var(--amber)",  label:"WATCH",    col:"var(--amber)"},
};
function classBulbStyle(tier,conf){
  const b=clamp(conf,0.15,1); const s=TIER_STYLE[tier]||TIER_STYLE.watch;
  const blur=(6+b*16).toFixed(0), alpha=(0.3+b*0.5).toFixed(2);
  return `background:radial-gradient(circle at 38% 34%,${s.light},${s.dark});border-color:${s.light};`
       + `box-shadow:0 0 ${blur}px 1px rgba(${s.rgb},${alpha});opacity:${(0.55+b*0.45).toFixed(2)}`;
}

// build one horizon row: per-class lights + core/halo gauge.
// THREE-TIER, evaluated TOP-DOWN (show the highest tier that passes):
//   warning  (red)    <- lower bound (blo) >= warn thresh
//   advisory (orange) <- mean (br)         >= advisory thresh
//   watch    (yellow) <- upper bound (bup) >= watch thresh
// Each tier keys on a different quantity: more conservative number must clear a higher bar
// to escalate -> a confidence ladder.
function horizonRow(label,horizonKey,blendRows,th,knn){
  const knnConf = knn ? knn.conf : null;
  const lit=[]; let peak=0, headline=null;
  blendRows.forEach(r=>{
    const t=th[r.c]&&th[r.c][horizonKey]; if(!t) return;
    const adv=(t.advisory!=null?t.advisory:(t.watch+t.warn)/2);
    const p=classProgress(r.br,t);                                 // gauge progress on the MEAN
    if(p>peak){ peak=p; headline=r; }                              // current-worst class
    // upper bound = mirror of the up-side gap (skew-aware), same math as the ribbon top
    const bup = r.br*(2 - r.jointConf);
    let tier=null;
    if(r.blo>=t.warn)      tier="warn";                            // lower bound clears warn
    else if(r.br>=adv)     tier="advisory";                        // mean clears advisory
    else if(bup>=t.watch)  tier="watch";                           // upper bound clears watch
    if(tier) lit.push({...r,tier,bup,adv});
  });
  const headlineConf = headline ? clamp(headline.jointConf,0,1) : 0;
  const order={warn:0,advisory:1,watch:2};
  const lights=lit.sort((a,b)=> order[a.tier]-order[b.tier] || b.br-a.br).map(r=>{
    const s=TIER_STYLE[r.tier];
    const conf=clamp(headlineConfFor(r),0.15,1);                   // brightness proxy per class
    const ci=(r.hi!=null&&r.lo!=null)?((r.hi-r.lo)/2):null;
    return `<div class="clight"><div class="cbulb" style="${classBulbStyle(r.tier,conf)}"></div>
      <div><div class="cl-name">${r.c} <span style="color:${s.col};font-size:10px;font-weight:700">${s.label}</span></div>
      <div class="cl-sub">low ${r.blo.toFixed(2)}× · mid ${r.br.toFixed(2)} · up ${r.bup.toFixed(2)} · knn ${r.kr!=null?r.kr.toFixed(2):'—'}${knnConf!=null?' ('+knnConf.toFixed(2)+')':''}</div></div></div>`;
  }).join("");
  const meta = lit.length
    ? `<span class="hprox">${lit.length} active · peak ${(peak*100).toFixed(0)}% · conf ${(headlineConf*100).toFixed(0)}%</span>`
    : `<span class="hprox">calm · closest ${(peak*100).toFixed(0)}% to watch</span>`;
  const body = lit.length ? `<div class="lights">${lights}</div>`
                          : `<div class="hclear">${label} — clear</div>`;
  return `<div class="hrow"><div class="htop">
      <div class="gauge" style="${gaugeStyle(peak,headlineConf)}"></div>
      <div class="htier">${label}</div>${meta}</div>${body}</div>`;
}

function colHTML(title,model,noneProb,k){
  const rows=rankList(model,noneProb,k);
  const body=rows.map(r=>{const hue=CLASS_HUE[r.c]||"#8b98a8";const w=Math.max(2,Math.round(r.prob*100));
    const lo=Math.round(r.lo*100),hi=Math.round(r.hi*100);
    return `<div class="row ${r.active?'active':''}"><div class="name"><span class="dot" style="background:${hue}"></span>${r.c}
      ${r.active?'<span class="fire">▲</span>':''}</div>
      <div class="bar"><i style="width:${w}%;background:${hue}"></i><span class="ci" style="left:${lo}%;width:${Math.max(1,hi-lo)}%"></span></div>
      <div class="val">${r.prob.toFixed(2)}</div></div>`;}).join("");
  return `<div class="card"><h2>${title}</h2>${body}</div>`;
}

function knnHTML(rec){
  if(!rec.knn) return "";
  const ks=Object.keys(rec.knn);
  const blocks=ks.map(kk=>{
    const b=rec.knn[kk], v1=b.v1_distance_weighted||{}, v2=b.v2_prior_reweighted||{};
    const rows=META.classes.map(c=>({c,v1:v1[c]||0,v2:v2[c]||0}))
      .sort((a,b)=>b.v1-a.v1)
      .map(r=>{const hue=CLASS_HUE[r.c]||"#8b98a8";const w=Math.max(1,Math.round(r.v1*100));
        return `<div class="row"><div class="name"><span class="dot" style="background:${hue}"></span>${r.c}</div>
          <div class="bar"><i style="width:${w}%;background:${hue}"></i></div>
          <div class="val">${r.v1.toFixed(2)} <span style="color:var(--ink-faint)">/ ${r.v2.toFixed(2)}</span></div></div>`;}).join("");
    return `<div class="card"><h2>★ primary · 1-hour KNN · ${kk.toUpperCase()} · confidence ${b.v1_confidence.toFixed(2)}</h2>
      <div class="note" style="margin:-6px 0 10px">distance-weighted V1 / prior-reweighted V2 · bar = V1 · drives the lamps at ${(META.blend_knn_weight*100).toFixed(0)}% weight</div>${rows}</div>`;
  }).join("");
  return `<div class="cols">${blocks}</div>`;
}

function stampsHTML(rec){
  const ageMin=rec.data_latest_utc?(new Date()-new Date(rec.data_latest_utc))/60000:1e9;
  const c=ageMin>180?"bad":ageMin>90?"stale":"";
  return `<div><span class="k">now</span><span class="age" data-now="1">${utc(new Date().toISOString())}</span></div>
    <div><span class="k">last fetch</span>${utc(rec.last_fetch_utc)} <span style="color:var(--ink-faint)">(<span class="age" data-ts="${rec.last_fetch_utc}"></span>)</span></div>
    <div><span class="k">latest data</span><span class="${c}">${utc(rec.data_latest_utc)} (<span class="age" data-ts="${rec.data_latest_utc}"></span>)</span></div>`;
}
// live "time ago" ticker: recompute every second from stored timestamps with seconds
function ageStr(iso){ if(!iso)return"—"; const s=Math.max(0,Math.round((new Date()-new Date(iso))/1000));
  const h=Math.floor(s/3600), m=Math.floor((s%3600)/60), ss=s%60;
  if(h>0) return h+"h "+m+"m "+ss+"s ago"; if(m>0) return m+"m "+ss+"s ago"; return ss+"s ago"; }
function tickAges(){
  document.querySelectorAll(".age").forEach(e=>{
    if(e.dataset.now) e.textContent=utc(new Date().toISOString());
    else if(e.dataset.ts) e.textContent=ageStr(e.dataset.ts);
  });
}
function qualityHTML(rec){
  const q=rec.quality||{};const cov=(q.hourly_coverage!=null)?(q.hourly_coverage*100).toFixed(0)+"%":"—";
  const thin=q.hourly_coverage!=null&&q.hourly_coverage<0.9;const warns=(rec.active_warnings||[]).length;
  return `<div class="quality ${thin?'warn':''}">
    <div><span class="k">window</span>${q.window_ok?'ok':'<span class="flag">incomplete</span>'}</div>
    <div><span class="k">hourly coverage</span><span class="${thin?'flag':''}">${cov}</span> (${q.hourly_real_slots||0}/${q.hourly_total_slots||0})</div>
    <div><span class="k">subhourly obs</span>${q.subhourly_obs_used||0}</div>
    <div><span class="k">active nws warnings</span>${warns}</div>
    <div><span class="k">model</span>${rec.arch||'—'}</div></div>`;
}

async function load(key){
  CUR_KEY=key; await getMeta();
  const rec=await(await fetch("/api/latest?loc="+encodeURIComponent(key))).json();
  if(rec.error){document.getElementById("cols").innerHTML="<div class='empty'>"+rec.error+"</div>";return;}
  const th=await(await fetch("/api/thresholds")).json();
  const n1=rec.model_1h[META.none_label].prob, n24=rec.model_24h[META.none_label].prob;
  const knn=knnV1(rec), W=META.blend_knn_weight;
  const blend1=blended(rec.model_1h,n1,knn,W);
  const blend24=blended(rec.model_24h,n24,knn,W);

  document.getElementById("stamps").innerHTML=stampsHTML(rec);
  document.getElementById("hpanel").innerHTML=
    horizonRow("1-hour","h1",blend1,th,knn)+
    horizonRow("24-hour","h24",blend24,th,knn);
  document.getElementById("cols").innerHTML=
    colHTML("1-hour threat · P(class)",rec.model_1h,n1,META.red_k)+
    colHTML("24-hour threat · P(class)",rec.model_24h,n24,META.yellow_k);
  document.getElementById("knn").innerHTML=knnHTML(rec);
  document.getElementById("qwrap").innerHTML=qualityHTML(rec);
  document.getElementById("refreshnote").textContent=
    "loaded "+utc(new Date().toISOString())+" · lights use per-class watch/warn thresholds · blend "+(W*100).toFixed(0)+"% KNN · auto-refresh every 5 min";
}

// tick the "time ago" fields every second; refetch data every 5 minutes
setInterval(tickAges, 1000);
setInterval(()=>{ if(CUR_KEY) load(CUR_KEY); }, 5*60*1000);

(async()=>{ const cur=await mountLocations(load); if(cur)load(cur);
  else document.getElementById("cols").innerHTML="<div class='empty'>No locations logged yet. Run live_engine.py.</div>"; })();