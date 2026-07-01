
const bandPlugin={id:'bands',beforeDraw(chart){
  const ranges=(chart.options.plugins.bands||{}).ranges||[];
  const a=chart.chartArea, x=chart.scales.x, ctx=chart.ctx; if(!a)return; ctx.save();
  ranges.forEach(b=>{const x0=x.getPixelForValue(b.start),x1=x.getPixelForValue(b.end);
    ctx.fillStyle=b.color; ctx.fillRect(x0,a.top,Math.max(2,x1-x0),a.bottom-a.top);}); ctx.restore();}};
// horizontal threshold line (per chart, set live from the slider)
const threshPlugin={id:'thresh',afterDraw(chart){
  const tp=chart.options.plugins.thresh||{}; if(tp.value==null)return;
  const a=chart.chartArea, y=chart.scales.y, ctx=chart.ctx; if(!a)return;
  const yy=y.getPixelForValue(tp.value); if(yy<a.top||yy>a.bottom)return;
  ctx.save(); ctx.strokeStyle=tp.color||"#e6edf3"; ctx.setLineDash([6,4]); ctx.lineWidth=1.4;
  ctx.beginPath(); ctx.moveTo(a.left,yy); ctx.lineTo(a.right,yy); ctx.stroke();
  ctx.setLineDash([]); ctx.fillStyle=tp.color||"#e6edf3"; ctx.font="10px monospace";
  ctx.fillText((tp.label||"")+" "+tp.value.toFixed(2)+"x", a.left+4, yy-3); ctx.restore();}};
Chart.register(bandPlugin, threshPlugin);

function tsms(iso){return new Date(iso).getTime();}
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function xtick(v){const d=new Date(v);return (d.getUTCMonth()+1)+"/"+d.getUTCDate()+" "+String(d.getUTCHours()).padStart(2,"0")+"Z";}
function lineCfg(label,pts,hue,dash,w,alpha){return {label,data:pts,borderColor:hue,backgroundColor:hue,
  borderWidth:w||1.6,pointRadius:0,tension:.25,fill:false,borderDash:dash||[],
  borderColor:alpha!=null?`rgba(${hexrgb(hue)},${alpha})`:hue};}
const YMAX_BASE=3, YMAX_CAP=5;           // x-none axis: default 3x, auto-extend to 5x
let CHARTS={}, RECS=[], TH=null;

// ---- week-window state --------------------------------------------------
// RECS holds the full history (unchanged). VIEW picks what rebuildCharts reads:
//  mode "all"  -> every record (the original behavior)
//  mode "week" -> a 7-day slice ending at WIN.end, chosen by the slider.
// When windowed we also pin the x-axis to [start,end] so the week fills the
// width instead of being squished at the right edge (the whole point: no squint).
const WEEK_MS = 7*24*3600*1000;
const VIEW = { mode:"all", end:null };   // end = ms timestamp of the window's right edge

function recTs(r){ return tsms(r.data_latest_utc); }

// records whose timestamp falls in the active window (or all of them in "all" mode)
function windowedRecs(){
  if(VIEW.mode!=="week" || !RECS.length) return RECS;
  const end = VIEW.end!=null ? VIEW.end : recTs(RECS[RECS.length-1]);
  const start = end - WEEK_MS;
  return RECS.filter(r=>{ const x=recTs(r); return x>=start && x<=end; });
}
// the [min,max] the x-axis should span for the current view (null -> auto)
function windowSpan(){
  if(VIEW.mode!=="week" || !RECS.length) return null;
  const end = VIEW.end!=null ? VIEW.end : recTs(RECS[RECS.length-1]);
  return { min:end-WEEK_MS, max:end };
}


// blended none-relative ratio for one class at one record (matches the home page)
function blendRatio(rec,cls,horizon,W){
  const model=horizon==="h1"?rec.model_1h:rec.model_24h; if(!model||!model[cls])return null;
  const none=model[META.none_label].prob; const rr=none>0?model[cls].prob/none:0;
  let kr=0, has=false;
  if(rec.knn){const kk=rec.knn["k15"]||rec.knn[Object.keys(rec.knn)[0]];
    if(kk){const v1=kk.v1_distance_weighted||{}; const kn=v1[META.none_label]||0;
      kr=Math.min((v1[cls]||0)/Math.max(kn,0.02),5); has=true;}}
  const w=has?W:0; return (1-w)*rr+w*kr;
}
// confidence-discounted LOWER BOUND for one class (the quantity the home-page lights
// threshold on): blended * (modelConf * knnConf), same math as blended() in NOW_JS.
function blendLower(rec,cls,horizon,W){
  const model=horizon==="h1"?rec.model_1h:rec.model_24h; if(!model||!model[cls])return null;
  const br=blendRatio(rec,cls,horizon,W); if(br==null)return null;
  const p=model[cls].prob, lo=model[cls].lo;
  const modelConf=(p>0&&lo!=null)?(1-clamp((p-lo)/p,0,1)):0.5;
  let knnConf=null;
  if(rec.knn){const kk=rec.knn["k15"]||rec.knn[Object.keys(rec.knn)[0]]; if(kk)knnConf=kk.v1_confidence;}
  const jointConf=(knnConf!=null)?(modelConf*knnConf):modelConf;
  return br*jointConf;
}

function rebuildCharts(){
  const host=document.getElementById("charts"); const W=META.blend_knn_weight;
  const recs=windowedRecs();
  const t=recs.map(r=>tsms(r.data_latest_utc));
  host.innerHTML=""; CHARTS={};
  if(!recs.length){host.innerHTML="<div class=" + chr(39) + "empty" + chr(39) + ">No records in this week. Slide toward newer data.</div>";return;}
  META.classes.forEach(cls=>{
    const hue=CLASS_HUE[cls]||"#8b98a8";
    const blend1=recs.map((r,i)=>({x:t[i],y:blendRatio(r,cls,"h1",W)}));
    const knn=recs.map((r,i)=>{let y=null;if(r.knn){const kk=r.knn["k15"]||r.knn[Object.keys(r.knn)[0]];
      if(kk){const v1=kk.v1_distance_weighted||{};const kn=v1[META.none_label]||0;
        y=Math.min((v1[cls]||0)/Math.max(kn,0.02),5);}}return {x:t[i],y};});
    const mdl1=recs.map((r,i)=>{const m=r.model_1h;const none=m[META.none_label].prob;
      return {x:t[i],y:none>0?(m[cls]||{}).prob/none:null};});
    // BLENDED confidence ribbon = asymmetric [lowerBound, upperBound], skew-aware.
    //   modelDownConf = 1 - (prob-lo)/prob   (MC downside)  -> drives lower (lights use it)
    //   modelUpConf   = 1 - (hi-prob)/prob   (MC upside)    -> drives upper (graph only)
    //   knnConf       = v1_confidence (single scalar, both sides)
    //   lower = blended * (modelDownConf * knnConf)            [coincides with the dotted line]
    //   upper = blended * (2 - modelUpConf * knnConf)          [mirror of the gap, above center]
    // When hi is farther from the mean than lo, modelUpConf < modelDownConf, so the upper
    // gap exceeds the lower gap and the band leans up -> the MC skew is preserved.
    const ciHi=[], ciLo=[];
    recs.forEach((r,i)=>{
      const m=r.model_1h, none=m[META.none_label].prob, mc=m[cls]||{};
      const by=blendRatio(r,cls,"h1",W);
      if(by==null||none<=0){ciHi.push({x:t[i],y:null});ciLo.push({x:t[i],y:null});return;}
      const p=mc.prob, lo=mc.lo, hi=mc.hi;
      const downConf=(p>0&&lo!=null)?(1-clamp((p-lo)/p,0,1)):0.5;
      const upConf  =(p>0&&hi!=null)?(1-clamp((hi-p)/p,0,1)):0.5;
      let knnConf=null;
      if(r.knn){const kk=r.knn["k15"]||r.knn[Object.keys(r.knn)[0]]; if(kk)knnConf=kk.v1_confidence;}
      const downJoint=(knnConf!=null)?(downConf*knnConf):downConf;
      const upJoint  =(knnConf!=null)?(upConf*knnConf):upConf;
      ciLo.push({x:t[i],y:by*downJoint});          // = blendLower (dotted line sits here)
      ciHi.push({x:t[i],y:by*(2-upJoint)});        // mirror of the up-gap, above center
    });
    const lower1=recs.map((r,i)=>({x:t[i],y:blendLower(r,cls,"h1",W)}));
    const ranges=[];
    recs.forEach((r,i)=>{(r.active_warnings||[]).forEach(w=>{
      if((w.classes||[]).includes(cls)) ranges.push({start:t[i]-18e5,end:t[i]+18e5,color:"rgba("+hexrgb(hue)+",0.18)"});});});
    // per-chart y-axis cap: 3x by default, auto-extend toward 5x only when this class's
    // data (or a threshold) actually exceeds 3x -> calm classes stay readable, spiky ones
    // (t-storm) get the headroom without squashing everyone else.
    let dmax=0; [blend1,lower1,ciHi].forEach(arr=>arr.forEach(p=>{if(p.y!=null&&p.y>dmax)dmax=p.y;}));
    const tcls=(TH&&TH[cls])?TH[cls]:null;
    if(tcls) ["h1","h24"].forEach(h=>["watch","advisory","warn"].forEach(k=>{const v=tcls[h]&&tcls[h][k]; if(v!=null&&v>dmax)dmax=v;}));
    const ymax = dmax>YMAX_BASE ? YMAX_CAP : YMAX_BASE;
    const card=document.createElement("div");card.className="card chartcard";card.dataset.cls=cls;card.dataset.ymax=ymax;
    card.style.setProperty("--thumb", hue);   // slider triangle matches this class's line color
    card.innerHTML=`<h3><span class="dot" style="background:${hue}"></span>${cls}
      <span class="legend">solid = blended · thick dotted = lower bound (warning) · ribbon = CI · faint = model / KNN · shaded = NWS warning · dashed = threshold</span></h3>
      <div class="chartrow">
        <div class="vslide"><input type="range" orient="vertical" min="0" max="${ymax}" step="0.01">
          <div class="vticks"></div></div>
        <div class="chartwrap"><canvas></canvas></div>
      </div>
      <div class="tuner">
        <div class="field"><b class="tval">—</b>× none
          <span class="hzbtns tierbtns"><button data-hz="h1" class="on">1-hour</button><button data-hz="h24">24-hour</button></span>
          <span class="tierbtns"><button data-tier="watch">watch</button><button data-tier="advisory">advisory</button><button data-tier="warn" class="on">warning</button></span>
          <span class="toast tsaved"></span>
        </div>
      </div>`;
    host.appendChild(card);
    CHARTS[cls]=new Chart(card.querySelector("canvas"),{type:"line",
      data:{datasets:[
        {label:"ciHi",data:ciHi,borderColor:`rgba(${hexrgb(hue)},0)`,backgroundColor:`rgba(${hexrgb(hue)},0.13)`,
         borderWidth:0,pointRadius:0,fill:"+1",tension:.25},
        {label:"ciLo",data:ciLo,borderColor:`rgba(${hexrgb(hue)},0)`,backgroundColor:`rgba(${hexrgb(hue)},0.13)`,
         borderWidth:0,pointRadius:0,fill:false,tension:.25},
        lineCfg("blended",blend1,hue,[],2.2),
        lineCfg("lower",lower1,hue,[1,3],2.2),                    // thick like center, dotted like KNN
        lineCfg("model1h",mdl1,hue,[5,3],1,0.35),
        lineCfg("knn15",knn,hue,[1,3],1,0.5)]},
      options:{responsive:true,maintainAspectRatio:false,animation:false,spanGaps:true,
        scales:{x:{type:"linear",min:(windowSpan()?windowSpan().min:undefined),max:(windowSpan()?windowSpan().max:undefined),ticks:{color:"#5a6675",font:{size:10},maxTicksLimit:8,callback:xtick},grid:{color:"rgba(38,50,65,.4)"}},
          y:{min:0,max:ymax,ticks:{color:"#5a6675",font:{size:10},callback:v=>v+"x"},grid:{color:"rgba(38,50,65,.4)"}}},
        plugins:{legend:{display:false},bands:{ranges},thresh:{value:null},
          tooltip:{enabled:true,filter:it=>it.dataset.label==="blended",callbacks:{
            title:it=>xtick(it[0].parsed.x),
            afterBody:it=>{const i=it[0].dataIndex,r=recs[i],m=r.model_1h;
              const none=m[META.none_label].prob, ci=none>0?(((m[cls]||{}).hi-(m[cls]||{}).lo)/2/none):null;
              let kc=null; if(r.knn){const kk=r.knn["k15"]||r.knn[Object.keys(r.knn)[0]]; if(kk)kc=kk.v1_confidence;}
              return "model CI ±"+(ci!=null?ci.toFixed(2):"—")+"x · knn conf "+(kc!=null?kc.toFixed(2):"—");}}}}}});
    wireCardTuner(card, cls, ymax);
  });
}

const TIER_COLOR={watch:"#ffc828",advisory:"#ff8a20",warn:"#ff4d4d"};

// per-card tuner: accordion open/close, horizon+tier toggles, VERTICAL slider whose
// handle height maps to the threshold value on the chart's own y-scale (0..ymax), clamped
// to the tier's neighbor bounds with notch ticks, live threshold line + debounced save.
function wireCardTuner(card, cls, ymax){
  const h3=card.querySelector("h3"), tuner=card.querySelector(".tuner");
  const sl=card.querySelector(".vslide input[type=range]"), tval=card.querySelector(".tval");
  const ticks=card.querySelector(".vticks"), tsaved=card.querySelector(".tsaved");
  const wrap=card.querySelector(".chartwrap"), vslide=card.querySelector(".vslide");
  const sel={hz:"h1",tier:"warn"};

  h3.onclick=()=>{
    const wasOpen=card.classList.contains("open");
    document.querySelectorAll(".chartcard.open").forEach(c=>c.classList.remove("open"));
    if(!wasOpen){ card.classList.add("open"); requestAnimationFrame(()=>{anchorSlider(); syncSlider();}); }
    applyAllThreshLines();
  };
  tuner.onclick=e=>e.stopPropagation();
  vslide.onclick=e=>e.stopPropagation();

  // size the vertical slider track to the chart's rendered plot area (top..bottom of the
  // y-axis), so the handle sits at the same height as the dashed line it draws.
  // position the slider gutter over the chart's rendered plot area (top..bottom of the
  // y-axis), so the handle height maps to the threshold value. The native vertical slider
  // (writing-mode) fills .vslide via height:100%, so we only set top + height here.
  function anchorSlider(){
    const ch=CHARTS[cls]; if(!ch||!ch.chartArea) return;
    const a=ch.chartArea;                          // pixels within the chartwrap canvas
    const wrapTop=wrap.offsetTop;                  // chartwrap's offset inside .chartrow
    vslide.style.top=(wrapTop+a.top)+"px";
    vslide.style.height=(a.bottom-a.top)+"px";
  }
  function tierBounds(){
    const t=TH[cls][sel.hz];
    if(sel.tier==="advisory"){                     // must sit BETWEEN watch and warn, either order
      return [Math.min(t.watch,t.warn), Math.max(t.watch,t.warn)];
    }
    return [0, ymax];                              // watch and warn are free across the axis
  }
  function renderTicks(){
    ticks.innerHTML="";
    if(sel.tier!=="advisory") return;              // notches matter most for the clamped advisory
    const t=TH[cls][sel.hz];
    [["watch",t.watch],["warn",t.warn]].forEach(([nm,v])=>{
      const pct=clamp(v/ymax,0,1)*100; const d=document.createElement("div");
      d.className="vtick"; d.style.bottom=pct+"%"; d.textContent=nm+" "+v.toFixed(2);
      ticks.appendChild(d);
    });
  }
  function syncSlider(){
    const v=TH[cls][sel.hz][sel.tier]; sl.max=ymax;
    sl.value=(v!=null?v:0); tval.textContent=(v!=null?v:0).toFixed(2);
    renderTicks(); projectLine();
  }
  function projectLine(){
    const v=parseFloat(sl.value);
    CHARTS[cls].options.plugins.thresh={value:v,color:TIER_COLOR[sel.tier],
      label:cls+" "+(sel.hz==="h1"?"1h":"24h")+" "+sel.tier};
    CHARTS[cls].update("none");
  }
  card.querySelectorAll(".hzbtns button").forEach(b=>b.onclick=()=>{
    card.querySelectorAll(".hzbtns button").forEach(x=>x.classList.remove("on"));
    b.classList.add("on"); sel.hz=b.dataset.hz; syncSlider();
  });
  card.querySelectorAll(".tierbtns:not(.hzbtns) button").forEach(b=>b.onclick=()=>{
    card.querySelectorAll(".tierbtns:not(.hzbtns) button").forEach(x=>x.classList.remove("on"));
    b.classList.add("on"); sel.tier=b.dataset.tier; syncSlider();
  });
  sl.oninput=()=>{ const [lo,hi]=tierBounds(); let v=clamp(parseFloat(sl.value),lo,hi);
    sl.value=v; tval.textContent=v.toFixed(2);
    TH[cls][sel.hz][sel.tier]=v; projectLine(); };
  let saveTimer=null;
  sl.onchange=()=>{ const [lo,hi]=tierBounds(); const v=clamp(parseFloat(sl.value),lo,hi);
    clearTimeout(saveTimer); saveTimer=setTimeout(async()=>{
      const j=await(await fetch("/api/thresholds/set",{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({class:cls,horizon:sel.hz,tier:sel.tier,value:v})})).json();
      if(j.ok){tsaved.className="toast ok";tsaved.textContent="saved "+v.toFixed(2)+"x"; TH[cls]=j.thresholds; syncSlider();}
      else{tsaved.className="toast err";tsaved.textContent=j.error||"save failed";}
    },150); };
  // re-anchor on window resize while open
  window.addEventListener("resize",()=>{ if(card.classList.contains("open")) anchorSlider(); });
}

function applyAllThreshLines(){
  // clear lines on collapsed cards so only the open one projects
  Object.entries(CHARTS).forEach(([cls,ch])=>{
    const card=document.querySelector('.chartcard[data-cls="'+CSS.escape(cls)+'"]');
    if(!card||!card.classList.contains("open")){ ch.options.plugins.thresh={value:null}; ch.update("none"); }
  });
}

// ---- week-window controls -----------------------------------------------
// The slider indexes "week ending at record i". Sliding left walks the window
// back through history one record-step at a time; "latest" snaps to the newest.
// Rebuilding charts is cheap here (same path as a location switch), and pinning
// the x-axis makes the chosen week fill the width.
function wireWindow(){
  const bar=document.getElementById("winbar");
  const bAll=document.getElementById("win-all"), bWeek=document.getElementById("win-week");
  const ctrls=document.getElementById("win-ctrls"), slider=document.getElementById("win-slider");
  const label=document.getElementById("win-label"), bLatest=document.getElementById("win-latest");
  if(!bar) return;
  bar.style.display = RECS.length ? "flex" : "none";

  function fmtRange(end){
    const s=new Date(end-WEEK_MS), e=new Date(end);
    const d=t=>(t.getUTCMonth()+1)+"/"+t.getUTCDate();
    return d(s)+" → "+d(e)+" (week ending "+xtick(end)+")";
  }
  function refreshLabel(){
    label.textContent = VIEW.mode==="week" && VIEW.end!=null ? fmtRange(VIEW.end) : "";
  }
  function setMode(mode){
    VIEW.mode=mode;
    bAll.classList.toggle("on", mode==="all");
    bWeek.classList.toggle("on", mode==="week");
    ctrls.style.display = mode==="week" ? "flex" : "none";
    if(mode==="week"){
      slider.min=0; slider.max=Math.max(0,RECS.length-1);
      if(VIEW.end==null) slider.value=slider.max;         // default: newest week
      VIEW.end = recTs(RECS[Math.round(+slider.value)]);
    }
    refreshLabel(); rebuildCharts();
  }
  bAll.onclick   = ()=>setMode("all");
  bWeek.onclick  = ()=>setMode("week");
  bLatest.onclick= ()=>{ slider.value=slider.max; VIEW.end=recTs(RECS[RECS.length-1]); refreshLabel(); rebuildCharts(); };
  slider.oninput = ()=>{ VIEW.end=recTs(RECS[Math.round(+slider.value)]); refreshLabel(); rebuildCharts(); };
}

async function load(key){
  await getMeta();
  RECS=await(await fetch("/api/history?loc="+encodeURIComponent(key))).json();
  TH=await(await fetch("/api/thresholds")).json();
  const host=document.getElementById("charts");
  if(!RECS.length){host.innerHTML="<div class='empty'>No history yet.</div>";return;}
  const last=RECS[RECS.length-1];
  document.getElementById("stamps").innerHTML=
    `<div><span class="k">records</span>${RECS.length}</div>
     <div><span class="k">span</span>${utc(RECS[0].data_latest_utc)} → ${utc(last.data_latest_utc)}</div>
     <div><span class="k">blend</span>${(META.blend_knn_weight*100).toFixed(0)}% KNN</div>`;
  // populate class dropdown
  VIEW.mode="all"; VIEW.end=null;   // reset window when (re)loading a location
  rebuildCharts();
  wireWindow();
}

(async()=>{ const cur=await mountLocations(load); if(cur)load(cur); })();