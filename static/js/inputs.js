function tsms(iso){return new Date(iso).getTime();}
function xtick(v){const d=new Date(v);return (d.getUTCMonth()+1)+"/"+d.getUTCDate()+" "+String(d.getUTCHours()).padStart(2,"0")+"Z";}

const CH=[
  {k:"temp",label:"Temperature",unit:"°C",hue:"#ff6b4a",ribbonHue:"rgba(255,107,74,0.3)"},
  {k:"pressure",label:"Pressure",unit:"hPa",hue:"#4aa3ff",ribbonHue:"rgba(74,163,255,0.3)"},
  {k:"humidity",label:"Humidity",unit:"%",hue:"#39d0c8",ribbonHue:"rgba(57,208,200,0.3)"}
];

async function load(key){
  const recs=await(await fetch("/api/history?loc="+encodeURIComponent(key))).json();
  const host=document.getElementById("charts");
  if(!recs.length){host.innerHTML="<div class='empty'>No history yet.</div>";return;}
  const withObs=recs.filter(r=>r.obs_latest);
  const last=recs[recs.length-1];
  
  document.getElementById("stamps").innerHTML= `
    <div><span class="k">now</span>${utc(new Date().toISOString())}</div>
    <div><span class="k">latest data</span>${utc(last.data_latest_utc)} (${fmtAge(last.data_latest_utc)})</div>
    <div><span class="k">observations</span>${withObs.length}</div>`;
    
  if(!withObs.length){host.innerHTML="<div class='empty'>No observations logged yet.</div>";return;}
  
  const t=withObs.map(r=>tsms(r.data_latest_utc));
  host.innerHTML="";
  
  CH.forEach(ch=>{
    const pts=withObs.map((r,i)=>({x:t[i],y:r.obs_latest[ch.k]}));
    const card=document.createElement("div");
    card.className="card chartcard";
    card.innerHTML=`<h3><span class="dot" style="background:${ch.hue}"></span>${ch.label} <span class="legend">${ch.unit}</span></h3><div class="chartwrap"><canvas></canvas></div>`;
    host.appendChild(card);
    
    function getValueAtTime(targetTime) {
      let closestIdx = 0;
      let minDiff = Infinity;
      for(let i=0; i<t.length; i++) {
        let diff = Math.abs(t[i] - targetTime);
        if(diff < minDiff) { minDiff = diff; closestIdx = i; }
      }
      return minDiff > 5 * 60 * 1000 ? null : pts[closestIdx].y;
    }

    // --- REUSABLE PROJECTION DRAWER ---
    function drawProjection(targetIdx) {
      if (targetIdx < 0 || targetIdx >= pts.length) return;
      
      const t0 = pts[targetIdx].x;
      const hourMs = 60 * 60 * 1000;
      
      const v_0   = pts[targetIdx].y;
      const v_30  = getValueAtTime(t0 - 0.0 * hourMs) || v_0;
      const v_60  = getValueAtTime(t0 - 1.0 * hourMs) || v_0;
      const v_90  = getValueAtTime(t0 - 2.0 * hourMs) || v_60;
      const v_120 = getValueAtTime(t0 - 3.0 * hourMs) || v_60;
      
      const v_minus_1 = (v_0 + v_30) / 2;
      const v_minus_2 = (v_60 + v_90) / 2;
      
      const denom = v_minus_2 === 0 ? 1e-6 : v_minus_2;
      const pct_change = ((v_minus_1 - v_minus_2) / denom) / 1.5;
      
      const projected_mean = v_minus_1 * (1.0 + pct_change);
      
      const numbers = [Math.abs(projected_mean * pct_change), 1];
      let projected_std = Math.max(...numbers);
      if (projected_std < 1e-4) projected_std = 1e-4;
      
      const t_plus_1 = t0 + hourMs;
      
      // Update Area Ribbon (Dataset 1)
      // Connects cleanly at the anchor (v_0) and flares open to the standard deviation
  chart.data.datasets[1].data = [
  { x: t0, y: v_0 },
  { x: t_plus_1, y: projected_mean + projected_std }
];

      // Update Trend Line (Dataset 2)
      chart.data.datasets[2].data = [
  { x: t0, y: v_0 },
  { x: t_plus_1, y: projected_mean - projected_std }
];
chart.data.datasets[3].data = [
  { x: t0, y: v_0 },
  { x: t_plus_1, y: projected_mean }
];


    }

    const chart = new Chart(card.querySelector("canvas"),{
      type:"line",
      data:{
        datasets:[
  {
    label:ch.label,
    data:pts,
    borderColor:ch.hue,
    backgroundColor:ch.hue,
    borderWidth:1.7,
    pointRadius:0,
    tension:.25
  },
  {
    label: ch.label + " Upper Bound",
    data: [],
    borderColor: "transparent", 
    pointRadius: 1.5,
    pointBackgroundColor: ch.hue,
    showLine: true
  },
  {
    label: ch.label + " Lower Bound",
    data: [],
    borderColor: "transparent",
    pointRadius: 1.5,
    pointBackgroundColor: ch.hue,
    fill: '-1', // 📌 This tells Chart.js to fill the space up to the Upper Bound dataset
    backgroundColor: ch.ribbonHue,
    showLine: true
  },
  {
    label: ch.label + " Projection",
    data: [],
    borderColor: ch.hue,
    borderWidth: 1.5,
    pointRadius: 0,
    showLine: true,
    spanGaps: true
  }
],
      },
      options:{
        responsive:true,
        maintainAspectRatio:false,
        animation:false,
        spanGaps:true,
        scales:{
          x:{type:"linear",ticks:{color:"#5a6675",font:{size:5},maxTicksLimit:20,callback:xtick},grid:{color:"rgba(38,50,65,.8)"}},
          y:{ticks:{color:"#5a6675",font:{size:5}},grid:{color:"rgba(38,50,65,.8)"}}
        },
        plugins:{
          legend:{display:false},
          tooltip:{
            enabled:true,
            callbacks:{title:i=>xtick(i.parsed.x)}
          }
        },
        onHover: (event, chartElements) => {
          if (!chartElements || chartElements.length === 0) {
            drawProjection(pts.length - 1);
            chart.update("none");
            return;
          }
          const activeEl = chartElements[0];
          if (activeEl && activeEl.datasetIndex === 0) {
            drawProjection(activeEl.index);
            chart.update("none");
          }
        }
      }
    });

    // Explicitly configure line dashes for dataset 2 to keep code parsing safe
    chart.data.datasets[3].borderDash =[0.1,0.2];

    setTimeout(() => {
      drawProjection(pts.length - 1);
      chart.update("none");
    }, 50);

  });
}

(async()=>{
  const cur=await mountLocations(load);
  if(cur)load(cur);
})();