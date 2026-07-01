
const CLASS_HUE={"tornado":"#c77dff","flood":"#4aa3ff","wind":"#39d0c8","t-storm":"#ffb020",
  "blizzard":"#7fdfff","severe heat":"#ff6b4a","severe cold":"#9ec5ff","none":"#5a6675"};
let META=null;
function fmtAge(iso){ if(!iso)return"—"; const t=new Date(iso),n=new Date(); const m=Math.round((n-t)/60000);
  if(m<60)return m+"m ago"; return Math.floor(m/60)+"h "+(m%60)+"m ago"; }
function utc(iso){ return iso? iso.replace("T"," ").replace(/\..*/,"").replace("+00:00","")+"Z":"—"; }
async function getMeta(){ if(!META) META=await(await fetch("/api/meta")).json(); return META; }
async function locations(){ return await(await fetch("/api/locations")).json(); }
function curLoc(){ return new URLSearchParams(location.search).get("loc"); }
function setLoc(k){ const u=new URL(location.href); u.searchParams.set("loc",k); location.href=u; }
async function mountLocations(onPick){
  const locs=await locations(); const sel=document.getElementById("loc"); sel.innerHTML="";
  locs.forEach(l=>{const o=document.createElement("option");o.value=l.key;o.textContent=l.location+" ("+l.station+")";sel.appendChild(o);});
  const cur=curLoc()||(locs[0]&&locs[0].key); if(cur)sel.value=cur;
  sel.onchange=()=>{ if(onPick) onPick(sel.value); else setLoc(sel.value); };
  return cur;
}
function hexrgb(h){h=h.replace('#','');return [parseInt(h.substr(0,2),16),parseInt(h.substr(2,2),16),parseInt(h.substr(4,2),16)].join(',');}