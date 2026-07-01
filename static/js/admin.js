
let SEL_ADD=new Set(), SEL_RM=null, CUR=null;
function toast(id,msg,ok){const e=document.getElementById(id);e.className="toast "+(ok?"ok":"err");e.textContent=msg;}
async function load(key){
  CUR=key; await getMeta();
  const rec=await(await fetch("/api/latest?loc="+encodeURIComponent(key))).json();
  const host=document.getElementById("knnlive");
  if(rec.error||!rec.knn){host.innerHTML="<div class='empty'>No KNN in latest record.</div>";}
  else{
    const m1=rec.model_1h||{};
    host.innerHTML=Object.keys(rec.knn).map(kk=>{const b=rec.knn[kk];
      const v1=b.v1_distance_weighted,v2=b.v2_prior_reweighted;
      const rows=META.classes.map(c=>{
        const mc=m1[c]||{}; const ci=(mc.hi!=null&&mc.lo!=null)?((mc.hi-mc.lo)/2):null;
        return `<tr><td><span class="dot" style="background:${CLASS_HUE[c]}"></span> ${c}</td>
        <td class="num">${(v1[c]||0).toFixed(3)}</td><td class="num">${(v2[c]||0).toFixed(3)}</td>
        <td class="num">${mc.prob!=null?mc.prob.toFixed(2):'—'}${ci!=null?' ±'+ci.toFixed(2):''}</td></tr>`;}).join("");
      return `<div style="margin-bottom:14px"><div class="note">${kk.toUpperCase()} · KNN confidence ${b.v1_confidence.toFixed(3)}</div>
        <table><tr><th>class</th><th style="text-align:right">V1 dist-wtd</th><th style="text-align:right">V2 prior-rewtd</th><th style="text-align:right">model 1h ±CI</th></tr>${rows}</table></div>`;}).join("");
  }
  const refs=await(await fetch("/api/knn/refs")).json();
  const rc=document.getElementById("refcounts");
  if(!META.has_refs){rc.innerHTML="<div class='empty'>No refs CSV configured (start server with --refs-csv).</div>";}
  else{
    const rows=META.classes.map(c=>`<tr><td><span class="dot" style="background:${CLASS_HUE[c]}"></span> ${c}</td>
      <td class="num">${refs.counts[c]||0}</td></tr>`).join("");
    rc.innerHTML=`<div class="note">${refs.total} references · ${refs.dim}-dim<br>${refs.path||''}</div>
      <table><tr><th>class</th><th style="text-align:right">refs</th></tr>${rows}</table>`;
  }
  const recsel=document.getElementById("recpick");
  const list=await(await fetch("/api/knn/records?loc="+encodeURIComponent(key))).json();
  const opts=list.filter(r=>r.has_emb).map(r=>`<option value="${r.idx}">${utc(r.data_latest_utc)}</option>`).join("");
  recsel.innerHTML=opts||"<option value=''>no records with embeddings</option>";
  const addc=document.getElementById("addchips"); addc.innerHTML="";
  const rmc=document.getElementById("rmchips"); rmc.innerHTML="";
  SEL_ADD=new Set(); SEL_RM=null;
  const addEls={}, rmEls={};
  META.classes.forEach(c=>{                       // all classes incl. none
    const a=document.createElement("span");a.className="chip";a.textContent=c;addEls[c]=a;
    a.onclick=()=>{
      if(SEL_ADD.has(c)) SEL_ADD.delete(c);
      else{
        // none is mutually exclusive with hazards: picking none clears the rest,
        // picking a hazard clears none -> no contradictory refs from one window
        if(c===META.none_label) SEL_ADD.clear(); else SEL_ADD.delete(META.none_label);
        SEL_ADD.add(c);
      }
      META.classes.forEach(x=>addEls[x].classList.toggle("sel",SEL_ADD.has(x)));
    };
    addc.appendChild(a);
    const b=document.createElement("span");b.className="chip";b.textContent=c;rmEls[c]=b;
    b.onclick=()=>{SEL_RM=c; META.classes.forEach(x=>rmEls[x].classList.toggle("sel",x===c));};
    rmc.appendChild(b);
  });
}
document.getElementById("addbtn").onclick=async()=>{
  const idx=parseInt(document.getElementById("recpick").value);
  if(isNaN(idx)){toast("addtoast","pick a window",false);return;}
  if(!SEL_ADD.size){toast("addtoast","select at least one class",false);return;}
  const j=await(await fetch("/api/knn/add",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({loc:CUR,record_idx:idx,classes:[...SEL_ADD]})})).json();
  if(j.ok){toast("addtoast","added "+j.added+" ref(s); total "+j.total,true);load(CUR);}
  else toast("addtoast",j.error||"failed",false);
};
document.getElementById("rmbtn").onclick=async()=>{
  if(!SEL_RM){toast("rmtoast","select a class",false);return;}
  const n=parseInt(document.getElementById("rmn").value);
  const j=await(await fetch("/api/knn/remove",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({classes:[SEL_RM],n})})).json();
  if(j.ok){toast("rmtoast","removed "+j.removed+"; total "+j.total,true);load(CUR);}
  else toast("rmtoast",j.error||"failed",false);
};
(async()=>{ const cur=await mountLocations(load); if(cur)load(cur); })();