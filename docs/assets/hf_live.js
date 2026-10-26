/* Live Space tools. All measures come from the real Python endpoints. */
(()=>{'use strict';
const $=id=>document.getElementById(id);
const safe=s=>String(s).replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
let batchRows=[];
async function selectedFile(){
  if(window.eyeqcService?.file)return window.eyeqcService.file;
  if(typeof gallery==='undefined'||!gallery.length)throw Error('Select a photograph first.');
  const e=gallery[current],url=new URL(e.image,document.baseURI);
  const r=await fetch(url);if(!r.ok)throw Error('Cannot load the selected photograph.');
  return new File([await r.blob()],e.id+'.webp',{type:'image/webp'});
}
async function post(endpoint,values={}){
  const base=window.eyeqcService?.base;
  if(!base)throw Error('The Python engine is unavailable. Check the Space status.');
  const f=await selectedFile(),body=new FormData();body.append('photo',f,f.name);
  Object.entries(values).forEach(([k,v])=>body.append(k,String(v)));
  const resp=await fetch(base+'/api/'+endpoint,{method:'POST',body});
  let out;try{out=await resp.json()}catch{throw Error('The server returned an unreadable response.');}
  if(!resp.ok)throw Error(String(out.detail||'Request failed ('+resp.status+')'));
  return out;
}
function lineChart(rows){
  if(!rows?.length)return '';
  let values=rows.map(r=>Number(r.values?.local_readability||0)),w=480,h=165,p=24;
  const x=i=>p+(w-2*p)*i/(values.length-1),y=v=>h-p-v*(h-2*p);
  const points=values.map((v,i)=>`${x(i)},${y(v)}`).join(' ');
  return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Measured local readability at five degradation doses">
   ${[0,.25,.5,.75,1].map(v=>`<line x1="${p}" x2="${w-p}" y1="${y(v)}" y2="${y(v)}" stroke="#405961" stroke-opacity=".45"/>`).join('')}
   <polyline points="${points}" fill="none" stroke="#72dfc4" stroke-width="3" stroke-linejoin="round"/>
   ${values.map((v,i)=>`<circle cx="${x(i)}" cy="${y(v)}" r="4" fill="#b6f5e4"/><text x="${x(i)}" y="${h-5}" fill="#b9d7d2" text-anchor="middle" font-size="12">${Math.round(Number(rows[i].dose)*100)}%</text>`).join('')}
   <text x="${p}" y="12" fill="#c9efdf" font-size="12">Measured local readability · 0–1</text>
  </svg>`;
}
function init(){
 const run=$('live-run'),slider=$('live-severity'),out=$('live-result');
 if(!run||!slider)return;
 slider.oninput=()=>$('live-severity-label').textContent=slider.value+'%';
 run.onclick=async()=>{
  run.disabled=true;out.innerHTML='<p>Running the original Python measurement and five-dose analysis…</p>';
  try{
   const kind=$('defect-select').value,dose=Number(slider.value)/100;
   const a=await post('degradation',{kind,dose});
   const curve=await post('dose-curve',{kind});
   const before=a.measurements_before||{},after=a.measurements_after||{};
   let measures=['local_readability','focus','contrast','illumination_uniformity'];
   let rows=measures.map(k=>{const b=before[k],v=after[k];return `<div><small>${safe(k.replaceAll('_',' '))}</small><b>${v==null?'—':Number(v).toFixed(3)}</b><small>From ${b==null?'—':Number(b).toFixed(3)}</small></div>`}).join('');
   out.innerHTML=`<div class="live-result-grid"><div><img src="${a.photo_png}" alt="Image changed by the selected degradation"/><p>Altered photograph</p></div><div><img src="${a.layers?.map_png||a.layers?.overlay_png||''}" alt="Image-derived local-readability map"/><p>Measured quality map</p></div><div><img src="${a.layers?.overlay_png||''}" alt="Local quality overlaid on altered photograph"/><p>Quality overlay</p></div></div><div class="live-measures">${rows}</div><div><p>Measured dose response for this photograph · values are not trained-model risk</p>${lineChart(curve.series)}</div>`;
  }catch(err){out.innerHTML=`<p>${safe(err.message)}</p>`}
  finally{run.disabled=false}
 };
 const input=$('batch-files'),submit=$('batch-run'),csv=$('batch-csv'),status=$('batch-status');
 submit.onclick=async()=>{
  if(!window.eyeqcService?.base){status.textContent='Python service unavailable; check the Space status.';return}
  const files=Array.from(input.files||[]);
  if(!files.length){status.textContent='Choose up to 20 images first.';return}
  if(files.length>20){status.textContent='Please select no more than 20 images.';return}
  submit.disabled=true;status.textContent=`Measuring ${files.length} photographs…`;
  try{
   const fd=new FormData();files.forEach(f=>fd.append('photos',f,f.name));
   const res=await fetch(window.eyeqcService.base+'/api/batch',{method:'POST',body:fd});
   const json=await res.json();if(!res.ok)throw Error(String(json.detail||res.status));
   batchRows=json.rows||[];csv.disabled=!batchRows.length;
   status.innerHTML=`<strong>${batchRows.filter(x=>!x.error).length} of ${batchRows.length} photographs measured.</strong> Download the CSV for all ten measurements. No trained-model risk is generated.`;
  }catch(err){status.textContent=safe(err.message)}finally{submit.disabled=false}
 };
 csv.onclick=()=>{
  if(!batchRows.length)return;
  const fns=['file','scope','error','focus','detail_uniformity','local_readability','illumination_uniformity','exposure','contrast','field_coverage','artifact_burden','clipping','color_balance'];
  const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';
  const text=[fns.join(','),...batchRows.map(r=>fns.map(k=>quote(k in r?r[k]:r.measurements?.[k])).join(','))].join('\n');
  const u=URL.createObjectURL(new Blob([text],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=u;a.download='EyeQC_batch_measurements.csv';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);
 };
}
window.addEventListener('eyeqc:ready',init);if(typeof gallery!=='undefined'&&gallery.length)init();
})();
