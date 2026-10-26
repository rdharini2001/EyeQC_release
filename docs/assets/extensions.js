/* EyeQC research extensions. Uses measured images and the optional first-party Python service. */
(()=>{
'use strict';
const get=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let mode='overlay',alpha=.65,ready=false,uploadPreviewUrl=null;
function showCase(){if(!ready||!gallery.length)return; if(uploadPreviewUrl){URL.revokeObjectURL(uploadPreviewUrl);uploadPreviewUrl=null;}const g=gallery[current];
  get('ext-case-name').textContent=`Photograph ${g.label}`;
  get('ext-photo').src=asset(g.image);
  get('ext-vessel').src=asset('assets/vessels/'+g.id+'_contrast.webp');
  document.querySelectorAll('[data-vessel-mode]').forEach(b=>b.disabled=false);get('ext-save-vessel').disabled=false;get('ext-download-report').disabled=false;setMode(mode);
}
function setMode(m){mode=m;get('ext-photo').style.display=m==='contrast'?'none':'block';get('ext-vessel').style.display=m==='original'?'none':'block';get('ext-vessel').style.opacity=m==='overlay'?alpha:1;
 document.querySelectorAll('[data-vessel-mode]').forEach(b=>b.classList.toggle('active',b.dataset.vesselMode===m));
 get('vessel-source').textContent=m==='original'?'ORIGINAL IMAGE': 'IMAGE-DERIVED CONTRAST';}
function downloadJSON(){if(!gallery.length)return;const g=gallery[current];const doc={source:'EyeQC research site',sample:g.id,provenance:'Measured features and 6x6 local-readability tile map, not calibrated classifier output',features:g.features,local_readability_tiles:g.grid};const u=URL.createObjectURL(new Blob([JSON.stringify(doc,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=u;a.download='eyeqc_'+g.id+'_research.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),500)}
async function chosenPhoto(){if(window.eyeqcService?.file)return window.eyeqcService.file;const g=gallery[current];const r=await fetch(asset(g.image));if(!r.ok)throw Error('Cannot load selected photograph');return new File([await r.blob()],g.id+'.webp',{type:'image/webp'});}
async function callModel(which){const pane=get('ext-'+which+'-result');const base=window.eyeqcService?.base;
 if(!base){pane.innerHTML='Connect the Python service below. The site remains here while the model runs.';get('ext-api-url').focus();return;}
 pane.textContent='Running model on selected photograph…';const button=get('ext-run-'+which);button.disabled=true;
 try{const f=await chosenPhoto(),fd=new FormData();fd.append('photo',f,f.name);const r=await fetch(base+'/api/'+(which==='vessel'?'vessel':'flair'),{method:'POST',body:fd});const d=await r.json();if(!r.ok)throw Error(String(d.detail||`HTTP ${r.status}`));
 if(which==='vessel'){pane.innerHTML=`<strong>RRWNet artery / vein / vessel view</strong><p>${esc(d.note||'Separate research model.')}</p>${d.image_png?`<img alt="Computed RRWNet vessel maps" src="${d.image_png}">`:''}<div class="mini-proof">${Object.entries(d.features||{}).slice(0,6).map(([k,v])=>`<span title="${esc(k)}">${esc(k.replace('rrwnet_','').replaceAll('_',' '))} · ${v===null?'—':Number(v).toFixed(2)}</span>`).join('')}</div>`;}
 else{const rs=d.result?.scores||[];pane.innerHTML=`<strong>${esc(d.result?.predicted_label||'FLAIR output')}</strong><p>${esc(d.note||'Research-only comparison')}</p>`+rs.map(x=>`<div class="flair-row"><span class="flair-name" title="${esc(x.category)}">${esc(x.category)}</span><span class="flair-track"><span class="flair-fill" style="display:block;width:${(100*Math.max(0,Math.min(1,Number(x.probability)||0))).toFixed(1)}%"></span></span><span class="flair-val">${(100*Number(x.probability)).toFixed(1)}%</span></div>`).join('')+'<p>Scores compare the listed prompts for this image. They are not calibrated image-quality risks.</p>';
 }
 }catch(e){pane.textContent='Unavailable: '+e.message+'. Confirm optional dependencies, model access, and server logs.';}finally{button.disabled=false;}
}
async function autoConnect(){const input=get('ext-api-url');const u=new URL(window.location.href);
 // Same-origin Python hosting works without any manual API configuration.
 const looksLikePython=location.protocol.startsWith('http') && !location.hostname.endsWith('github.io');
 if(looksLikePython){try{const r=await fetch('/health');if(r.ok&&(await r.json()).ok){input.value=location.origin;await connect();return;}}catch(_e){}}
 // Separately hosted backend can be set once in docs/assets/config.json; never redirect visitors.
 try{const r=await fetch(asset('assets/config.json'));if(r.ok){const cfg=await r.json();if(cfg.apiBase&&/^https:\/\//.test(cfg.apiBase)){input.value=cfg.apiBase;await connect();}}}catch(_e){}
}
async function connect(){const value=get('ext-api-url').value.trim().replace(/\/+$/,'');
 if(!/^https?:\/\//.test(value)){get('ext-api-status').textContent='Enter a complete https:// server URL.';return;}
 get('api-url').value=value;
 await get('api-connect').click();
 // The existing button handles the health probe asynchronously.
 const limit=Date.now()+9000;while(Date.now()<limit&&!window.eyeqcService?.base){await new Promise(r=>setTimeout(r,180));}
 const okay=!!window.eyeqcService?.base;
 get('ext-api-status').textContent=okay?'Connected. Vessel and FLAIR analysis will run here without changing sites.':'Connection unavailable. Check /health and whether the service allows requests from this site.';
}
function init(){if(ready)return;ready=true;const originalChoose=window.chooseSample;
 if(typeof originalChoose==='function')window.chooseSample=function(i){originalChoose(i);showCase()};
 showCase();document.querySelectorAll('[data-vessel-mode]').forEach(b=>b.addEventListener('click',()=>setMode(b.dataset.vesselMode)));
 get('vessel-opacity').addEventListener('input',e=>{alpha=Number(e.target.value)/100;get('vessel-alpha-label').textContent=e.target.value+'%';setMode(mode)});
 get('ext-save-vessel').addEventListener('click',()=>{const g=gallery[current];const a=document.createElement('a');a.download='eyeqc_'+g.id+'_vessel_contrast.webp';a.href=asset('assets/vessels/'+g.id+'_contrast.webp');a.click()});
 get('ext-download-report').addEventListener('click',downloadJSON);
 get('local-upload').addEventListener('change',e=>{const file=e.target.files?.[0];if(!file)return;if(uploadPreviewUrl)URL.revokeObjectURL(uploadPreviewUrl);uploadPreviewUrl=URL.createObjectURL(file);get('ext-photo').src=uploadPreviewUrl;get('ext-case-name').textContent=file.name;get('ext-vessel').style.display='none';document.querySelectorAll('[data-vessel-mode]').forEach(b=>b.disabled=true);get('ext-save-vessel').disabled=true;get('ext-download-report').disabled=true;get('vessel-source').textContent='UPLOADED PHOTO · MODEL TOOLS AVAILABLE';mode='original';get('ext-photo').style.display='block';get('ext-vessel-result').textContent='For this upload, use Run vessel model with the Python service. The gallery-only contrast view is not computed in the browser.';});
 get('ext-run-vessel').addEventListener('click',()=>callModel('vessel'));
 get('ext-run-flair').addEventListener('click',()=>callModel('flair'));
 get('ext-connect').addEventListener('click',connect);
 // Precomputed contrast gallery updates when the selected photograph changes.
 document.getElementById('sample-strip')?.addEventListener('click',()=>setTimeout(showCase,0));
 autoConnect();
}
window.addEventListener('eyeqc:ready',init);if(typeof gallery!=='undefined'&&gallery.length)init();
})();
