"""Research web server; file uploads remain in request memory.

The 778-feature trained risk head is only activated by an explicit env setting.
The distributed source omitted eyeqc_v2/backbone.py; without parity validation
we must not present candidate DINOv2 preprocessing as validated clinical scoring.
"""
from __future__ import annotations
import io,os,sys,time,json,base64,logging
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from fastapi import FastAPI,UploadFile,File,Form,HTTPException
from fastapi.responses import FileResponse,HTMLResponse,Response
from fastapi.staticfiles import StaticFiles
from eyeqc.core import FEATURES,DISPLAY_NAMES,ACTION_RULES,extract_features,scope_check
from eyeqc.experiments import apply_degradation,DEGRADATIONS
from portal.spatial import make_layers
from eyeqc_runtime import EyeQCValidatedHead,DINOv2Encoder

ROOT=Path(__file__).resolve().parents[1]
app=FastAPI(title='EyeQC research workspace',version='1.0',description='Research-use retinal image quality analysis')
app.mount('/static',StaticFiles(directory=ROOT/'portal'/'static'),name='static')
_HEAD=EyeQCValidatedHead()
_ENCODER=None
_MAX_FILE=15*1024*1024
_MAX_PIXELS=20_000_000

def _load_rgb(data):
    if len(data)>_MAX_FILE:raise HTTPException(413,'Image exceeds 15 MB limit')
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.width*im.height>_MAX_PIXELS:raise HTTPException(413,'Image resolution exceeds supported limit')
            rgb=np.asarray(im.convert('RGB'))
    except HTTPException:raise
    except Exception:raise HTTPException(422,'Please choose a readable JPEG, PNG or TIFF image')
    return rgb

def _encoder():
    global _ENCODER
    if os.environ.get('EYEQC_ENABLE_CANDIDATE_INFERENCE','0')!='1':
        raise RuntimeError('The published encoder preprocessing has not been verified; calibrated predictions are disabled')
    if _ENCODER is None:_ENCODER=DINOv2Encoder('cpu')
    return _ENCODER

def analyze(rgb):
    start=time.perf_counter()
    features=extract_features(rgb)
    scope=scope_check(features)
    clean={k:round(float(features[k]),4) for k in FEATURES}
    layers=make_layers(np.asarray(Image.fromarray(rgb).resize((features['analysis_rgb'].shape[1],features['analysis_rgb'].shape[0]))),features)
    ranked=sorted(clean,key=clean.get)
    out={'scope':scope,'measurements':[{'key':k,'label':DISPLAY_NAMES[k],'value':v} for k,v in clean.items()],
       'needs_attention':[{'label':DISPLAY_NAMES[k],'score':clean[k],'guidance':ACTION_RULES[k]} for k in ranked[:3]],
       'layers':layers,'risk':None,'verdict':None,'model_status':'Measured feature preview; trained risk model disabled pending encoder preprocessing parity',
       'model_version':_HEAD.contract['version']}
    if scope['in_scope']:
        try:
            emb=_encoder().encode(rgb)
            pred=_HEAD.score_vector(features,emb)
            out.update(risk=round(pred['risk_bad'],6),verdict=pred['verdict'],model_status='EXPERIMENTAL end-to-end reconstruction; not certified for clinical use')
        except Exception as exc:
            out['model_error']=str(exc)[:220]
    out['elapsed_ms']=round((time.perf_counter()-start)*1000)
    return out

@app.get('/')
def home():return FileResponse(ROOT/'portal'/'index.html')

@app.get('/health')
def health():return {'ok':True,'model_version':_HEAD.contract['version'],'trained_risk_enabled':os.getenv('EYEQC_ENABLE_CANDIDATE_INFERENCE','0')=='1'}

@app.get('/api/evidence')
def evidence():return json.loads((ROOT/'portal'/'static'/'evidence.json').read_text())

@app.post('/api/check')
async def check(photo:UploadFile=File(...)):
    rgb=_load_rgb(await photo.read())
    return analyze(rgb)

@app.post('/api/degradation')
async def degradation(photo:UploadFile=File(...),kind:str=Form('darkness'),dose:float=Form(.5)):
    if kind not in DEGRADATIONS:raise HTTPException(422,'Unrecognized degradation')
    if not 0<=dose<=1:raise HTTPException(422,'Dose must be between 0 and 1')
    rgb=_load_rgb(await photo.read())
    altered,mask=apply_degradation(rgb,kind,dose,np.random.default_rng(20260822))
    before=extract_features(rgb);after=extract_features(altered)
    preview=Image.fromarray(altered)
    buf=io.BytesIO();preview.thumbnail((1000,1000));preview.save(buf,format='PNG')
    response={'kind':kind,'dose':dose,'photo_png':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode(),
        'measurements_before':{k:round(float(before[k]),3) for k in FEATURES},
        'measurements_after':{k:round(float(after[k]),3) for k in FEATURES},
        'layers':make_layers(after['analysis_rgb'],after)}
    return response

@app.post('/api/dose-curve')
async def dose_curve(photo:UploadFile=File(...),kind:str=Form('darkness')):
    if kind not in DEGRADATIONS:raise HTTPException(422,'Unrecognized degradation')
    rgb=_load_rgb(await photo.read())
    doses=[0,.25,.5,.75,1]
    rows=[]
    for i,d in enumerate(doses):
        variant,_=apply_degradation(rgb,kind,d,np.random.default_rng(20260822+i))
        f=extract_features(variant)
        rows.append({'dose':d,'values':{k:round(float(f[k]),4) for k in FEATURES}})
    return {'kind':kind,'series':rows,'axis_note':'Measured image-quality features; not calibrated EyeQC risk without verified DINOv2 encoder'}

@app.post('/api/batch')
async def batch(photos:list[UploadFile]=File(...)):
    if len(photos)>20:raise HTTPException(413,'Maximum 20 photographs per batch')
    items=[]
    for p in photos:
        try:
            img=_load_rgb(await p.read());f=extract_features(img)
            items.append({'file':p.filename,'scope':scope_check(f)['in_scope'],'measurements':{k:round(float(f[k]),4) for k in FEATURES},'error':None})
        except Exception as exc:items.append({'file':p.filename,'error':str(exc),'scope':False})
    return {'rows':items,'note':'Research feature extraction; scored risks are not produced in this batch endpoint'}

@app.post('/api/vessel')
async def vessel(photo:UploadFile=File(...)):
    rgb=_load_rgb(await photo.read())
    try:
        from eyeqc_vessel.engine import predict_vessels,summarize_vessel_features
        from eyeqc_vessel.render import vessel_overlay_figure
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        p=predict_vessels(rgb,device='cpu');scores=summarize_vessel_features(p['prob_canvas'],p['roi_canvas'])
        fig=vessel_overlay_figure(rgb,p);buf=io.BytesIO();fig.savefig(buf,format='png',dpi=140,bbox_inches='tight');plt.close(fig)
        return {'image_png':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode(),
             'features':{k:(round(float(v),4) if np.isfinite(v) else None) for k,v in scores.items()},
             'note':'Optional RRWNet vessel model; not used in published EyeQC risk score'}
    except Exception as exc:
        logging.exception('Vessel plugin failure')
        raise HTTPException(503,'Vessel weights or optional dependencies are unavailable: '+str(exc)[:120])

@app.post('/api/flair')
async def flair(photo:UploadFile=File(...)):
    rgb=_load_rgb(await photo.read())
    try:
        from eyeqc_flair.engine import benchmark_image
        return {'result':benchmark_image(rgb,device='cpu'), 'note':'FLAIR research-only comparison; prompt scores are not calibrated quality probabilities.'}
    except Exception as exc:
        logging.exception('FLAIR optional model failure')
        raise HTTPException(503,'FLAIR weights or optional dependencies are unavailable: '+str(exc)[:120])
