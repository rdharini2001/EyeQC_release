from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient
from eyeqc.core import FEATURES,extract_features
from eyeqc_runtime import EyeQCValidatedHead
from portal.app import app
from PIL import Image
import io

ROOT=Path(__file__).resolve().parents[1]

def test_contract_dimensions_and_thresholds():
    m=EyeQCValidatedHead();assert len(m.coef)==778
    assert abs(m.pass_threshold-0.3452816754758538)<1e-10
    assert abs(m.fail_threshold-0.38608837893453785)<1e-10

def test_math_head_runs_independent_of_encoder():
    m=EyeQCValidatedHead();hand={x:.5 for x in FEATURES}
    p=m.score_vector(hand,np.zeros(768));assert 0<=p['risk_bad']<=1
    assert p['verdict'] in ('PASS','REVIEW','FAIL')

def test_local_map_comes_from_image():
    im=np.asarray(Image.open(ROOT/'portal/static/samples/clear.png').convert('RGB'))
    f=extract_features(im);assert len(f['tiles'])>0
    assert all(np.isfinite(f[k]) for k in FEATURES)

def test_site_health_and_evidence():
    c=TestClient(app);assert c.get('/').status_code==200
    d=c.get('/health').json();assert d['ok']
    e=c.get('/api/evidence').json();assert e['total']==21091

def test_upload_maps_and_degradation():
    c=TestClient(app);blob=(ROOT/'portal/static/samples/clear.png').read_bytes()
    r=c.post('/api/check',files={'photo':('clear.png',blob,'image/png')});assert r.status_code==200,r.text
    d=r.json();assert len(d['measurements'])==10
    assert d['layers']['overlay_png'].startswith('data:image/png;base64,')
    assert d['risk'] is None # default fails closed, no imaginary calibrated predictions
    q=c.post('/api/degradation',files={'photo':('clear.png',blob,'image/png')},data={'kind':'darkness','dose':'0.6'})
    assert q.status_code==200,q.text;assert q.json()['measurements_after']
    curve=c.post('/api/dose-curve',files={'photo':('clear.png',blob,'image/png')},data={'kind':'darkness'})
    assert curve.status_code==200 and len(curve.json()['series'])==5

def test_reject_invalid_image():
    c=TestClient(app);r=c.post('/api/check',files={'photo':('bad.jpg',b'not an image','image/jpeg')});assert r.status_code==422
