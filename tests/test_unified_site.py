from pathlib import Path
from fastapi.testclient import TestClient
from portal.app import app

ROOT=Path(__file__).resolve().parents[1]

def test_combined_home_has_both_extra_models():
    page=TestClient(app).get('/').text
    assert 'id="extensions"' in page
    assert 'id="ext-run-vessel"' in page
    assert 'id="ext-run-flair"' in page
    assert 'EyeQC_release' in page
    assert 'Dharini24/EyeQC' not in page

def test_all_extra_assets_serve_from_same_app():
    client=TestClient(app)
    for path in ['/assets/extensions.js','/assets/extensions.css','/assets/config.json',
                 '/assets/vessels/1_left_contrast.webp','/assets/figures/eyeqc_spatial_atlas.webp']:
        assert client.get(path).status_code==200,path

def test_capability_endpoint_never_claims_trained_risk_by_default():
    d=TestClient(app).get('/api/capabilities').json()
    assert d['measure'] and d['degradation'] and d['batch']
    assert d['vessel']['source_available'] and d['flair']['source_available']
    assert not d['trained_eyeqc_risk']
