"""The Hugging Face release is a self-contained Python + web app."""
from pathlib import Path
import json
from fastapi.testclient import TestClient
from portal.app import app
ROOT=Path(__file__).resolve().parents[1]

def test_hf_space_metadata():
    meta=(ROOT/'space/README.md').read_text()
    assert 'sdk: docker' in meta
    assert 'app_port: 7860' in meta
    assert (ROOT/'space/Dockerfile').exists()
    assert not (ROOT/'.github/workflows/pages.yml').exists()

def test_integrated_frontend_live_tools():
    client=TestClient(app)
    html=client.get('/').text
    for x in ('id="live-run"','id="batch-run"','id="ext-run-vessel"','id="ext-run-flair"'):
        assert x in html,x
    assert client.get('/assets/hf_live.js').status_code==200
    assert client.get('/api/capabilities').json()['batch']

def test_batch_outputs_feature_values():
    client=TestClient(app)
    data=(ROOT/'docs/assets/images/1_left.webp').read_bytes()
    r=client.post('/api/batch',files=[('photos',('photo.webp',data,'image/webp'))]);assert r.status_code==200,r.text
    row=r.json()['rows'][0]
    assert row['scope'] is not None
    assert len(row['measurements'])==10

def test_space_manifest_has_no_private_research_csv():
    from scripts.package_space import DIRECTORIES
    assert 'research' not in DIRECTORIES
