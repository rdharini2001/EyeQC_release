"""Verify all GitHub Pages content uses real, shipped files."""
from pathlib import Path
import json,re,sys
root=Path(__file__).resolve().parents[1]
docs=root/'docs'
missing=[]
for p in ['index.html','assets/site.js','assets/site.css','assets/extensions.js','assets/extensions.css','assets/figures/eyeqc_spatial_atlas.webp','assets/figures/eyeqc_degradation_atlas.webp','assets/config.json','assets/icon.svg','assets/data/gallery.json','assets/data/doses.json','assets/data/evidence.json','assets/data/pairs.json','assets/data/baselines_full.json']:
 if not (docs/p).is_file():missing.append(p)
if missing:raise SystemExit('Missing essential content: '+', '.join(missing))
gallery=json.loads((docs/'assets/data/gallery.json').read_text())
dose=json.loads((docs/'assets/data/doses.json').read_text())
evidence=json.loads((docs/'assets/data/evidence.json').read_text())
for row in gallery:
 for k in ['image','map','overlay']:
  if not (docs/row[k]).is_file():missing.append(row[k])
 if not (docs/'assets/vessels'/ (row['id']+'_contrast.webp')).is_file():missing.append(row['id']+'_contrast.webp')
 if len(row['grid'])!=6 or any(len(x)!=6 for x in row['grid']):raise SystemExit('Bad 6x6 spatial grid '+row['id'])
 if len(row['features'])!=10:raise SystemExit('Wrong feature count: '+row['id'])
for key,degs in dose.items():
 if len(degs)!=9:raise SystemExit(f'{key}: expected nine degradations, got {len(degs)}')
 for kind,series in degs.items():
  if len(series)!=5 or [x['dose'] for x in series]!=[0,.25,.5,.75,1.]:raise SystemExit(f'Invalid doses for {key}/{kind}')
  for point in series:
   for k in ['image','map','region']:
    if k in point and not (docs/point[k]).is_file():missing.append(point[k])
if missing:raise SystemExit('Missing referenced files:\n'+'\n'.join(sorted(set(missing))[:40]))
print(f'OK: {len(gallery)} real photographic samples, {len(dose)*9*5} code-generated degradation frames, {len(evidence["cohorts"])} validation cohorts; all assets resolve.')
