"""Rebuild a static, privacy-conscious visual gallery from user-provided, publishable photos.
Requires: numpy, pillow, opencv-python-headless, scipy. Source EyeQC feature
functions and degradation parameters are imported unchanged. Images must be
redistribution-cleared before committing them publicly.
Run from the repository root: python scripts/build_assets.py --images /path/to/approved/fundus
"""
from pathlib import Path
import sys, argparse, json, base64
import numpy as np
import cv2
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from eyeqc.core import extract_features,FEATURES
from eyeqc.experiments import apply_degradation,DEGRADATIONS
from portal.spatial import make_layers

def save_img(rgb,path,size=450):
 img=Image.fromarray(rgb.astype('uint8'));img.thumbnail((size,size));img.save(path,format='WEBP',quality=82,method=3)

def save_data_uri(uri,path):path.write_bytes(base64.b64decode(uri.split(',',1)[1]))

def shrink(rgb,side):
 h,w=rgb.shape[:2];scale=min(1.,side/max(h,w));return cv2.resize(rgb,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA) if scale<1 else rgb

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--images',type=Path,required=True);args=parser.parse_args()
 image_dir=args.images
 files=sorted([*image_dir.glob('*.jpeg'),*image_dir.glob('*.jpg')])
 if not files:files=sorted(image_dir.glob('*.png'))  # fallback for a dedicated photo-only PNG directory
 if not files:raise RuntimeError('No input photographs in '+str(image_dir))
 docs=ROOT/'docs/assets';gallery=[]
 for p in files:
  key=p.stem;im=shrink(np.array(Image.open(p).convert('RGB')),540)
  f=extract_features(im);layers=make_layers(f['analysis_rgb'],f)
  save_img(im,docs/'images'/f'{key}.webp',size=540)
  save_data_uri(layers['map_png'],docs/'maps'/f'{key}_map.png')
  save_data_uri(layers['overlay_png'],docs/'maps'/f'{key}_overlay.png')
  gallery.append({'id':key,'label':key.replace('_',' · '),'image':f'assets/images/{key}.webp','map':f'assets/maps/{key}_map.png','overlay':f'assets/maps/{key}_overlay.png','features':{k:round(float(f[k]),4) for k in FEATURES},'grid':layers['quality_grid'],'mask_fraction':round(float(f['retina_fraction']),4),'retinal_tiles':len(f['tiles'])})
  print('Gallery',key,flush=True)
 (docs/'data/gallery.json').write_text(json.dumps(gallery,separators=(',',':')))
 data={};dose_keys=[p.stem for p in files][:3]
 for key in dose_keys:
  p=next(p for p in files if p.stem==key);im=shrink(np.array(Image.open(p).convert('RGB')),310);data[key]={}
  for kind in DEGRADATIONS:
   points=[]
   for j,d in enumerate([0,.25,.5,.75,1.]):
    altered,mask=apply_degradation(im,kind,d,np.random.default_rng(20260822+j))
    f=extract_features(altered);layers=make_layers(f['analysis_rgb'],f);stem=f'{key}_{kind}_{j}'
    save_img(altered,docs/'doses'/f'{stem}.webp',size=320)
    save_data_uri(layers['map_png'],docs/'doses'/f'{stem}_map.png')
    if mask is not None:
     marked=altered.copy();edges=cv2.Canny(mask.astype(np.uint8)*255,50,110)>0;marked[edges]=[252,190,87]
     save_img(marked,docs/'doses'/f'{stem}_region.webp',size=320)
    point={'dose':d,'image':f'assets/doses/{stem}.webp','map':f'assets/doses/{stem}_map.png','features':{k:round(float(f[k]),4) for k in FEATURES},'grid':layers['quality_grid']}
    if mask is not None:point['region']=f'assets/doses/{stem}_region.webp'
    points.append(point)
   data[key][kind]=points
   (docs/'data/doses.json').write_text(json.dumps(data,separators=(',',':')))
   print('Degradation',key,kind,flush=True)
 (docs/'data/pairs.json').write_text(json.dumps({'artifact':{'correct':390,'total':465,'pct':83.9},'clarity':{'correct':328,'total':462,'pct':71.0},'field':{'correct':327,'total':418,'pct':78.2},'composite':{'correct':587,'total':728,'pct':80.6}}))
 print('Finished',len(files),'samples and',len(dose_keys)*len(DEGRADATIONS)*5,'degradation images',flush=True)
if __name__=='__main__':main()
