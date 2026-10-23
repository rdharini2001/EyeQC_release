"""Measured local-readability maps from EyeQC's six-by-six retinal tiles.
The map is an interpolation of measured tile quality, not pixel-level GT,
not a DINOv2 attribution map, and not a lesion segmentation.
"""
from __future__ import annotations
import base64,io
import cv2,numpy as np
from PIL import Image
from eyeqc.experiments import tile_quality_map

def _data_png(a):
    b=io.BytesIO();Image.fromarray(a.astype(np.uint8)).save(b,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()

def make_layers(rgb,features):
    g=tile_quality_map(features).astype(np.float32)
    h,w=rgb.shape[:2]
    # Interpolate for visualization only; original 6x6 scores are returned as well.
    quality=cv2.resize(g,(w,h),interpolation=cv2.INTER_CUBIC)
    quality=np.clip(cv2.GaussianBlur(quality,(0,0),sigmaX=max(1,w/180)),0,1)
    mask=cv2.resize(features['support_mask'].astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
    cmap=cv2.applyColorMap(np.clip((1-quality)*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO)
    # Turbo reversed? map high risk warm (red), low risk green; fixed transparent palette below.
    xp=np.array([0.0,.25,.5,.75,1.0]); stops=np.array([[44,151,109],[129,184,130],[239,207,102],[231,134,72],[183,62,63]],float)
    heat=np.stack([np.interp(1-quality,xp,stops[:,i]) for i in range(3)],axis=-1).astype(np.uint8)
    overlay=(.65*rgb+.35*heat).astype(np.uint8)
    overlay[~mask]=rgb[~mask]
    rgba=np.concatenate([heat,np.where(mask,235,0).astype(np.uint8)[...,None]],axis=2)
    masked=np.where(mask[...,None],rgb,0).astype(np.uint8)
    return {'quality_grid':np.round(g,4).tolist(),'mask_png':_data_png(masked),
        'map_png':_data_png(rgba),'overlay_png':_data_png(overlay),
        'grid_size':'6 × 6 measured tiles','map_definition':'interpolated local readability, not pixel-level validation'}
