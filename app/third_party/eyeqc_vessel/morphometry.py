"""Resolution-explicit retinal vascular morphometry from *real* RRWNet probability maps.

Reference definitions: vessel area density, skeleton length density, centerline
arc/chord distance tortuosity, Euclidean-distance-transform width, skeleton
box-counting fractal dimension. These are retinal-field image measurements,
not calibrated micrometers or CRAE/CRVE. Quantities depend on segmentation
threshold, model field and field mask, and should not be cross-camera pooled
without an acquisition/scale protocol.  No pathology diagnosis is attempted.
"""
from __future__ import annotations
import math
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

NEIGHBORS=((dy,dx) for dy in (-1,0,1) for dx in (-1,0,1) if (dy,dx)!=(0,0))
NEIGHBORS=tuple(NEIGHBORS)

def clean_binary(prob,roi, threshold=.5,min_component=12):
    arr=np.asarray(prob,float)
    mask=np.asarray(roi)>0.5
    if arr.shape!=mask.shape: raise ValueError('Probability map and ROI shapes differ')
    if not (0<threshold<1):raise ValueError('Threshold must be within (0,1)')
    binary=(arr>=threshold)&mask
    if binary.any() and min_component>1:
        labels,n=ndi.label(binary,structure=np.ones((3,3),dtype=np.uint8))
        counts=np.bincount(labels.ravel())
        keep=counts>=min_component
        keep[0]=False
        binary=keep[labels]
    return binary.astype(bool)

def _neighbors(point,points):
    y,x=point
    return [(y+dy,x+dx) for dy,dx in NEIGHBORS if (y+dy,x+dx) in points]

def _arc(a,b):
    return math.hypot(a[0]-b[0],a[1]-b[1])

def trace_skeleton_segments(skeleton, min_segment_length=6.0):
    """Trace degree-2 centerline paths between graph nodes, excluding junction pixels.

    Eight-connected skeleton pixels create diagonal contacts near junctions;
    branch-pixel neighborhoods are condensed by leaving paths at first nodes.
    Closed loops (no endpoints) are excluded from tortuosity, but still
    included in total skeleton length density.
    """
    pts=set(map(tuple,np.argwhere(np.asarray(skeleton,dtype=bool)).tolist()))
    if not pts:return [],0,0
    neigh={p:_neighbors(p,pts) for p in pts}
    nodes={p for p,v in neigh.items() if len(v)!=2}
    seen=set();segments=[]
    for start in sorted(nodes):
        for nxt in neigh[start]:
            edge=frozenset((start,nxt))
            if edge in seen:continue
            seen.add(edge); path=[start,nxt]; prev,cur=start,nxt
            while cur not in nodes:
                opts=[p for p in neigh[cur] if p!=prev]
                if not opts:break
                future=opts[0];e=frozenset((cur,future))
                if e in seen:break
                seen.add(e);path.append(future);prev,cur=cur,future
            arc=float(sum(_arc(a,b) for a,b in zip(path,path[1:])))
            chord=_arc(path[0],path[-1])
            if arc>=min_segment_length and chord>=3 and path[0]!=path[-1]:
                segments.append({'arc_px':arc,'chord_px':chord,'tortuosity':max(1.,arc/chord),'points':path})
    ends=sum(len(v)==1 for v in neigh.values())
    branches=sum(len(v)>=3 for v in neigh.values())
    return segments,ends,branches

def boxcount_fd(skeleton,roi,min_boxes=4):
    arr=np.asarray(skeleton,dtype=bool)&(np.asarray(roi)>0.5)
    n=min(arr.shape)
    sizes=[2**k for k in range(1,int(math.log2(n))+1) if 2**k<=n//2]
    x=[];y=[]
    for size in sizes:
        h,w=arr.shape
        if size>min(h,w)//2:continue
        ph=(-h)%size;pw=(-w)%size
        grid=np.pad(arr,((0,ph),(0,pw)),constant_values=False)
        blocks=grid.reshape(grid.shape[0]//size,size,grid.shape[1]//size,size).any(axis=(1,3))
        count=int(blocks.sum())
        if count>0 and count<blocks.size:
            x.append(math.log(1.0/size));y.append(math.log(count))
    if len(x)<min_boxes:return float('nan')
    return float(np.clip(np.polyfit(x,y,1)[0],0,2))

def measure_morphometry(vessel_probability, roi, *, artery_probability=None,vein_probability=None,
                        threshold=.5,min_component=12,min_segment_length=6.):
    """Measure vessel geometry in the model's 576x576 canvas (pixel units).

    Length is a count of skeleton centerline pixels / retained ROI pixels;
    arc estimates use Euclidean 8-neighbor segment distances for tortuosity.
    Width is 2x Euclidean distance transform sampled at centerline pixels,
    in *model canvas pixels*, not physical caliber or CRAE/CRVE.
    """
    mask=np.asarray(roi)>0.5
    if mask.sum()<64:raise ValueError('Insufficient retinal field area for morphometry')
    binary=clean_binary(vessel_probability,mask,threshold,min_component)
    skeleton=skeletonize(binary)
    segments,ends,branches=trace_skeleton_segments(skeleton,min_segment_length)
    edt=ndi.distance_transform_edt(binary)
    width=(2*edt[skeleton]).astype(float)
    nroi=float(mask.sum())
    r={'Retinal ROI (px²)':int(nroi),'Vessel area density (%)':float(binary.sum()/nroi*100),
       'Centerline length density (px/1000 ROI px²)':float(skeleton.sum()/nroi*1000),
       'Skeleton length (px)':int(skeleton.sum()),
       'Mean centerline diameter (canvas px)':float(np.mean(width)) if width.size else float('nan'),
       'Median centerline diameter (canvas px)':float(np.median(width)) if width.size else float('nan'),
       '90th percentile diameter (canvas px)':float(np.percentile(width,90)) if width.size else float('nan'),
       'Distance tortuosity (arc/chord)':float(np.average([s['tortuosity'] for s in segments],weights=[s['arc_px'] for s in segments])) if segments else float('nan'),
       'Traceable vessel segments':len(segments),'Branch-point pixels (/1000 ROI px²)':float(branches/nroi*1000),
       'Endpoint pixels (/1000 ROI px²)':float(ends/nroi*1000),
       'Skeleton box-counting fractal dimension':boxcount_fd(skeleton,mask)}
    for key,prob in (('Artery',artery_probability),('Vein',vein_probability)):
        if prob is not None:
            sub=clean_binary(prob,mask,threshold,min_component)
            r[f'{key} area density (%)']=float(sub.sum()/nroi*100)
            subsk=skeletonize(sub)
            r[f'{key} centerline density (px/1000 ROI px²)']=float(subsk.sum()/nroi*1000)
    aux={'binary':binary,'skeleton':skeleton,'edt':edt,'segments':segments,'roi':mask}
    return r,aux
