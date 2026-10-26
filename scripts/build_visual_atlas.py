from pathlib import Path
import sys,json
import cv2
import numpy as np
from PIL import Image
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
ROOT=Path(__file__).resolve().parents[1]/'docs';AS=ROOT/'assets';
GAL=json.load(open(AS/'data/gallery.json'));D=json.load(open(AS/'data/doses.json'))
VD=AS/'vessels';VD.mkdir(exist_ok=True);FD=AS/'figures';FD.mkdir(exist_ok=True)
cm=LinearSegmentedColormap.from_list('eyeqc',['#c45358','#ef9e65','#f4d681','#85bea4','#2fa295'])
for item in GAL:
    rgb=np.array(Image.open(ROOT/item['image']).convert('RGB'))
    h,w=rgb.shape[:2]
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    tissue=(np.max(rgb,axis=2)>28)&(gray>12)
    green=rgb[:,:,1]
    clahe=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(green)
    # Image-derived, not RRWNet: multi-scale black-hat filtering for dark retinal vessel contrast.
    responses=[]
    for k in [9,15,23]:
        t=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
        responses.append(cv2.morphologyEx(clahe,cv2.MORPH_BLACKHAT,t).astype(np.float32))
    v=np.maximum.reduce(responses);q=np.quantile(v[tissue],[.2,.97]) if tissue.any() else [0,1]
    strength=np.clip((v-q[0])/(max(1,q[1]-q[0])),0,1)**1.4
    strength=cv2.GaussianBlur(strength,(3,3),0)
    strength[~tissue]=0
    out=np.zeros((h,w,4),np.uint8);out[:,:,:3]=np.array([49,228,183]);out[:,:,3]=(strength*242).astype(np.uint8)
    Image.fromarray(out,'RGBA').save(VD/(item['id']+'_contrast.webp'),quality=90,method=5)

def prepare(i):
    g=GAL[i];rgb=np.asarray(Image.open(ROOT/g['image']).convert('RGB'))
    rgb=cv2.resize(rgb,(420,420),interpolation=cv2.INTER_AREA)
    mask=(rgb.max(axis=2)>30)&(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)>14)
    data=np.array(g['grid']);
    # fixed [0,1] use for all images, no per-image re-normalization
    interp=cv2.resize(data.astype('float32'),(420,420),interpolation=cv2.INTER_CUBIC).clip(0,1)
    return rgb,mask,data,interp

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.facecolor':'#081119','axes.facecolor':'#081119',
 'text.color':'#e8f2ee','axes.labelcolor':'#e8f2ee','xtick.color':'#a9babd','ytick.color':'#a9babd'})
indices=[1,3,5,6,8,10]
fig,axs=plt.subplots(6,4,figsize=(13.8,19.6),facecolor='#081119',gridspec_kw={'wspace':.045,'hspace':.08})
for j,t in enumerate(['Fundus photograph','Measured 6 × 6 tiles','Interpolated quality map','Local quality overlay']):axs[0,j].set_title(t,fontsize=12,fontweight='bold',pad=15)
for row,i in enumerate(indices):
    g=GAL[i];rgb,mask,tiles,v=prepare(i)
    heat=cm(v)[...,:3];heat[~mask]=0
    overlay=rgb.astype(float)*.45 + heat*255*.55;overlay[~mask]=0
    figs=[rgb,np.full_like(rgb,0),heat,overlay.astype('uint8')]
    figs[1]=np.array(Image.fromarray(np.uint8(cm(cv2.resize(tiles.astype('float32'),(420,420),interpolation=cv2.INTER_NEAREST))[...,:3]*255)))
    figs[1][~mask]=0
    for j in range(4):
        ax=axs[row,j];ax.imshow(figs[j],interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
        for s in ax.spines.values():s.set_visible(False)
        if j in (2,3):
            try:ax.contour(v,levels=[.35,.65],colors=['#f4f4e7','#062a31'],linewidths=[.75,.95],linestyles=['--','-'],alpha=.85)
            except Exception: pass
        if j==0:ax.text(.02,.96,'  '+g['id'].replace('_',' · ')+'  ',va='top',transform=ax.transAxes,color='white',fontweight='bold',fontsize=10,bbox={'facecolor':'#081119','edgecolor':'none','alpha':.8,'pad':4})
        if j==3:ax.text(.97,.035,f'READABILITY  {g["features"]["local_readability"]:.2f}',ha='right',va='bottom',transform=ax.transAxes,color='white',fontsize=9,fontweight='bold',bbox={'facecolor':'#081119','alpha':.85,'pad':3,'edgecolor':'none'})
fig.subplots_adjust(top=.90,bottom=.055,left=.025,right=.94)
fig.text(.03,.972,'EyeQC  /  Spatial quality atlas',fontsize=23,fontweight='bold')
fig.text(.03,.941,'Local readability measured on six-by-six retinal tiles. A fixed colour scale is used throughout.',fontsize=10,color='#9cbeb9')
cax=fig.add_axes([.955,.18,.014,.64]);cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap=cm),cax=cax);cb.ax.tick_params(labelsize=9);cb.outline.set_edgecolor('#69958d');cb.set_label('Measured readability  ·  lower  →  higher',color='#dbe8e6',labelpad=10)
fig.text(.035,.022,'Displayed contours are visual reference levels (0.35 and 0.65), not clinical PASS/REVIEW/FAIL thresholds. The maps do not represent DINOv2 attribution.',fontsize=9,color='#a0b8b8')
fig.savefig(FD/'eyeqc_spatial_atlas.png',dpi=180,facecolor='#081119');fig.savefig(FD/'eyeqc_spatial_atlas.webp',dpi=140,facecolor='#081119');plt.close(fig)

# Code-generated frames and actual measured tile changes, no simulated risk head.
labels=[('shadow','1_left'),('glare','1_right'),('local_blur','12_right')]
fig,axs=plt.subplots(3,5,figsize=(17,11.7),facecolor='#081119',gridspec_kw={'wspace':.07,'hspace':.16})
for j,t in enumerate(['Original','Controlled defect · dose 1.0','Measured local readability','Change from original','Five-dose response']):axs[0,j].set_title(t,fontsize=11,fontweight='bold',pad=14)
for row,(kind,k) in enumerate(labels):
    original=np.asarray(Image.open(ROOT/next(x['image'] for x in GAL if x['id']==k)).convert('RGB'))
    frame=D[k][kind][-1]
    degraded=np.asarray(Image.open(ROOT/frame['image']).convert('RGB'))
    og=D[k][kind][0];d1=np.array(og['grid']);d2=np.array(frame['grid']);delta=d2-d1
    for j,im in enumerate([original,degraded]):
        axs[row,j].imshow(im);axs[row,j].set_xticks([]);axs[row,j].set_yticks([])
    quality=cv2.resize(d2.astype('float32'),(240,240),interpolation=cv2.INTER_CUBIC).clip(0,1)
    axs[row,2].imshow(quality,cmap=cm,vmin=0,vmax=1)
    change=cv2.resize(delta.astype('float32'),(240,240),interpolation=cv2.INTER_CUBIC)
    axs[row,3].imshow(change,cmap='RdBu',vmin=-.5,vmax=.5)
    axs[row,3].text(.97,.04,f' Δ mean {delta.mean():+.3f}',ha='right',va='bottom',transform=axs[row,3].transAxes,color='white',fontweight='bold',bbox={'facecolor':'#0d2028','alpha':.9,'pad':4})
    for j in [2,3]:axs[row,j].set_xticks([]);axs[row,j].set_yticks([])
    curves=[x['features']['local_readability'] for x in D[k][kind]]
    ax=axs[row,4];ax.plot([0,.25,.5,.75,1],curves,color='#7cdbbd',lw=2.7,marker='o',markersize=7);ax.set_ylim(0,1);ax.set_xlim(0,1);ax.set_yticks([0,.25,.5,.75,1]);ax.set_xticks([0,.25,.5,.75,1]);ax.grid(alpha=.22,color='#9cbaaf');ax.tick_params(labelsize=9);ax.set_xlabel('Applied dose',fontsize=10)
    axs[row,0].text(.04,.96,kind.replace('_',' ').upper(),transform=axs[row,0].transAxes,va='top',fontweight='bold',fontsize=12,bbox={'facecolor':'#0c1c26','alpha':.9,'pad':4})
    for col in range(5):
        for s in axs[row,col].spines.values():s.set_color('#2f4d55')
fig.subplots_adjust(top=.9,bottom=.09,left=.03,right=.97)
fig.text(.035,.967,'EyeQC  /  Controlled acquisition defects',fontsize=23,fontweight='bold')
fig.text(.035,.938,'Actual altered fundus photographs and feature measurements recomputed using EyeQC research code',fontsize=10,color='#a9c5c0')
fig.text(.035,.035,'Displayed dose responses reflect the local-readability measurement, not calibrated poor-quality probabilities. Colour maps share fixed bounds.',fontsize=9,color='#a0b8b8')
fig.savefig(FD/'eyeqc_degradation_atlas.png',dpi=180,facecolor='#081119');fig.savefig(FD/'eyeqc_degradation_atlas.webp',dpi=130,facecolor='#081119');plt.close(fig)
print('wrote',len(list(VD.glob('*.webp'))),'contrast overlays and 2 atlases')
