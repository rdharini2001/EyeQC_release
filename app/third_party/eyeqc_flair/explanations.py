"""Gradient-based FLAIR image–text localization; NOT native attention maps.

ResNet50 FLAIR has no Transformer image-attention matrices. To avoid
misrepresenting its internals, we compute Grad-CAM on its last ResNet layer
for each user-supplied text prompt, using actual text–image similarity
logits and backpropagation. Results are qualitative and exploratory, NOT
pathology lesion masks or a validated quality/pathology disentangler.
"""
from __future__ import annotations
from pathlib import Path
import tempfile
import cv2
import numpy as np
from PIL import Image


def _normalize_map(arr, image_shape):
    if hasattr(arr,'detach'):
        arr=arr.detach().float().cpu().numpy()
    arr=np.asarray(arr,dtype=np.float32)
    if arr.ndim!=2:
        raise ValueError('Grad-CAM expected a 2-D activation map')
    arr=cv2.resize(arr, (image_shape[1],image_shape[0]),interpolation=cv2.INTER_CUBIC)
    arr=cv2.GaussianBlur(arr,(0,0),4)
    vmax=float(np.percentile(arr,99))
    if vmax>1e-9: arr=np.clip(arr/vmax,0,1)
    else: arr=np.zeros_like(arr)
    return arr


def _overlay(image, heat, weight=.48):
    color=cv2.applyColorMap(np.uint8(255*np.clip(heat,0,1)),cv2.COLORMAP_MAGMA)
    color=cv2.cvtColor(color,cv2.COLOR_BGR2RGB)
    mix=np.uint8(np.clip((1-weight)*image.astype(float)+weight*color,0,255))
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    tissue=(image.max(axis=2)>22)&(gray>7)
    mix[~tissue]=image[~tissue]
    mix[heat < 0.04]=image[heat < 0.04]
    return mix


def _with_scale(image):
    """Append a legible fixed scale for normalized within-image Grad-CAM."""
    h,w=image.shape[:2]
    side=max(50,int(round(w*.13)))
    canvas=np.full((h,w+side,3),[12,25,36],dtype=np.uint8)
    canvas[:,:w]=image
    y0=int(h*.08);y1=int(h*.90);x0=w+int(side*.20);x1=x0+max(12,int(side*.30))
    steps=np.linspace(1,0,max(2,y1-y0)).astype(np.float32)
    colored=cv2.applyColorMap(np.uint8(steps[:,None]*255),cv2.COLORMAP_MAGMA)
    colored=cv2.cvtColor(colored,cv2.COLOR_BGR2RGB)
    canvas[y0:y1,x0:x1]=colored
    font=cv2.FONT_HERSHEY_SIMPLEX
    fontsize=max(.32,min(.54,w/900))
    cv2.putText(canvas,'1.0',(x0,y0-7),font,fontsize,(250,245,223),1,cv2.LINE_AA)
    cv2.putText(canvas,'0.0',(x0,y1+18),font,fontsize,(250,245,223),1,cv2.LINE_AA)
    return canvas


def _animation(original, q, p):
    """Animated fade between two actual Grad-CAM maps, not generated images."""
    h,w=original.shape[:2]
    if max(h,w)>560:
        factor=560/max(h,w)
        shape=(max(1,int(w*factor)),max(1,int(h*factor)))
        original=cv2.resize(original,shape,interpolation=cv2.INTER_AREA)
        q=cv2.resize(q,shape,interpolation=cv2.INTER_LINEAR)
        p=cv2.resize(p,shape,interpolation=cv2.INTER_LINEAR)
    q_rgb=_overlay(original,q,.53).astype(np.float32)
    p_rgb=_overlay(original,p,.53).astype(np.float32)
    values=np.concatenate([np.linspace(0,1,9),np.linspace(1,0,9)])
    frames=[]
    for t in values:
        rgb=np.uint8(np.clip(q_rgb*(1-t)+p_rgb*t,0,255))
        label='QUALITY PROMPT' if t<0.5 else 'PATHOLOGY PROMPT'
        cv2.rectangle(rgb,(8,8),(min(rgb.shape[1]-1,197),34),(10,25,34),-1)
        cv2.putText(rgb,label,(14,26),cv2.FONT_HERSHEY_SIMPLEX,.45,(245,245,245),1,cv2.LINE_AA)
        frames.append(Image.fromarray(_with_scale(rgb)))
    dest=Path(tempfile.mkdtemp(prefix='eyeqc_flair_'))/'flair_quality_pathology_gradcam.gif'
    frames[0].save(dest,save_all=True,append_images=frames[1:],duration=145,loop=0,optimize=True)
    return str(dest)


def explain_flair_gradcam(model, rgb: np.ndarray, quality_prompt: str, pathology_prompt: str):
    """Return two prompt-conditioned *Grad-CAM* overlays and a real animation.

    The source model is `flair.FLAIRModel` ResNet50 and uses its original
    preprocessing. If the ResNet architecture changes, fail transparently.
    Called from a real `@spaces.GPU` event, never from initial render.
    """
    import torch
    resnet=getattr(getattr(model,'vision_model',None),'model',None)
    if resnet is None or not hasattr(resnet,'layer4'):
        raise RuntimeError('FLAIR vision encoder is not ResNet50; gradient localization unavailable.')
    rgb=np.asarray(rgb,dtype=np.uint8)
    prompts=[quality_prompt,pathology_prompt]
    activations={}
    def hook(module, inputs, outputs):
        activations['feature']=outputs
    handle=resnet.layer4[-1].register_forward_hook(hook)
    try:
        with torch.enable_grad():
            image_tensor=model.preprocess_image(rgb.copy())
            with torch.no_grad():
                tok,mask=model.preprocess_text(prompts)
                text_emb=model.text_model(tok,mask).detach()
            img_emb=model.vision_model(image_tensor)
            logits=model.compute_logits(img_emb,text_emb)
            act=activations['feature']
            maps=[]
            for i in range(2):
                grad=torch.autograd.grad(logits[0,i],act,retain_graph=i==0,allow_unused=False)[0]
                weights=grad.mean(dim=(2,3),keepdim=True)
                cam=torch.relu((weights*act).sum(dim=1))[0]
                maps.append(_normalize_map(cam,rgb.shape))
            sims=logits.detach().float().cpu().numpy()[0].tolist()
    finally:
        handle.remove()
    q,p=maps
    q_ov=_with_scale(_overlay(rgb,q))
    p_ov=_with_scale(_overlay(rgb,p))
    overlap=float(np.sum(np.minimum(q,p))/max(float(np.sum(np.maximum(q,p))),1e-12))
    anim=_animation(rgb,q,p)
    return {'quality_overlay':q_ov,'pathology_overlay':p_ov,'quality_map':q,
            'pathology_map':p,'animation':anim,'logits':sims,'saliency_overlap':overlap}
