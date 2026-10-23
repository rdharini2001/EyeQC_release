"""Exact EyeQC-v3 linear/calibration head; DINOv2 preprocessing adapter is under parity review.

This is deliberately split into an independently testable mathematical head
and a separate frozen-encoder adapter. The model contract is authoritative.
Do not claim validated end-to-end equivalence until the source preprocessing
and reference image embeddings have been checked against the paper pipeline.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
from eyeqc.core import FEATURES

class RuntimeErrorNotReady(RuntimeError):
    pass

class EyeQCValidatedHead:
    def __init__(self, model_path=None):
        p=Path(model_path) if model_path else Path(__file__).resolve().parents[1]/'models'/'eyeqc_v3_model.json'
        self.contract=json.loads(p.read_text())
        if tuple(self.contract['hand_features'])!=FEATURES or int(self.contract['input_dim'])!=778:
            raise ValueError('The supplied model contract is incompatible with this extractor.')
        self.mean=np.asarray(self.contract['standardizer_mean'],float)
        self.scale=np.asarray(self.contract['standardizer_scale'],float)
        self.coef=np.asarray(self.contract['logistic_coefficients_standardized'],float)
        if (self.mean.shape,self.scale.shape,self.coef.shape)!=((778,),)*3:
            raise ValueError('778D model contract malformed')
        if not np.all(np.isfinite(self.scale)) or np.any(self.scale<=0):
            raise ValueError('invalid standardizer scales')
        self.pass_threshold=float(self.contract['pass_risk_threshold'])
        self.fail_threshold=float(self.contract['fail_risk_threshold'])
        self.path=p

    def score_vector(self, hand_features, dinov2_embedding):
        hand=np.asarray([hand_features[k] for k in FEATURES],dtype=float)
        emb=np.asarray(dinov2_embedding,dtype=float).ravel()
        if hand.shape!=(10,) or emb.shape!=(768,) or not np.isfinite(hand).all() or not np.isfinite(emb).all():
            raise ValueError('Exactly ten finite measurements and 768 finite embedding values required')
        x=np.concatenate((hand,emb))
        logit=float(self.contract['logistic_intercept']+np.dot((x-self.mean)/self.scale,self.coef))
        calibrated=float(self.contract['platt_intercept']+self.contract['platt_coefficient']*logit)
        risk=float(1/(1+np.exp(-np.clip(calibrated,-700,700))))
        verdict='PASS' if risk < self.pass_threshold else ('FAIL' if risk>self.fail_threshold else 'REVIEW')
        return {'risk_bad':risk,'verdict':verdict,'quality_score':1-risk,'model_version':self.contract['version']}

class DINOv2Encoder:
    """Candidate encoder adapter. Explicitly NOT yet certified against study image embeddings.

    The archived code references eyeqc_v2.backbone, which was absent from the
    supplied code ZIP. Verify resolution, cropping and normalization against
    the original module before claiming numerical equivalence.
    """
    def __init__(self,device='cpu'):
        import torch
        from torchvision import transforms
        self.torch=torch;self.device=device
        self.model=torch.hub.load('facebookresearch/dinov2','dinov2_vitb14',pretrained=True,trust_repo=True).to(device).eval()
        self.preprocess=transforms.Compose([
            transforms.Resize((518,518), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)),
        ])

    def encode(self,rgb):
        from PIL import Image
        x=self.preprocess(Image.fromarray(rgb.astype(np.uint8))).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            emb=self.model(x)
        if isinstance(emb,dict):emb=emb['x_norm_clstoken']
        result=emb.detach().cpu().numpy().ravel().astype(float)
        if result.shape!=(768,):raise ValueError(f'Unexpected embedding shape {result.shape}')
        return result
