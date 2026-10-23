"""Check frozen 778D logistic head against original archived per-image predictions.

Requires the private `data_final` archive and `pip install pyarrow`. This test
needs no DINOv2 weight download because cached 768D embeddings are supplied.
It does NOT independently validate photo-to-embedding preprocessing.
"""
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from eyeqc_runtime import EyeQCValidatedHead

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=Path('data_final'))
 p.add_argument('--cohort',default='DRIMDB');a=p.parse_args();root=a.data_root
 model=EyeQCValidatedHead()
 idx=pd.read_parquet(root/'cache'/f'{a.cohort}_index.parquet')
 emb=np.load(root/'cache'/f'{a.cohort}_dinov2_embeddings.npy')
 ok=np.load(root/'cache'/f'{a.cohort}_dinov2_ok.npy').astype(bool)
 ok &=idx[list(model.contract['hand_features'])].notna().all(axis=1).to_numpy()
 idx=idx[ok].reset_index(drop=True);emb=emb[ok]
 scores=np.array([model.score_vector(row,vector)['risk_bad'] for row,vector in zip(idx.to_dict('records'),emb)])
 published=pd.read_parquet(root/'tables'/f'image_level_v3_{a.cohort}.parquet')
 key='risk_bad_v3' if 'risk_bad_v3' in published else 'risk_bad'
 if len(published)!=len(scores):raise ValueError('Reference score count differs; verify sorted/order and valid mask')
 err=np.abs(published[key].to_numpy(float)-scores)
 print(f'{a.cohort}: n={len(scores)}, max_abs_error={err.max():.3g}, mean_abs_error={err.mean():.3g}')
 if err.max()>1e-6:raise SystemExit('FAIL: coefficient contract does not reproduce archived risks')
 print('PASS: the 778D mathematical head reproduces cached-image scores')
