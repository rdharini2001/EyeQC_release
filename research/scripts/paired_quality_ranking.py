"""Recompute EyeQC's within-eye quality ranking from private DeepDRiD pairs.

Input: data_final/tables/deepdrid_test_retest_pairs_v3.csv
Output: aggregate quality ranking, participant-clustered bootstrap CI.

Eye identifiers appear only in the private input; the script prints aggregates.
This does not test a clinical reacquisition policy or diagnostic adequacy.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

MEASURES=[('artifact',1),('clarity',-1),('field_definition',-1)]

def evaluate(df:pd.DataFrame,bootstraps:int=5000,seed:int=20261023):
    df=df.copy()
    df['participant']=df.eye_id.astype(str).str.replace(r'_[lr]$', '',regex=True)
    riskdiff=df.risk_bad_v3_2-df.risk_bad_v3_1
    bad_deltas={}
    for key,sign in MEASURES:
        x=df[f'deepdrid_{key}_1'].astype(float)
        y=df[f'deepdrid_{key}_2'].astype(float)
        values=pd.concat([x,y],ignore_index=True).dropna()
        sigma=float(values.std(ddof=0))
        bad_deltas[key]=(y-x)*sign/sigma
    bad_deltas['combined']=sum(bad_deltas.values())
    rng=np.random.default_rng(seed)
    unique=df['participant'].unique()
    groups={p:np.flatnonzero(df['participant'].to_numpy()==p) for p in unique}
    out=[]
    for measure,delta in bad_deltas.items():
        valid=np.isfinite(delta) & (delta!=0) & np.isfinite(riskdiff)
        concord=(delta*riskdiff>0).to_numpy(dtype=bool)
        n=int(valid.sum());hit=int(np.sum(concord&valid))
        samples=[]
        for _ in range(bootstraps):
            sampled=rng.choice(unique,size=len(unique),replace=True)
            sel=np.concatenate([groups[x] for x in sampled])
            keep=valid.iloc[sel].to_numpy()
            if keep.any():samples.append(float(concord[sel][keep].mean()))
        lo,hi=np.percentile(samples,[2.5,97.5]);out.append({'measure':measure,'n_informative_pairs':n,
           'correctly_ranked':hit,'concordance':hit/n,'ci_low':float(lo),'ci_high':float(hi)})
    return pd.DataFrame(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('input',help='Private paired DeepDRiD CSV')
    ap.add_argument('--out',default='paired_ranking_summary.csv');args=ap.parse_args()
    summary=evaluate(pd.read_csv(args.input));summary.to_csv(args.out,index=False)
    print(summary.to_string(index=False,float_format=lambda n:f'{n:.4f}'))
