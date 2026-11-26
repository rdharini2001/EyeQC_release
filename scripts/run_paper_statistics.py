"""Cohort performance and paired model comparisons as reported in the paper.

Checks that the stored predictions follow from the stored embeddings and the released model, then writes, per cohort,
discrimination, calibration and fixed-threshold metrics with participant-bootstrap 95% intervals (the intervals of the
paper's cohort table), confusion counts, and paired differences between EyeQC and the DINOv2-only, AutoMorph, RETFound
and MCF-Net predictions.

Needs data_final/ (manifests, embeddings, predictions) and the comparison caches under scripts/comparisons/.
Recalibration, same-eye pairs and localisation have their own scripts.

usage: python scripts/run_paper_statistics.py [--bootstraps 1000]
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "generated" / "paper_statistics"
TABLES = ROOT / "data_final" / "tables"
CACHE = ROOT / "data_final" / "cache"
MODELS = ROOT / "models"
COMPARISONS = ROOT / "scripts" / "comparisons"
SEED = 20261024
COHORTS = ["EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD"]
MODEL = json.loads((MODELS / "eyeqc_v3_model.json").read_text())
PT, FT = MODEL["pass_risk_threshold"], MODEL["fail_risk_threshold"]


def save(df,name):
    df.to_csv(OUT/(name+".csv"),index=False)
    print(name,df.shape,flush=True)


def load(c):
    d=pd.read_parquet(TABLES/f"image_level_v3_{c}.parquet")
    assert d.image_id.is_unique and d.gt_bad.notna().all()
    assert np.isfinite(d.risk_bad_v3).all()
    return d


def score(x,m):
    z=(x-np.asarray(m["standardizer_mean"]))/np.asarray(m["standardizer_scale"])
    return expit(m["platt_intercept"]+m["platt_coefficient"]*(m["logistic_intercept"]+z@np.asarray(m["logistic_coefficients_standardized"])))


def metrics(y,p,w=None):
    y=np.asarray(y,int); p=np.asarray(p,float)
    w=np.ones(len(y)) if w is None else np.asarray(w,float)
    def frac(a,b):
        return float(w[a].sum()/w[b].sum()) if w[b].sum()>0 else np.nan
    all_=np.ones(len(y),bool); pas=p<=PT; fail=p>=FT; rev=~(pas|fail)
    ece=0.
    bins=np.minimum((p*10).astype(int),9)
    for j in range(10):
        a=bins==j
        if w[a].sum(): ece+=w[a].sum()/w.sum()*abs(np.average(p[a]-y[a],weights=w[a]))
    two=w[y==0].sum()>0 and w[y==1].sum()>0
    return dict(AUROC=roc_auc_score(y,p,sample_weight=w) if two else np.nan,
        AUPRC=average_precision_score(y,p,sample_weight=w) if two else np.nan,
        Brier=np.average((p-y)**2,weights=w),ECE=ece,
        prediction_minus_observed=np.average(p-y,weights=w),
        PASS_NPV=frac(pas&(y==0),pas),false_PASS_rate=frac(pas&(y==1),y==1),
        FAIL_sensitivity=frac(fail&(y==1),y==1),
        FAIL_specificity=frac(~fail&(y==0),y==0),
        PASS_fraction=frac(pas,all_),REVIEW_fraction=frac(rev,all_),FAIL_fraction=frac(fail,all_))


def weights(groups,rng):
    _,inv=np.unique(np.asarray(groups).astype(str),return_inverse=True)
    ng=inv.max()+1
    return np.bincount(rng.integers(0,ng,ng),minlength=ng)[inv]


def audit_and_fixed(B):
    audits=[]; rows=[]; counts=[]
    dev=pd.read_parquet(ROOT/"data_final/source_labels/development_features.parquet")
    for ci,c in enumerate(COHORTS):
        d=load(c); y=d.gt_bad.to_numpy(int); p=d.risk_bad_v3.to_numpy()
        ix=pd.read_parquet(CACHE/f"{c}_index.parquet")
        em=np.load(CACHE/f"{c}_dinov2_embeddings.npy")
        ok=np.load(CACHE/f"{c}_dinov2_ok.npy") & ix[MODEL["hand_features"]].notna().all(axis=1).to_numpy()
        ix=ix[ok].reset_index(drop=True); em=em[ok]
        assert d.image_id.astype(str).tolist()==ix.image_id.astype(str).tolist()
        calc=score(np.c_[ix[MODEL["hand_features"]].to_numpy(float),em],MODEL)
        assert np.max(np.abs(calc-p))<1e-10
        verdict=np.where(p>=FT,"FAIL",np.where(p<=PT,"PASS","REVIEW"))
        assert np.array_equal(verdict,d.verdict_v3.to_numpy())
        audits.append(dict(cohort=c,n=len(d),clusters=d.cluster_id.nunique(),poor=int(y.sum()),
            max_score_error=np.max(abs(calc-p)),legacy_score_max_difference=np.max(abs(d.risk_bad-p)),
            available_image_paths=int(d.image_path.map(lambda s:Path(s).is_file()).sum()),
            development_participant_overlap=len(set(dev.cluster_id.astype(str))&set(d.cluster_id.astype(str))) if c=="EyeQ-test" else np.nan,
            gt_label_counts=json.dumps(d.gt_label.value_counts().to_dict())))
        point=metrics(y,p); rng=np.random.default_rng(SEED+ci)
        boots=[metrics(y,p,weights(d.cluster_id,rng)) for _ in range(B)]
        row=dict(cohort=c,n=len(d),poor_n=int(y.sum()),bootstrap_unit="recorded cluster_id",**point)
        for k in point:
            vals=[b[k] for b in boots]
            row[k+"_lo"],row[k+"_hi"]=np.nanquantile(vals,[.025,.975])
        rows.append(row)
        for label in (0,1):
            for v in ("PASS","REVIEW","FAIL"):
                counts.append(dict(cohort=c,reference_poor=label,verdict=v,n=int(((y==label)&(verdict==v)).sum())))
        print("audited",c,flush=True)
    save(pd.DataFrame(audits),"audit")
    save(pd.DataFrame(rows),"fixed_operating_metrics")
    save(pd.DataFrame(counts),"confusion_counts")


def paired(B):
    rows=[]; errors=[]
    for ci,c in enumerate(COHORTS):
        d=load(c)
        comparators={"DINOv2_only":d[["image_id","risk_bad_v2"]].rename(columns={"risk_bad_v2":"comparator"})}
        a=pd.read_parquet(COMPARISONS/f"automorph/cache/{c}_softmax_bad.parquet")
        a=a[a.ok]
        comparators["AutoMorph_native_score"]=d[["image_id","image_path"]].merge(a,on="image_path",validate="one_to_one")[["image_id","softmax_bad"]].rename(columns={"softmax_bad":"comparator"})
        r=pd.read_parquet(TABLES/f"image_level_v3_retfound_{c}.parquet")
        comparators["RETFound_hybrid"]=r[["image_id","risk_bad"]].rename(columns={"risk_bad":"comparator"})
        mc=COMPARISONS/f"mcfnet/scored_{c}.parquet"
        if mc.exists():
            comparators["MCFNet_adapted"]=pd.read_parquet(mc)[["image_id","risk_bad"]].rename(columns={"risk_bad":"comparator"})
        for j,(name,cmp) in enumerate(comparators.items()):
            t=d[["image_id","cluster_id","gt_bad","risk_bad_v3"]].merge(cmp,on="image_id",validate="one_to_one")
            assert not t.isna().any().any()
            y=t.gt_bad.to_numpy(int); p=t.risk_bad_v3.to_numpy(); q=t.comparator.to_numpy()
            delta=lambda w: [roc_auc_score(y,p,sample_weight=w)-roc_auc_score(y,q,sample_weight=w),
                np.average((p-y)**2-(q-y)**2,weights=w)]
            point=delta(np.ones(len(t))); rng=np.random.default_rng(SEED+100+ci*10+j)
            bs=[]
            for b in range(B):
                w=weights(t.cluster_id,rng)
                if w[y==0].sum() and w[y==1].sum(): bs.append(delta(w))
            bs=np.asarray(bs)
            lo,hi=np.nanquantile(bs,[.025,.975],axis=0)
            rows.append(dict(cohort=c,comparator=name,paired_n=len(t),excluded_eyeqc_n=len(d)-len(t),
                AUROC_eyeqc=roc_auc_score(y,p),AUROC_comparator=roc_auc_score(y,q),
                delta_AUROC=point[0],delta_AUROC_lo=lo[0],delta_AUROC_hi=hi[0],
                delta_Brier=point[1],delta_Brier_lo=lo[1],delta_Brier_hi=hi[1],
                calibration_comparability="native score; Brier descriptive" if name=="AutoMorph_native_score" else "development-adapted model"))
        # Matched decisions at each model's own development-derived thresholds.
        m2=json.loads((MODELS/"eyeqc_v2_model.json").read_text())
        v2=d.risk_bad_v2.to_numpy()>=m2["fail_risk_threshold"]
        v3=d.risk_bad_v3.to_numpy()>=FT; y=d.gt_bad.to_numpy(int)
        for label in (0,1):
            s=y==label
            errors.append(dict(cohort=c,reference_poor=label,n=int(s.sum()),
                corrected_by_hybrid=int((s & (v2!=y)&(v3==y)).sum()),
                introduced_by_hybrid=int((s & (v2==y)&(v3!=y)).sum()),
                both_wrong=int((s & (v2!=y)&(v3!=y)).sum()),
                both_correct=int((s & (v2==y)&(v3==y)).sum())))
        print("paired",c,flush=True)
    save(pd.DataFrame(rows),"paired_model_differences")
    save(pd.DataFrame(errors),"hybrid_error_transitions")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    audit_and_fixed(args.bootstraps)
    paired(args.bootstraps)


if __name__ == "__main__":
    main()
