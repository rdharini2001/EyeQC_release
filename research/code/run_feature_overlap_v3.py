"""DINOv2 embedding vs. hand-crafted feature overlap.

The Methods section already asserts a hybrid-design justification (the
embedding-only variant lost sensitivity to certain synthetic defects that
the ten interpretable axes restore) but never quantifies how much of each
hand-crafted axis's information is already linearly recoverable from the
frozen DINOv2 embedding alone. This fills that gap directly:

1. Per-axis held-out R^2: fit a ridge regression predicting each of the ten
   hand-crafted features from the 768-dim embedding alone (fit on a train
   split, evaluate on a held-out split, grouped by participant/cluster to
   avoid leakage). High R^2 = that axis is largely redundant with what the
   embedding already encodes; low R^2 = that axis contributes information
   the embedding does not capture well -- the direct quantitative evidence
   for why concatenation (not embedding-only) is the right design.
2. Canonical correlation analysis (CCA) between the 10-dim hand-crafted
   feature matrix and a PCA-reduced embedding (reduced because CCA with a
   10-vs-768 dimensional pair is degenerate; reducing the embedding to its
   top 10 PCs, fit on train only, keeps the comparison well-posed) --
   summarizes overall shared variance in one aggregate statistic alongside
   the per-axis breakdown.

Sample: a stratified sample across all five validation cohorts (not just
development), since the question is about the general relationship between
the embedding and the hand-crafted axes, not a development-set artifact.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from eyeqc_v2.backbone import embed_images  # noqa: E402
from eyeqc.core import FEATURES, DISPLAY_NAMES  # noqa: E402

TABLES = REPO_ROOT / "data_final" / "tables"

COHORTS = ["EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD"]
N_PER_COHORT = 500
SEED = 20261011
N_FOLDS = 5


def sample_cohort(cohort: str, rng: np.random.Generator) -> pd.DataFrame:
    frame = pd.read_parquet(TABLES / f"image_level_v3_{cohort}.parquet")
    frame = frame[frame["image_path"].map(lambda p: Path(str(p)).exists())].copy()
    n = min(N_PER_COHORT, len(frame))
    idx = rng.choice(frame.index.to_numpy(), size=n, replace=False)
    sampled = frame.loc[idx].copy()
    sampled["cohort"] = cohort
    group_col = "cluster_id" if "cluster_id" in sampled.columns else "image_id"
    sampled["_group"] = cohort + "_" + sampled[group_col].astype(str)
    return sampled


def main() -> None:
    rng = np.random.default_rng(SEED)
    frames = [sample_cohort(c, rng) for c in COHORTS]
    data = pd.concat(frames, ignore_index=True)
    print(f"sampled {len(data)} images across {len(COHORTS)} cohorts")

    paths = data["image_path"].tolist()
    print(f"embedding {len(paths)} images...")
    embeddings, ok = embed_images(paths, batch_size=96)
    data = data[ok].reset_index(drop=True)
    embeddings = embeddings[ok]
    print(f"{len(data)} images successfully embedded")

    hand = data[list(FEATURES)].to_numpy(float)
    groups = data["_group"].to_numpy()

    gkf = GroupKFold(n_splits=N_FOLDS)
    per_axis_r2 = {name: [] for name in FEATURES}
    for train_idx, test_idx in gkf.split(embeddings, hand, groups):
        x_train, x_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = hand[train_idx], hand[test_idx]
        for j, name in enumerate(FEATURES):
            model = Ridge(alpha=10.0)
            model.fit(x_train, y_train[:, j])
            pred = model.predict(x_test)
            per_axis_r2[name].append(r2_score(y_test[:, j], pred))

    r2_summary = pd.DataFrame(
        {
            "feature": list(FEATURES),
            "display_name": [DISPLAY_NAMES[f] for f in FEATURES],
            "mean_held_out_r2": [np.mean(per_axis_r2[f]) for f in FEATURES],
            "std_held_out_r2": [np.std(per_axis_r2[f]) for f in FEATURES],
        }
    ).sort_values("mean_held_out_r2", ascending=False)
    print("\n=== Per-axis held-out R^2 (embedding -> hand-crafted feature) ===")
    print(r2_summary.round(3).to_string(index=False))

    # ---- CCA summary (fit on a single train/test split for simplicity, grouped) ----
    train_idx, test_idx = next(GroupKFold(n_splits=N_FOLDS).split(embeddings, hand, groups))
    x_train, x_test = embeddings[train_idx], embeddings[test_idx]
    y_train, y_test = hand[train_idx], hand[test_idx]

    pca = PCA(n_components=10, random_state=SEED)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    explained = pca.explained_variance_ratio_.sum()

    cca = CCA(n_components=10, max_iter=2000)
    cca.fit(x_train_pca, y_train)
    x_c_test, y_c_test = cca.transform(x_test_pca, y_test)
    canonical_corrs = [np.corrcoef(x_c_test[:, k], y_c_test[:, k])[0, 1] for k in range(10)]

    print(f"\n=== CCA (embedding top-10 PCs, explaining {explained*100:.1f}% of embedding variance, vs 10 hand-crafted features) ===")
    for k, corr in enumerate(canonical_corrs):
        print(f"  canonical pair {k+1}: r = {corr:.3f}")
    print(f"  mean |canonical correlation| = {np.mean(np.abs(canonical_corrs)):.3f}")
    print(f"  first canonical correlation (shared-variance headline number) = {canonical_corrs[0]:.3f}")

    r2_summary.to_csv(TABLES / "dinov2_feature_overlap_r2_v3.csv", index=False)
    cca_summary = pd.DataFrame({"canonical_pair": range(1, 11), "correlation": canonical_corrs})
    cca_summary.to_csv(TABLES / "dinov2_feature_overlap_cca_v3.csv", index=False)
    print(f"\nwrote {TABLES / 'dinov2_feature_overlap_r2_v3.csv'}")
    print(f"wrote {TABLES / 'dinov2_feature_overlap_cca_v3.csv'}")


if __name__ == "__main__":
    main()
