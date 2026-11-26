"""How much of each quality measurement the frozen DINOv2 embedding already encodes (all 21,091 photographs).

Ridge probes (alpha = 10) with five-fold grouped cross-validation (held-out R^2), and cross-validated canonical
correlations between ten embedding principal components and the ten measurements.
usage: python scripts/run_concept_probes.py
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from eyeqc.core import FEATURES, DISPLAY_NAMES  # noqa: E402

DATA = REPO_ROOT / "data_final"
TABLES = DATA / "tables"
CACHE = DATA / "cache"
RESULTS_TABLES = REPO_ROOT / "results" / "generated"

COHORTS = ["EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD"]
SEED = 20261011
N_FOLDS = 5

# Capped-sample (N_PER_COHORT=500, ~2,194/21,091 images) numbers already
# reported in the manuscript, for direct side-by-side comparison.
CAPPED_R2 = {
    "local_readability": 0.700,
    "color_balance": 0.685,
    "contrast": 0.674,
    "field_coverage": 0.614,
    "exposure": 0.552,
    "clipping": 0.538,
    "detail_uniformity": 0.493,
    "artifact_burden": 0.493,
    "illumination_uniformity": 0.243,
    "focus": 0.063,
}
CAPPED_CCA = [0.804, 0.666, 0.665, 0.509, 0.432, 0.361, 0.191, 0.179, 0.097, -0.054]


def load_full_cohort(cohort: str) -> pd.DataFrame:
    """Full (uncapped) hand-crafted features + cached DINOv2 embedding for
    every image in one cohort, joined on image_id."""
    frame = pd.read_parquet(TABLES / f"image_level_v3_{cohort}.parquet")
    idx = pd.read_parquet(CACHE / f"{cohort}_index.parquet")[["image_id"]].reset_index().rename(
        columns={"index": "_emb_row"}
    )
    embeddings = np.load(CACHE / f"{cohort}_dinov2_embeddings.npy")
    ok = np.load(CACHE / f"{cohort}_dinov2_ok.npy")
    idx["_ok"] = ok

    merged = frame.merge(idx, on="image_id", how="inner", validate="one_to_one")
    n_dropped = len(frame) - len(merged)
    merged = merged[merged["_ok"]].copy()
    if n_dropped:
        print(f"  [{cohort}] {n_dropped} image(s) in parquet had no matching cached embedding row (dropped)")
    merged["cohort"] = cohort
    group_col = "cluster_id" if "cluster_id" in merged.columns else "image_id"
    merged["_group"] = cohort + "_" + merged[group_col].astype(str)
    merged["_embedding"] = list(embeddings[merged["_emb_row"].to_numpy()])
    return merged


def main() -> None:
    frames = []
    for c in COHORTS:
        f = load_full_cohort(c)
        print(f"[{c}] {len(f)} images (full cohort, cached embeddings)")
        frames.append(f)
    data = pd.concat(frames, ignore_index=True)
    print(f"\npooled full-scale sample: {len(data)} images across {len(COHORTS)} cohorts (no cap)")

    embeddings = np.stack(data["_embedding"].to_numpy())
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
            "mean_held_out_r2_fullscale": [np.mean(per_axis_r2[f]) for f in FEATURES],
            "std_held_out_r2_fullscale": [np.std(per_axis_r2[f]) for f in FEATURES],
        }
    ).sort_values("mean_held_out_r2_fullscale", ascending=False).reset_index(drop=True)
    r2_summary["mean_held_out_r2_capped_n500"] = r2_summary["feature"].map(CAPPED_R2)
    r2_summary["r2_delta_fullscale_minus_capped"] = (
        r2_summary["mean_held_out_r2_fullscale"] - r2_summary["mean_held_out_r2_capped_n500"]
    )
    print(f"\n=== Per-axis held-out R^2 (embedding -> hand-crafted feature), full-scale n={len(data)} ===")
    print(r2_summary.round(3).to_string(index=False))

    # ---- CCA summary (fit on a single grouped train/test split, as original) ----
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

    print(
        f"\n=== CCA (embedding top-10 PCs, explaining {explained*100:.1f}% of embedding variance, "
        f"vs 10 hand-crafted features), full-scale n={len(data)} ==="
    )
    for k, corr in enumerate(canonical_corrs):
        print(f"  canonical pair {k+1}: r = {corr:.3f}  (capped n=500 value: {CAPPED_CCA[k]:.3f})")
    print(f"  mean |canonical correlation| = {np.mean(np.abs(canonical_corrs)):.3f}")
    print(f"  first canonical correlation (shared-variance headline number) = {canonical_corrs[0]:.3f}")

    cca_summary = pd.DataFrame(
        {
            "canonical_pair": range(1, 11),
            "correlation_fullscale": canonical_corrs,
            "correlation_capped_n500": CAPPED_CCA,
        }
    )
    cca_summary["delta_fullscale_minus_capped"] = (
        cca_summary["correlation_fullscale"] - cca_summary["correlation_capped_n500"]
    )

    # ---- combined table ----
    combined = r2_summary.reset_index(drop=False).rename(columns={"index": "r2_rank_0idx"})
    combined["r2_rank"] = combined["r2_rank_0idx"] + 1
    combined = combined.drop(columns=["r2_rank_0idx"])
    combined = combined.merge(
        cca_summary.rename(
            columns={
                "canonical_pair": "r2_rank",
                "correlation_fullscale": "canonical_corr_fullscale",
                "correlation_capped_n500": "canonical_corr_capped_n500",
                "delta_fullscale_minus_capped": "canonical_corr_delta_fullscale_minus_capped",
            }
        ),
        on="r2_rank",
        how="left",
    )
    combined["n_images_fullscale"] = len(data)

    out_main = TABLES / "feature_overlap_v3_fullscale.csv"
    out_main_results = RESULTS_TABLES / "feature_overlap_v3_fullscale.csv"
    combined.to_csv(out_main, index=False)
    combined.to_csv(out_main_results, index=False)
    print(f"\nwrote {out_main}")
    print(f"wrote {out_main_results}")

    # also keep the original two-file layout (r2-only / cca-only) for parity with the
    # original script's outputs, suffixed _fullscale
    r2_summary.to_csv(TABLES / "dinov2_feature_overlap_r2_v3_fullscale.csv", index=False)
    cca_summary.to_csv(TABLES / "dinov2_feature_overlap_cca_v3_fullscale.csv", index=False)
    r2_summary.to_csv(RESULTS_TABLES / "dinov2_feature_overlap_r2_v3_fullscale.csv", index=False)
    cca_summary.to_csv(RESULTS_TABLES / "dinov2_feature_overlap_cca_v3_fullscale.csv", index=False)


if __name__ == "__main__":
    main()
