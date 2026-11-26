"""Agreement of all raters together, with and without EyeQC as a fourth rater: Fleiss' kappa and ICC(3,1), ICC(3,k),
with participant-bootstrap intervals for kappa.

Needs the reader-study data, which come from UK Biobank (restricted access) and are not distributed.

usage: python scripts/run_reader_group_agreement.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
from eyeqc.agreement import fleiss_kappa  # noqa: E402
from eyeqc.backbone import embed_images  # noqa: E402
from eyeqc.core import FEATURES  # noqa: E402

V1_TABLES = REPO_ROOT / "data_final" / "source_labels"
MODELS = REPO_ROOT / "models"
TABLES = REPO_ROOT / "data_final" / "tables"
RESULTS_TABLES = REPO_ROOT / "results" / "generated"


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def score_v3(frame: pd.DataFrame, embeddings: np.ndarray, contract: dict) -> pd.DataFrame:
    mean = np.asarray(contract["standardizer_mean"])
    scale = np.asarray(contract["standardizer_scale"])
    coef = np.asarray(contract["logistic_coefficients_standardized"])
    intercept = float(contract["logistic_intercept"])
    hand = frame[list(FEATURES)].to_numpy(float)
    x = np.concatenate([hand, embeddings], axis=1)
    z = (x - mean) / scale
    raw_logit = intercept + z @ coef
    calibrated_logit = contract["platt_intercept"] + contract["platt_coefficient"] * raw_logit
    risk = _sigmoid(calibrated_logit)
    pass_t, fail_t = contract["pass_risk_threshold"], contract["fail_risk_threshold"]
    out = frame.copy()
    out["risk_bad"] = risk
    out["verdict_ordinal"] = np.where(risk >= fail_t, 0, np.where(risk <= pass_t, 2, 1))  # FAIL=0, REVIEW=1, PASS=2
    return out


def fleiss_kappa_for(wide: pd.DataFrame, raters: list[str]) -> float:
    """wide: one row per image, one column per rater, values in {0,1,2}."""
    return fleiss_kappa(wide[raters].to_numpy(int))


def icc_3_anova(wide: pd.DataFrame, raters: list[str]) -> dict:
    """Two-way mixed-effects ANOVA (subjects random, raters fixed) ->
    ICC(3,1) and ICC(3,k), Shrout & Fleiss 1979 / McGraw & Wong 1996.
    Returns the full ANOVA table alongside the derived ICCs."""
    x = wide[raters].to_numpy(float)
    n, k = x.shape  # n subjects (images), k raters

    grand_mean = x.mean()
    subject_means = x.mean(axis=1)
    rater_means = x.mean(axis=0)

    ss_total = float(((x - grand_mean) ** 2).sum())
    ss_subjects = float(k * ((subject_means - grand_mean) ** 2).sum())
    ss_raters = float(n * ((rater_means - grand_mean) ** 2).sum())
    ss_error = ss_total - ss_subjects - ss_raters

    df_subjects = n - 1
    df_raters = k - 1
    df_error = (n - 1) * (k - 1)

    ms_subjects = ss_subjects / df_subjects
    ms_raters = ss_raters / df_raters
    ms_error = ss_error / df_error

    icc_3_1 = (ms_subjects - ms_error) / (ms_subjects + (k - 1) * ms_error)
    icc_3_k = (ms_subjects - ms_error) / ms_subjects

    f_stat_raters = ms_raters / ms_error
    from scipy.stats import f as f_dist

    p_raters = float(1 - f_dist.cdf(f_stat_raters, df_raters, df_error))

    return {
        "n_subjects": n,
        "n_raters": k,
        "ss_subjects": ss_subjects,
        "ss_raters": ss_raters,
        "ss_error": ss_error,
        "df_subjects": df_subjects,
        "df_raters": df_raters,
        "df_error": df_error,
        "ms_subjects": ms_subjects,
        "ms_raters": ms_raters,
        "ms_error": ms_error,
        "F_rater_effect": float(f_stat_raters),
        "p_rater_effect": p_raters,
        "ICC_3_1": float(icc_3_1),
        "ICC_3_k": float(icc_3_k),
    }


def bootstrap_ci(wide: pd.DataFrame, raters: list[str], stat_fn, n_resamples: int = 2000, seed: int = 20261012) -> tuple[float, float]:
    """Cluster bootstrap over participants (``cluster_id``), not images, because a participant can contribute more than
    one photograph; every photograph of a drawn participant is included.  Falls back to image resampling without
    ``cluster_id``.
    """
    rng = np.random.default_rng(seed)
    if "cluster_id" in wide.columns:
        cluster = wide["cluster_id"].astype(str).to_numpy()
        groups = [np.flatnonzero(cluster == value) for value in np.unique(cluster)]
        rows = wide[raters]
        values = np.empty(n_resamples)
        for b in range(n_resamples):
            selected = rng.integers(0, len(groups), len(groups))
            row_idx = np.concatenate([groups[s] for s in selected])
            values[b] = stat_fn(rows.iloc[row_idx], raters)
        return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

    idx = wide.index.to_numpy()
    values = np.empty(n_resamples)
    for b in range(n_resamples):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        values[b] = stat_fn(wide.loc[sample_idx], raters)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main() -> None:
    contract = json.loads((MODELS / "eyeqc_v3_model.json").read_text(encoding="utf-8"))

    reader_long = pd.read_parquet(V1_TABLES / "reader_case_level_exact_links.parquet")
    unique_paths = reader_long["image_path"].dropna().unique().tolist()
    print(f"embedding {len(unique_paths)} unique reader-study images...")
    embeddings, ok = embed_images(unique_paths, batch_size=64)
    embed_map = {path: embeddings[i] for i, path in enumerate(unique_paths) if ok[i]}

    keep = reader_long["image_path"].map(lambda p: p in embed_map)
    reader_long = reader_long[keep].reset_index(drop=True)
    emb = np.stack([embed_map[p] for p in reader_long["image_path"]])
    reader_long = score_v3(reader_long, emb, contract)

    wide = reader_long.pivot_table(index="image_id", columns="reader", values="class_ordinal", aggfunc="first")
    eyeqc_col = reader_long.groupby("image_id")["verdict_ordinal"].first()
    cluster_col = reader_long.groupby("image_id")["cluster_id"].first()
    wide = wide.join(eyeqc_col.rename("EyeQC")).join(cluster_col).dropna()
    readers3 = ["R1", "R2", "R3"]
    readers4 = ["R1", "R2", "R3", "EyeQC"]
    n = len(wide)
    n_participants = wide["cluster_id"].nunique()
    print(f"{n} images with complete R1/R2/R3/EyeQC data ({n_participants} unique participants)")

    # ---- Fleiss' kappa: 3-reader sanity check vs. 4-rater (with EyeQC) ----
    kappa3 = fleiss_kappa_for(wide, readers3)
    kappa4 = fleiss_kappa_for(wide, readers4)
    kappa3_lo, kappa3_hi = bootstrap_ci(wide, readers3, fleiss_kappa_for)
    kappa4_lo, kappa4_hi = bootstrap_ci(wide, readers4, fleiss_kappa_for)
    print(f"\nFleiss kappa, 3 readers only: {kappa3:.4f}  95% CI [{kappa3_lo:.4f}, {kappa3_hi:.4f}]")
    print(f"Fleiss kappa, 4 raters (+EyeQC): {kappa4:.4f}  95% CI [{kappa4_lo:.4f}, {kappa4_hi:.4f}]")

    # ---- ANOVA-based ICC(3,1) / ICC(3,k) ----
    icc3 = icc_3_anova(wide, readers3)
    icc4 = icc_3_anova(wide, readers4)
    print("\n=== Two-way mixed-effects ANOVA, 3 readers only ===")
    for k, v in icc3.items():
        print(f"  {k}: {v}")
    print("\n=== Two-way mixed-effects ANOVA, 4 raters (+EyeQC) ===")
    for k, v in icc4.items():
        print(f"  {k}: {v}")

    summary = pd.DataFrame(
        [
            {"group": "R1+R2+R3", "n_images": n, "fleiss_kappa": kappa3, "fleiss_kappa_ci_lo": kappa3_lo, "fleiss_kappa_ci_hi": kappa3_hi, **icc3},
            {"group": "R1+R2+R3+EyeQC", "n_images": n, "fleiss_kappa": kappa4, "fleiss_kappa_ci_lo": kappa4_lo, "fleiss_kappa_ci_hi": kappa4_hi, **icc4},
        ]
    )

    TABLES.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    out_data = TABLES / "four_rater_agreement_v3.csv"
    out_results = RESULTS_TABLES / "four_rater_agreement_v3.csv"
    summary.to_csv(out_data, index=False)
    summary.to_csv(out_results, index=False)
    print(f"\nwrote {out_data}")
    print(f"wrote {out_results}")


if __name__ == "__main__":
    main()
