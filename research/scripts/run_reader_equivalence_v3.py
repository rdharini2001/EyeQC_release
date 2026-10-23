"""Reader equivalence analysis for the hybrid v3 model.

Existing manuscript numbers report EyeQC-vs-reader weighted kappa and
reader-vs-reader weighted kappa separately (0.553-0.663 vs 0.774-0.798) but
never formally test whether they're statistically distinguishable. This
answers the sharper, clinically meaningful question directly: for each
reader, is EyeQC's agreement with that reader equivalent to how much that
reader agrees with the *other* two readers -- the standard "AI functions as
another reader" framing in diagnostic-AI validation literature.

Design: paired bootstrap (resample the 80 images with replacement, since all
three readers graded the same images) of
    delta_i = kappa(EyeQC, R_i) - mean_{j != i} kappa(R_i, R_j)
for each reader i. A 95% percentile CI on delta_i that falls entirely within
a pre-specified equivalence margin supports equivalence; a CI that excludes
0 and lies outside the margin would support a real difference.

Equivalence margin: +/-0.15 on the weighted-kappa scale, matching the width
of one Landis & Koch interpretation band (e.g. "moderate" 0.41-0.60 is a
~0.19-wide band; 0.15 is a slightly conservative, round choice within that
scale) -- stated explicitly here and in the manuscript footnote rather than
left implicit.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from eyeqc_v2.backbone import embed_images  # noqa: E402
from eyeqc.core import FEATURES  # noqa: E402

V1_TABLES = REPO_ROOT / "data_final" / "source_labels"
MODELS = REPO_ROOT / "data_final" / "models"
TABLES = REPO_ROOT / "data_final" / "tables"

EQUIVALENCE_MARGIN = 0.15
N_BOOTSTRAP = 10000
SEED = 20261011


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
    out["verdict"] = np.where(risk >= fail_t, "FAIL", np.where(risk <= pass_t, "PASS", "REVIEW"))
    out["verdict_ordinal"] = out["verdict"].map({"FAIL": 0, "REVIEW": 1, "PASS": 2})
    return out


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

    # Wide pivot: one row per image, one column per reader's class_ordinal,
    # plus EyeQC's own verdict_ordinal (identical for a given image across
    # reader-rows, so take the first).
    wide = reader_long.pivot_table(index="image_id", columns="reader", values="class_ordinal", aggfunc="first")
    eyeqc_col = reader_long.groupby("image_id")["verdict_ordinal"].first()
    cluster_col = reader_long.groupby("image_id")["cluster_id"].first()
    wide = wide.join(eyeqc_col).join(cluster_col)
    wide = wide.dropna()
    readers = [c for c in wide.columns if c not in ("verdict_ordinal", "cluster_id")]
    n_images = len(wide)
    n_participants = wide["cluster_id"].nunique()
    print(f"{n_images} images with complete reader+EyeQC data ({n_participants} unique participants); readers = {readers}")

    rng = np.random.default_rng(SEED)
    results = []
    for reader_i in readers:
        others = [r for r in readers if r != reader_i]

        def point_estimate(df: pd.DataFrame) -> float:
            kappa_ai = cohen_kappa_score(df[reader_i], df["verdict_ordinal"], weights="quadratic")
            kappa_peers = np.mean([cohen_kappa_score(df[reader_i], df[o], weights="quadratic") for o in others])
            return kappa_ai - kappa_peers

        observed_delta = point_estimate(wide)
        observed_kappa_ai = cohen_kappa_score(wide[reader_i], wide["verdict_ordinal"], weights="quadratic")
        observed_kappa_peers = np.mean([cohen_kappa_score(wide[reader_i], wide[o], weights="quadratic") for o in others])

        # Cluster bootstrap over participants (cluster_id), not images: one UK
        # Biobank participant contributes 2 of the 80 images, and resampling
        # at the image level would draw that participant's two images
        # independently, understating between-participant variance -- the
        # same fix applied in run_four_rater_agreement_v3.py's bootstrap_ci.
        cluster = wide["cluster_id"].astype(str).to_numpy()
        groups = [np.flatnonzero(cluster == value) for value in np.unique(cluster)]
        positions = np.arange(len(wide))
        boot_deltas = np.empty(N_BOOTSTRAP)
        for b in range(N_BOOTSTRAP):
            selected = rng.integers(0, len(groups), len(groups))
            row_idx = positions[np.concatenate([groups[s] for s in selected])]
            boot_deltas[b] = point_estimate(wide.iloc[row_idx])

        ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])
        equivalent = bool(ci_lo >= -EQUIVALENCE_MARGIN and ci_hi <= EQUIVALENCE_MARGIN)
        excludes_zero = bool(ci_lo > 0 or ci_hi < 0)

        results.append(
            {
                "reader": reader_i,
                "n_images": n_images,
                "kappa_EyeQC_vs_reader": observed_kappa_ai,
                "kappa_reader_vs_peers_mean": observed_kappa_peers,
                "delta": observed_delta,
                "delta_ci_lo": ci_lo,
                "delta_ci_hi": ci_hi,
                "equivalence_margin": EQUIVALENCE_MARGIN,
                "equivalent": equivalent,
                "ci_excludes_zero": excludes_zero,
            }
        )
        print(
            f"{reader_i}: kappa(EyeQC,{reader_i})={observed_kappa_ai:.3f}  "
            f"mean kappa({reader_i},peers)={observed_kappa_peers:.3f}  "
            f"delta={observed_delta:+.3f}  95% CI=[{ci_lo:+.3f},{ci_hi:+.3f}]  "
            f"equivalent(+/-{EQUIVALENCE_MARGIN})={equivalent}"
        )

    out = pd.DataFrame(results)
    out_path = TABLES / "reader_equivalence_v3.csv"
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
