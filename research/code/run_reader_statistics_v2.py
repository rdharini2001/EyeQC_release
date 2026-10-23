"""Reader-statistics v2: addresses two specific peer-review critiques on top
of the existing run_four_rater_agreement_v3.py and run_reader_equivalence_v3.py
analyses. Supersedes/extends both (hence "v2", not "v3") but reuses their
logic by import rather than copy-pasting it, so every number that overlaps
with the two existing scripts is guaranteed identical.

Critique 1 (statistical framing). A reviewer wrote: "The two-sided +/-0.15
equivalence design is the wrong frame. The question is non-inferiority, and
with n=80 the CIs (about +/-0.1) make the test underpowered." Non-inferiority
differs from two-sided equivalence: it only asks whether EyeQC is not *worse*
than reader-reader agreement by more than a margin -- it does not also
penalize EyeQC for being *better*. This script adds a proper one-sided
non-inferiority test for each reader, on top of (not instead of) the
existing two-sided result, using the exact same paired-bootstrap machinery
as run_reader_equivalence_v3.py (10,000 resamples, images resampled jointly,
same seed, same delta_i = kappa(EyeQC,R_i) - mean_j kappa(R_i,R_j)):

    old (two-sided):   95% CI = [2.5th pct, 97.5th pct] of boot_delta;
                        "equivalent" iff CI within [-0.15, +0.15].
    new (one-sided):   one-sided 95% lower bound = 5th pct of boot_delta;
                        "non-inferior" iff that lower bound > -0.15.

Both are reported side by side per reader, along with the achieved two-sided
CI width (full width and half-width) to directly confirm or refute the
reviewer's "~+/-0.1, underpowered" claim with the real number rather than an
assumption.

Critique 2 (missing statistic). A reviewer wrote: "Fleiss kappa (unweighted,
0.536) is compared informally with quadratic-weighted kappa (0.77-0.80). Use
an ordinal multi-rater statistic such as Krippendorff's alpha." This script
computes Krippendorff's alpha with ORDINAL (not nominal) distance weighting
on the same 0/1/2 PASS/REVIEW/FAIL scale, for the same two rater groups
already used in run_four_rater_agreement_v3.py -- {R1,R2,R3} and
{R1,R2,R3,EyeQC} -- with a bootstrap 95% CI (2,000 resamples, same
image-level resampling pattern as that file's own bootstrap_ci helper,
which is imported and reused directly here). This is a single, properly
ordinal multi-rater statistic that supersedes the apples-to-oranges
Fleiss-vs-weighted-kappa comparison the reviewer flagged; the previously
reported Fleiss kappa values (0.536 for 3 readers, 0.526 for 3 readers +
EyeQC; see data_final/tables/four_rater_agreement_v3.csv) are carried
through in the output table purely as a reference point for that
comparison.

Data loading and model scoring reuse score_v3 from run_reader_equivalence_v3
(current v3 model contract, development-only thresholds 0.3453/0.3861,
already fixed elsewhere -- not touched here) and bootstrap_ci from
run_four_rater_agreement_v3, imported directly rather than reimplemented.
"""
import json
import sys
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

from eyeqc_v2.backbone import embed_images  # noqa: E402

# Reuse the existing scripts' own logic by import instead of copy-pasting it.
import run_four_rater_agreement_v3 as four_rater  # noqa: E402
import run_reader_equivalence_v3 as equivalence  # noqa: E402

V1_TABLES = REPO_ROOT / "data_final" / "source_labels"
MODELS = REPO_ROOT / "data_final" / "models"
TABLES = REPO_ROOT / "data_final" / "tables"
RESULTS_TABLES = REPO_ROOT / "results_final" / "tables"

NONINFERIORITY_MARGIN = equivalence.EQUIVALENCE_MARGIN  # 0.15 -- same value as the two-sided test, now used one-sided
N_BOOTSTRAP_EQUIV = equivalence.N_BOOTSTRAP  # 10000
SEED_EQUIV = equivalence.SEED  # 20261011, identical seed -> identical boot_deltas as the two-sided-only script

# Previously reported Fleiss kappa values (data_final/tables/four_rater_agreement_v3.csv),
# carried through only as a labeled reference point for the Krippendorff-alpha table.
FLEISS_KAPPA_REFERENCE = {"R1+R2+R3": 0.5360077332044466, "R1+R2+R3+EyeQC": 0.5261909939013658}


def load_wide() -> pd.DataFrame:
    """Rebuild the same image x {R1,R2,R3,EyeQC} wide table (0=FAIL/1=REVIEW/
    2=PASS) used throughout v3 reader-agreement analyses. Identical data
    source and scoring contract to both existing scripts; score_v3 itself is
    imported (not reimplemented) from run_reader_equivalence_v3."""
    contract = json.loads((MODELS / "eyeqc_v3_model.json").read_text(encoding="utf-8"))

    reader_long = pd.read_parquet(V1_TABLES / "reader_case_level_exact_links.parquet")
    unique_paths = reader_long["image_path"].dropna().unique().tolist()
    print(f"embedding {len(unique_paths)} unique reader-study images...")
    embeddings, ok = embed_images(unique_paths, batch_size=64)
    embed_map = {path: embeddings[i] for i, path in enumerate(unique_paths) if ok[i]}

    keep = reader_long["image_path"].map(lambda p: p in embed_map)
    reader_long = reader_long[keep].reset_index(drop=True)
    emb = np.stack([embed_map[p] for p in reader_long["image_path"]])
    reader_long = equivalence.score_v3(reader_long, emb, contract)

    wide = reader_long.pivot_table(index="image_id", columns="reader", values="class_ordinal", aggfunc="first")
    eyeqc_col = reader_long.groupby("image_id")["verdict_ordinal"].first()
    cluster_col = reader_long.groupby("image_id")["cluster_id"].first()
    wide = wide.join(eyeqc_col.rename("EyeQC")).join(cluster_col).dropna()
    return wide


def equivalence_and_noninferiority(wide: pd.DataFrame, readers: list[str]) -> pd.DataFrame:
    """For each reader, run the identical paired-bootstrap delta_i machinery
    as run_reader_equivalence_v3.py, then report both the old two-sided
    equivalence verdict and the new one-sided non-inferiority verdict from
    the SAME bootstrap draws (so the two are directly comparable, not
    computed from independent resamples)."""
    rng = np.random.default_rng(SEED_EQUIV)
    n_images = len(wide)
    # Cluster bootstrap over participants (cluster_id), not images -- same
    # fix applied in run_four_rater_agreement_v3.bootstrap_ci and
    # run_reader_equivalence_v3.main: one UK Biobank participant contributes
    # 2 of the 80 images, so image-level resampling would draw that
    # participant's two images independently.
    cluster = wide["cluster_id"].astype(str).to_numpy()
    groups = [np.flatnonzero(cluster == value) for value in np.unique(cluster)]
    positions = np.arange(len(wide))
    results = []

    for reader_i in readers:
        others = [r for r in readers if r != reader_i]

        def point_estimate(df: pd.DataFrame) -> float:
            kappa_ai = cohen_kappa_score(df[reader_i], df["EyeQC"], weights="quadratic")
            kappa_peers = np.mean([cohen_kappa_score(df[reader_i], df[o], weights="quadratic") for o in others])
            return kappa_ai - kappa_peers

        observed_delta = point_estimate(wide)
        observed_kappa_ai = cohen_kappa_score(wide[reader_i], wide["EyeQC"], weights="quadratic")
        observed_kappa_peers = np.mean([cohen_kappa_score(wide[reader_i], wide[o], weights="quadratic") for o in others])

        boot_deltas = np.empty(N_BOOTSTRAP_EQUIV)
        for b in range(N_BOOTSTRAP_EQUIV):
            selected = rng.integers(0, len(groups), len(groups))
            row_idx = positions[np.concatenate([groups[s] for s in selected])]
            boot_deltas[b] = point_estimate(wide.iloc[row_idx])

        # --- old: two-sided 95% equivalence CI (unchanged convention) ---
        ci_lo_2sided, ci_hi_2sided = np.percentile(boot_deltas, [2.5, 97.5])
        equivalent_2sided = bool(ci_lo_2sided >= -NONINFERIORITY_MARGIN and ci_hi_2sided <= NONINFERIORITY_MARGIN)
        ci_width_2sided = float(ci_hi_2sided - ci_lo_2sided)
        ci_halfwidth_2sided = ci_width_2sided / 2.0

        # --- new: one-sided 95% non-inferiority lower bound ---
        # Judgment call: the one-sided 95% lower confidence bound is the 5th
        # percentile of the bootstrap distribution (not the 2.5th -- that is
        # the two-sided convention, and would overstate confidence here).
        # Non-inferior iff this lower bound exceeds -margin.
        ci_lo_1sided = float(np.percentile(boot_deltas, 5))
        noninferior_1sided = bool(ci_lo_1sided > -NONINFERIORITY_MARGIN)

        results.append(
            {
                "reader": reader_i,
                "n_images": n_images,
                "kappa_EyeQC_vs_reader": observed_kappa_ai,
                "kappa_reader_vs_peers_mean": observed_kappa_peers,
                "delta": observed_delta,
                "margin": NONINFERIORITY_MARGIN,
                "two_sided_ci_lo": ci_lo_2sided,
                "two_sided_ci_hi": ci_hi_2sided,
                "two_sided_ci_width": ci_width_2sided,
                "two_sided_ci_halfwidth": ci_halfwidth_2sided,
                "two_sided_equivalent": equivalent_2sided,
                "one_sided_95pct_lower_bound": ci_lo_1sided,
                "one_sided_noninferior": noninferior_1sided,
            }
        )
        print(
            f"{reader_i}: delta={observed_delta:+.4f}  "
            f"two-sided 95% CI=[{ci_lo_2sided:+.4f},{ci_hi_2sided:+.4f}] "
            f"(width={ci_width_2sided:.4f}, half-width={ci_halfwidth_2sided:.4f})  "
            f"equivalent(+/-{NONINFERIORITY_MARGIN})={equivalent_2sided}  |  "
            f"one-sided 95% lower bound={ci_lo_1sided:+.4f}  "
            f"non-inferior(> -{NONINFERIORITY_MARGIN})={noninferior_1sided}"
        )

    return pd.DataFrame(results)


def krippendorff_ordinal_alpha(wide: pd.DataFrame, raters: list[str]) -> float:
    """Krippendorff's alpha with ordinal distance weighting on the shared
    0=FAIL/1=REVIEW/2=PASS scale. reliability_data must be (n_raters, n_units)."""
    data = wide[raters].to_numpy(float).T
    return float(
        krippendorff.alpha(reliability_data=data, value_domain=[0, 1, 2], level_of_measurement="ordinal")
    )


def main() -> None:
    wide = load_wide()
    readers3 = ["R1", "R2", "R3"]
    readers4 = ["R1", "R2", "R3", "EyeQC"]
    n = len(wide)
    print(f"{n} images with complete R1/R2/R3/EyeQC data")

    # ============ Critique 1: non-inferiority ============
    print("\n=== Critique 1: one-sided non-inferiority vs. old two-sided equivalence ===")
    equiv_df = equivalence_and_noninferiority(wide, readers3)

    # ============ Critique 2: Krippendorff's ordinal alpha ============
    print("\n=== Critique 2: Krippendorff's alpha (ordinal) ===")
    alpha_rows = []
    for label, raters in [("R1+R2+R3", readers3), ("R1+R2+R3+EyeQC", readers4)]:
        point = krippendorff_ordinal_alpha(wide, raters)
        # Reuse run_four_rater_agreement_v3's own bootstrap_ci helper directly
        # (2,000 resamples, seed 20261012 -- its existing default), rather than
        # reimplementing image-level bootstrap resampling here.
        ci_lo, ci_hi = four_rater.bootstrap_ci(wide, raters, krippendorff_ordinal_alpha)
        alpha_rows.append(
            {
                "group": label,
                "n_images": n,
                "n_raters": len(raters),
                "krippendorff_alpha_ordinal": point,
                "alpha_ci_lo": ci_lo,
                "alpha_ci_hi": ci_hi,
                "fleiss_kappa_reference_unweighted": FLEISS_KAPPA_REFERENCE[label],
            }
        )
        print(
            f"{label}: Krippendorff's alpha (ordinal) = {point:.4f}  "
            f"95% CI=[{ci_lo:.4f},{ci_hi:.4f}]  "
            f"(reference Fleiss kappa, unweighted/nominal = {FLEISS_KAPPA_REFERENCE[label]:.4f})"
        )
    alpha_df = pd.DataFrame(alpha_rows)

    TABLES.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)

    noninf_name = "reader_statistics_v2_noninferiority.csv"
    alpha_name = "reader_statistics_v2_krippendorff_alpha.csv"

    equiv_df.to_csv(TABLES / noninf_name, index=False)
    equiv_df.to_csv(RESULTS_TABLES / noninf_name, index=False)
    alpha_df.to_csv(TABLES / alpha_name, index=False)
    alpha_df.to_csv(RESULTS_TABLES / alpha_name, index=False)

    print(f"\nwrote {TABLES / noninf_name}")
    print(f"wrote {RESULTS_TABLES / noninf_name}")
    print(f"wrote {TABLES / alpha_name}")
    print(f"wrote {RESULTS_TABLES / alpha_name}")


if __name__ == "__main__":
    main()
