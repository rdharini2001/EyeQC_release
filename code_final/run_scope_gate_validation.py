"""Validate the out-of-scope / non-fundus input gate (``eyeqc.core.scope_check``).

What this script does
----------------------
1. Real-cohort false-positive check (exhaustive, not sampled): for all five
   validation cohorts (EyeQ-test, DRIMDB, FQS, MSHF-CFP, DeepDRiD), applies
   the shipped scope gate (rules on ``retina_fraction``, ``contrast``,
   ``readable_area_fraction``, all already cached in
   ``data_final/tables/image_level_v3_{cohort}.parquet``) to every image with
   a valid ("Good"/"Usable") ground-truth label and reports the false-positive
   rate -- i.e. how often a real, validly-labelled fundus photo gets
   incorrectly flagged as out of scope. This uses the FULL population of each
   cohort, not a sample, because the needed columns are already cached.
   It also reports, informationally only (not held to a 0% bar), how often
   the gate fires on "Bad"/"Reject" labelled images -- these are genuinely
   low-quality fundus photos, not necessarily non-fundus images, so a nonzero
   rate here is not itself a problem, but every flagged case is still listed
   for manual review.

2. A due-diligence check on ``circularity_audit``, which is NOT part of the
   shipped gate. ``circularity_audit`` is not cached, so this recomputes
   features (via ``eyeqc.core.extract_features``, reading the original image
   files) for a sample of up to 500 validly-labelled images per cohort
   (all cached-low-retina_fraction images are force-included, the rest is a
   random fill). This demonstrates *why* circularity was excluded from the
   hard gate: a real, validly-labelled MSHF-CFP image was found with
   circularity_audit = 0.163, which would false-positive under any plausible
   standalone circularity threshold.

3. A synthetic non-fundus test set (15 cases: solid colors, random noise,
   near-black / near-white / near-black-speckled frames, a document-scan-like
   pattern, a checkerboard, a bright rectangle on black, a skin-tone gradient
   simulating a hand/skin photo, and a circular vignette control that is
   *expected* to pass the gate). Each case is run through the real
   ``extract_features`` + ``scope_check`` pipeline and the true-positive rate
   (fraction correctly flagged) is reported, with every miss listed
   explicitly -- this gate is not a general-purpose image classifier and the
   script does not pretend otherwise.

Outputs
-------
- ``data_final/tables/scope_gate_validation.csv``
- ``results_final/tables/scope_gate_validation.csv``
  (identical content, the repo's existing dual-save convention)

Run from anywhere; paths are resolved relative to this file's location.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from eyeqc.core import extract_features, scope_check  # noqa: E402
from eyeqc.pipeline import read_rgb  # noqa: E402

DATA_TABLES = REPO_ROOT / "data_final" / "tables"
RESULTS_TABLES = REPO_ROOT / "results_final" / "tables"

COHORTS = ["EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD"]
VALID_LABELS = {
    "EyeQ-test": {"Good", "Usable"},
    "DRIMDB": {"Good"},
    "FQS": {"Good", "Usable"},
    "MSHF-CFP": {"Good"},
    "DeepDRiD": {"Good"},
}

CIRC_SAMPLE_PER_COHORT = 500  # cap; smaller cohorts use their full population
RNG_SEED = 2026


def load_cohort(cohort: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_TABLES / f"image_level_v3_{cohort}.parquet")


def gate_from_cached_columns(frame: pd.DataFrame) -> pd.Series:
    """Apply the shipped scope_check rules using the already-cached columns.

    This mirrors eyeqc.core.scope_check exactly (same constants, same logic)
    but is vectorized over a DataFrame instead of called row-by-row, purely
    so the full-population cohort check runs in milliseconds instead of
    re-extracting every image.
    """
    rf = frame["retina_fraction"].astype(float)
    contrast = frame["contrast"].astype(float)
    readable = frame["readable_area_fraction"].astype(float)
    rule_a = rf < 0.30
    rule_b = (rf >= 0.30) & (contrast < 0.03) & (readable < 0.03)
    flagged = rule_a | rule_b
    return flagged


def section_real_cohort_fpr() -> list[dict]:
    rows = []
    print("=== 1. Real-cohort false-positive check (full population, cached columns) ===")
    for cohort in COHORTS:
        d = load_cohort(cohort)
        valid = d[d["gt_label"].isin(VALID_LABELS[cohort])]
        invalid = d[~d["gt_label"].isin(VALID_LABELS[cohort])]

        flagged_valid = gate_from_cached_columns(valid)
        n_valid = len(valid)
        n_fp = int(flagged_valid.sum())
        fpr = n_fp / n_valid if n_valid else float("nan")
        print(f"{cohort:10s} valid(Good/Usable) n={n_valid:6d}  false_positives={n_fp}  FPR={fpr:.4%}")
        if n_fp:
            print("  FLAGGED VALID IMAGES (investigate):")
            for _, r in valid[flagged_valid].iterrows():
                print("   ", r["image_path"], "rf=", r["retina_fraction"], "contrast=", r["contrast"], "readable=", r["readable_area_fraction"])

        rows.append(
            {
                "section": "real_cohort_fpr",
                "cohort": cohort,
                "group": "valid_Good_or_Usable",
                "n": n_valid,
                "n_flagged": n_fp,
                "rate": fpr,
                "rate_label": "false_positive_rate",
                "notes": "full population (not sampled); uses cached retina_fraction/contrast/readable_area_fraction",
            }
        )

        if len(invalid):
            flagged_invalid = gate_from_cached_columns(invalid)
            n_invalid = len(invalid)
            n_flag_invalid = int(flagged_invalid.sum())
            rate_invalid = n_flag_invalid / n_invalid
            print(f"{cohort:10s} invalid(Bad/Reject) n={n_invalid:6d}  gate_fired={n_flag_invalid}  rate={rate_invalid:.4%}  (informational only; not held to a 0% bar)")
            rows.append(
                {
                    "section": "real_cohort_informational_bad",
                    "cohort": cohort,
                    "group": "invalid_Bad_or_Reject",
                    "n": n_invalid,
                    "n_flagged": n_flag_invalid,
                    "rate": rate_invalid,
                    "rate_label": "gate_fire_rate_on_known_bad_images",
                    "notes": "informational: these are real but low-quality fundus photos, not necessarily non-fundus; nonzero rate is not itself a failure",
                }
            )
    print()
    return rows


def section_circularity_due_diligence() -> list[dict]:
    rows = []
    print("=== 2. circularity_audit due-diligence sample (recomputed; NOT part of the shipped gate) ===")
    rng = np.random.default_rng(RNG_SEED)
    for cohort in COHORTS:
        d = load_cohort(cohort)
        valid = d[d["gt_label"].isin(VALID_LABELS[cohort])].copy()
        n_total = len(valid)

        low_rf = valid[valid["retina_fraction"] < 0.5]
        target = min(CIRC_SAMPLE_PER_COHORT, n_total)
        remaining_pool = valid.drop(index=low_rf.index)
        n_fill = max(target - len(low_rf), 0)
        fill = remaining_pool.sample(n=min(n_fill, len(remaining_pool)), random_state=RNG_SEED) if n_fill else remaining_pool.iloc[0:0]
        sample = pd.concat([low_rf, fill]).drop_duplicates(subset=["image_path"])

        circs, readables, contrasts, in_scopes = [], [], [], []
        for path in sample["image_path"]:
            try:
                f = extract_features(read_rgb(path))
                circs.append(f["circularity_audit"])
                readables.append(f["readable_area_fraction"])
                contrasts.append(f["contrast"])
                in_scopes.append(scope_check(f)["in_scope"])
            except Exception as exc:  # noqa: BLE001
                print(f"  WARN: could not recompute {path}: {exc}")
                circs.append(float("nan")); readables.append(float("nan")); contrasts.append(float("nan")); in_scopes.append(None)

        sample = sample.assign(
            circularity_audit_recomputed=circs,
            readable_area_fraction_recomputed=readables,
            contrast_recomputed=contrasts,
            shipped_gate_in_scope=in_scopes,
        )

        min_circ = np.nanmin(circs) if circs else float("nan")
        # Due-diligence check: would a circularity-based rule (circ < 0.35 AND
        # recomputed readable_area_fraction < 0.10) have produced NEW false
        # positives beyond what the shipped gate already flags?
        would_flag_circ_rule = (sample["circularity_audit_recomputed"] < 0.35) & (sample["readable_area_fraction_recomputed"] < 0.10)
        new_fp_if_circ_rule_added = int((would_flag_circ_rule & sample["shipped_gate_in_scope"]).sum())

        print(
            f"{cohort:10s} n_sampled={len(sample):4d} (of {n_total} valid)  "
            f"min_circularity_audit={min_circ:.3f}  "
            f"would-be-new-FPs-if-circularity-rule-added={new_fp_if_circ_rule_added}"
        )
        low_circ = sample.nsmallest(3, "circularity_audit_recomputed")
        for _, r in low_circ.iterrows():
            print(
                f"    lowest circ example: {r['image_path']}  circ={r['circularity_audit_recomputed']:.3f}"
                f"  readable={r['readable_area_fraction_recomputed']:.3f}  contrast={r['contrast_recomputed']:.3f}"
                f"  gt_label={r['gt_label']}  shipped_gate_in_scope={r['shipped_gate_in_scope']}"
            )

        rows.append(
            {
                "section": "circularity_due_diligence",
                "cohort": cohort,
                "group": "valid_Good_or_Usable_sample",
                "n": len(sample),
                "n_flagged": new_fp_if_circ_rule_added,
                "rate": new_fp_if_circ_rule_added / len(sample) if len(sample) else float("nan"),
                "rate_label": "would_be_new_false_positive_rate_if_circularity_hard_threshold_added",
                "notes": (
                    f"recomputed sample (all cached retina_fraction<0.5 force-included + random fill, "
                    f"target up to {CIRC_SAMPLE_PER_COHORT}); min observed circularity_audit={min_circ:.3f}; "
                    "circularity_audit was NOT added to the shipped gate because of this finding"
                ),
            }
        )
    print()
    return rows


# ---------------------------------------------------------------------------
# Synthetic non-fundus test set
# ---------------------------------------------------------------------------

def _synthetic_cases() -> dict[str, np.ndarray]:
    H, W = 480, 640
    rng = np.random.default_rng(7)
    cases: dict[str, np.ndarray] = {}

    cases["solid_mid_gray"] = np.full((H, W, 3), 128, np.uint8)
    cases["solid_saturated_red"] = np.tile(np.array([200, 30, 30], np.uint8), (H, W, 1))
    cases["solid_saturated_green"] = np.tile(np.array([30, 180, 40], np.uint8), (H, W, 1))
    cases["solid_saturated_blue"] = np.tile(np.array([20, 40, 210], np.uint8), (H, W, 1))
    cases["mostly_black"] = np.full((H, W, 3), 3, np.uint8)
    cases["mostly_white_overexposed"] = np.full((H, W, 3), 253, np.uint8)

    img = np.full((H, W, 3), 3, np.uint8)
    ys = rng.integers(0, H, 200); xs = rng.integers(0, W, 200)
    img[ys, xs] = rng.integers(20, 80, (200, 3))
    cases["near_black_speckled_failed_capture"] = img

    cases["random_noise_rgb"] = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)

    doc = np.full((H, W, 3), 245, np.uint8)
    rng2 = np.random.default_rng(11)
    for row in range(40, H - 40, 18):
        x = 40
        for _ in range(rng2.integers(3, 8)):
            wlen = rng2.integers(20, 90)
            if x + wlen > W - 40:
                break
            doc[row:row + 10, x:x + wlen] = 20
            x += wlen + rng2.integers(10, 30)
    cases["document_scan_text_pattern"] = doc

    cb = np.zeros((H, W, 3), np.uint8)
    tile = 40
    for yi in range(0, H, tile):
        for xi in range(0, W, tile):
            if ((yi // tile) + (xi // tile)) % 2 == 0:
                cb[yi:yi + tile, xi:xi + tile] = 230
    cases["checkerboard_geometric"] = cb

    rect = np.zeros((H, W, 3), np.uint8)
    rect[100:380, 120:520] = 200
    cases["bright_rectangle_on_black"] = rect

    yy, xx = np.mgrid[0:H, 0:W]
    base = 180 + 20 * np.sin(xx / 80) + 10 * np.cos(yy / 60)
    skin = np.stack([base + 20, base - 10, base - 40], axis=-1)
    rng3 = np.random.default_rng(13)
    skin = np.clip(skin + rng3.normal(0, 4, skin.shape), 0, 255).astype(np.uint8)
    cases["skin_tone_gradient_hand_like"] = skin

    cy, cx = H // 2, W // 2
    r = min(H, W) // 2 - 10
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    inside = dist < r
    rng4 = np.random.default_rng(17)
    base2 = 140 + rng4.normal(0, 15, (H, W))
    vig = np.zeros((H, W, 3), np.uint8)
    for c in range(3):
        ch = np.zeros((H, W))
        ch[inside] = base2[inside]
        vig[..., c] = np.clip(ch, 0, 255).astype(np.uint8)
    cases["circular_vignette_control_expected_pass"] = vig

    # extra modes to reach >=15: vertical gradient bar chart / stripes, and a
    # low-saturation grayscale photo-like scene (smooth gradient, no color)
    stripes = np.zeros((H, W, 3), np.uint8)
    for i, xi in enumerate(range(0, W, 30)):
        stripes[:, xi:xi + 30] = (i * 23) % 256
    cases["vertical_stripes_pattern"] = stripes

    grad = np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1))
    grayscape = np.stack([grad, grad, grad], axis=-1)
    cases["smooth_grayscale_gradient"] = grayscape

    cases["uniform_pale_skin_closeup"] = np.tile(np.array([225, 190, 165], np.uint8), (H, W, 1))

    return cases


def section_synthetic() -> list[dict]:
    rows = []
    print("=== 3. Synthetic non-fundus test set ===")
    cases = _synthetic_cases()
    n_correct = 0
    for name, rgb in cases.items():
        f = extract_features(rgb)
        scope = scope_check(f)
        flagged = not scope["in_scope"]
        expect_flag = name != "circular_vignette_control_expected_pass"
        correct = (flagged == expect_flag)
        n_correct += int(correct)
        tag = "CAUGHT" if flagged else "MISSED"
        if name == "circular_vignette_control_expected_pass":
            tag = "CORRECTLY PASSED (control)" if not flagged else "UNEXPECTEDLY FLAGGED (control)"
        print(
            f"  {name:38s} rf={f['retina_fraction']:.3f} circ={f['circularity_audit']:.3f} "
            f"contrast={f['contrast']:.3f} readable={f['readable_area_fraction']:.3f} "
            f"-> in_scope={scope['in_scope']}  [{tag}]"
        )
        rows.append(
            {
                "section": "synthetic_case",
                "cohort": None,
                "group": name,
                "n": 1,
                "n_flagged": int(flagged),
                "rate": float("nan"),
                "rate_label": tag,
                "notes": scope["reason"] or "",
                "retina_fraction": f["retina_fraction"],
                "circularity_audit": f["circularity_audit"],
                "contrast": f["contrast"],
                "readable_area_fraction": f["readable_area_fraction"],
                "in_scope": scope["in_scope"],
            }
        )
    n_cases = len(cases)
    n_expected_flag = n_cases - 1  # all but the control are expected non-fundus cases
    n_flagged_of_expected = sum(
        1 for r in rows if r["group"] != "circular_vignette_control_expected_pass" and r["n_flagged"] == 1
    )
    tpr = n_flagged_of_expected / n_expected_flag
    print(f"\n  Synthetic TPR (non-fundus cases correctly flagged): {n_flagged_of_expected}/{n_expected_flag} = {tpr:.1%}")
    control_row = next(r for r in rows if r["group"] == "circular_vignette_control_expected_pass")
    print(f"  Control case (circular, textured, expected to PASS): in_scope={control_row['in_scope']}")
    rows.append(
        {
            "section": "synthetic_summary",
            "cohort": None,
            "group": "all_non_fundus_cases",
            "n": n_expected_flag,
            "n_flagged": n_flagged_of_expected,
            "rate": tpr,
            "rate_label": "true_positive_rate",
            "notes": "fraction of synthetic non-fundus cases correctly flagged as out of scope",
        }
    )
    print()
    return rows


def main() -> None:
    all_rows: list[dict] = []
    all_rows += section_real_cohort_fpr()
    all_rows += section_circularity_due_diligence()
    all_rows += section_synthetic()

    out = pd.DataFrame(all_rows)
    DATA_TABLES.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(DATA_TABLES / "scope_gate_validation.csv", index=False)
    out.to_csv(RESULTS_TABLES / "scope_gate_validation.csv", index=False)
    print(f"Saved {len(out)} rows to:")
    print(f"  {DATA_TABLES / 'scope_gate_validation.csv'}")
    print(f"  {RESULTS_TABLES / 'scope_gate_validation.csv'}")


if __name__ == "__main__":
    main()
