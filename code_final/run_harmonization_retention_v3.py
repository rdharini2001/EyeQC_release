"""Supplementary Table 6B for v3: does harmonizing the ten interpretable
features (leaving each degraded image's own embedding untouched) preserve
the model's dose-response to synthetic degradations, or does it wash the
signal out the way it did for v1's haze/local_blur/overexposure response?

Same 400-image degradation sample and protocol as run_dose_response_v3.py
(same seed, same per-cohort split). Both a location-scale harmonizer and a
ComBat contract (fit exactly as v1's did — ComBat anchored to development-
plus-pooled-external, location-scale anchored to development alone) are
applied to each degraded image's own ten features and rescored with the
*same* embedding that image actually produced (kept in memory — 18,000 x
768 float32 is ~55MB, no need to touch disk or recompute it), isolating the
effect of feature harmonization from any change in what the image looked
like.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from eyeqc_v2.backbone import embed_rgb_batch  # noqa: E402
from eyeqc.batch_effects import apply_combat, fit_combat  # noqa: E402
from eyeqc.core import FEATURES, extract_features  # noqa: E402
from eyeqc.experiments import (  # noqa: E402
    DEGRADATIONS,
    apply_degradation,
    apply_location_scale_harmonizer,
    fit_location_scale_harmonizer,
)
from eyeqc.pipeline import read_rgb  # noqa: E402

DATA = REPO_ROOT / "data_final"
TABLES = DATA / "tables"
MODELS = DATA / "models"
SOURCE_LABELS = DATA / "source_labels"

DOSES = (0.0, 0.25, 0.50, 0.75, 1.0)
IMAGES_PER_COHORT = 80  # matches the 400-image / 5-cohort manuscript protocol
SEED = 20260822
COHORTS = ["EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD"]
DOSE_RESPONSE_RETENTION_THRESHOLD = 0.90
FEATURE_WORKERS = 12


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main() -> None:
    contract = json.loads((MODELS / "eyeqc_v3_model.json").read_text(encoding="utf-8"))
    mean = np.asarray(contract["standardizer_mean"])
    scale = np.asarray(contract["standardizer_scale"])
    coef = np.asarray(contract["logistic_coefficients_standardized"])
    intercept = float(contract["logistic_intercept"])

    def score_rows(hand: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        x = np.concatenate([hand, embeddings], axis=1)
        z = (x - mean) / scale
        raw_logit = intercept + z @ coef
        calibrated_logit = contract["platt_intercept"] + contract["platt_coefficient"] * raw_logit
        return _sigmoid(calibrated_logit)

    # --- fit both harmonizers exactly as v1's notebook did ---
    print("fitting harmonizers...")
    development = pd.read_parquet(SOURCE_LABELS / "development_features.parquet")
    pooled = pd.concat(
        [pd.read_parquet(TABLES / f"image_level_v3_{c}.parquet").assign(cohort=c) for c in COHORTS], ignore_index=True
    )
    combat_source = pd.concat([development.assign(cohort="EyeQ-train"), pooled], ignore_index=True)
    combat_contract = fit_combat(combat_source, batch_column="cohort", features=FEATURES)
    ls_harmonizer = fit_location_scale_harmonizer(pooled, development, batch_column="cohort")

    # --- sample the same 400 images (same seed, same per-cohort logic as run_dose_response_v3.py) ---
    rng = np.random.default_rng(SEED)
    samples = []
    for cohort in COHORTS:
        frame = pd.read_parquet(TABLES / f"image_level_v3_{cohort}.parquet")
        good = frame[frame["gt_bad"].eq(0)]
        n = min(IMAGES_PER_COHORT, len(good))
        picked = good.sample(n=n, random_state=int(rng.integers(1e6)))
        samples.append(picked.assign(cohort_=cohort))
    sample = pd.concat(samples, ignore_index=True)
    n_evals = len(sample) * len(DEGRADATIONS) * len(DOSES)
    print(f"harmonization retention (v3): {len(sample)} images x {len(DEGRADATIONS)} mechanisms x {len(DOSES)} doses = {n_evals} evaluations")

    rows = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=FEATURE_WORKERS) as pool:
        for image_index, row in enumerate(sample.itertuples()):
            base_rgb = read_rgb(row.image_path)
            for kind_index, kind in enumerate(DEGRADATIONS):
                variants = []
                for dose in DOSES:
                    if dose == 0.0:
                        img = base_rgb
                    else:
                        deg_rng = np.random.default_rng(SEED + kind_index * 10_000 + int(dose * 1000) + image_index)
                        img, _ = apply_degradation(base_rgb, kind, dose, deg_rng)
                    variants.append(img)
                feature_dicts = list(pool.map(extract_features, variants))
                embeddings = embed_rgb_batch(variants)  # kept, reused below for the harmonized rescoring
                hand = np.array([[fd[name] for name in FEATURES] for fd in feature_dicts], float)
                raw_risk = score_rows(hand, embeddings)

                # harmonize this image's own ten features (its cohort determines both contracts'
                # adjustment), then rescore with the SAME embeddings computed just above
                cohort_col = pd.Series([row.cohort_] * len(DOSES))
                chunk = pd.DataFrame(hand, columns=list(FEATURES))
                chunk["cohort"] = cohort_col.to_numpy()
                ls_hand = apply_location_scale_harmonizer(chunk, ls_harmonizer, batch_column="cohort")[list(FEATURES)].to_numpy(float)
                combat_hand = apply_combat(chunk, combat_contract)[list(FEATURES)].to_numpy(float)
                ls_risk = score_rows(ls_hand, embeddings)
                combat_risk = score_rows(combat_hand, embeddings)

                for i, dose in enumerate(DOSES):
                    rows.append(
                        {
                            "image_id": row.image_id, "cohort": row.cohort_, "degradation": kind, "dose": dose,
                            "risk_bad_raw": float(raw_risk[i]),
                            "risk_bad_location_scale": float(ls_risk[i]),
                            "risk_bad_combat": float(combat_risk[i]),
                        }
                    )
            if (image_index + 1) % 20 == 0:
                print(f"  {image_index + 1}/{len(sample)} images ({time.perf_counter() - t0:.1f}s elapsed)", flush=True)

    degraded = pd.DataFrame(rows)
    degraded.to_parquet(TABLES / "harmonization_degradation_raw_v3.parquet", index=False)

    def dose_response_retention(score_column: str) -> pd.DataFrame:
        raw_change = degraded.groupby(["degradation", "image_id"])["risk_bad_raw"].agg(lambda v: v.iloc[-1] - v.iloc[0])
        corrected_change = degraded.groupby(["degradation", "image_id"])[score_column].agg(lambda v: v.iloc[-1] - v.iloc[0])
        table = pd.concat(
            [raw_change.groupby("degradation").median().rename("median_raw_risk_change"),
             corrected_change.groupby("degradation").median().rename("median_corrected_risk_change")],
            axis=1,
        ).reset_index()
        table["dose_response_evaluable"] = table["median_raw_risk_change"].abs().gt(0.01)
        table["dose_response_retention"] = np.where(
            table["dose_response_evaluable"],
            table["median_corrected_risk_change"].abs() / table["median_raw_risk_change"].abs().clip(lower=1e-6),
            np.nan,
        )
        table["passes_dose_response_firewall"] = table["dose_response_evaluable"] & table["dose_response_retention"].ge(DOSE_RESPONSE_RETENTION_THRESHOLD)
        return table

    retention_ls = dose_response_retention("risk_bad_location_scale")
    retention_combat = dose_response_retention("risk_bad_combat")
    retention_ls.to_csv(TABLES / "harmonization_dose_response_retention_location_scale_v3.csv", index=False)
    retention_combat.to_csv(TABLES / "harmonization_dose_response_retention_combat_v3.csv", index=False)

    print("\n=== v3 degradation retention, location-scale ===")
    print(retention_ls.round(4).to_string(index=False))
    print("\n=== v3 degradation retention, ComBat ===")
    print(retention_combat.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
