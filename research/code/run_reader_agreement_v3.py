"""Reader-vs-model concordance for the hybrid v3 model, reusing the exact
80-image three-reader UKB linkage v1 built (`reader_case_level_*_links.parquet`)
and the same comparison logic (Spearman quality correlation, weighted kappa,
AUROC against each reader's own reject/non-reject split) — only the model's
risk_bad/verdict/quality_score are recomputed with v3."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from eyeqc_v2.backbone import embed_images  # noqa: E402
from eyeqc.core import FEATURES  # noqa: E402
from eyeqc.experiments import reader_operating_thresholds  # noqa: E402

import json

V1_TABLES = REPO_ROOT / "data_final" / "source_labels"
MODELS = REPO_ROOT / "data_final" / "models"
TABLES = REPO_ROOT / "data_final" / "tables"


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
    out["quality_score_eyeqc"] = 1 - risk
    out["verdict"] = np.where(risk >= fail_t, "FAIL", np.where(risk <= pass_t, "PASS", "REVIEW"))
    return out


def _auroc_or_status(reject: pd.Series, risk: pd.Series) -> tuple[float, str]:
    if reject.nunique() < 2:
        return float("nan"), "undefined: reader has only one reject class"
    return roc_auc_score(reject, risk), "estimated"


def main() -> None:
    contract = json.loads((MODELS / "eyeqc_v3_model.json").read_text(encoding="utf-8"))

    reader_long_all = pd.read_parquet(V1_TABLES / "reader_case_level_all_documented_links.parquet")
    reader_long_exact = pd.read_parquet(V1_TABLES / "reader_case_level_exact_links.parquet")

    unique_paths = reader_long_all["image_path"].dropna().unique().tolist()
    print(f"embedding {len(unique_paths)} unique reader-study images...")
    embeddings, ok = embed_images(unique_paths, batch_size=64)
    embed_map = {path: embeddings[i] for i, path in enumerate(unique_paths) if ok[i]}

    def attach_embeddings(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        keep = frame["image_path"].map(lambda p: p in embed_map)
        frame = frame[keep].reset_index(drop=True)
        emb = np.stack([embed_map[p] for p in frame["image_path"]])
        return frame, emb

    reader_long_all, emb_all = attach_embeddings(reader_long_all)
    reader_long_exact, emb_exact = attach_embeddings(reader_long_exact)

    reader_long_all = score_v3(reader_long_all, emb_all, contract)
    reader_long_exact = score_v3(reader_long_exact, emb_exact, contract)

    reader_comparison = []
    for linkage, linked_data in [("exact image key", reader_long_exact), ("documented sensitivity", reader_long_all)]:
        for reader, data in linked_data.groupby("reader"):
            reject = data["class_ordinal"].eq(0)
            auroc, auroc_status = _auroc_or_status(reject, data["risk_bad"])
            reader_comparison.append(
                {
                    "linkage": linkage,
                    "reader": reader,
                    "N": len(data),
                    "reader_FAIL_N": int(reject.sum()),
                    "reader_nonFAIL_N": int((~reject).sum()),
                    "Spearman_quality": spearmanr(data["quality_score_reader"], data["quality_score_eyeqc"], nan_policy="omit")[0],
                    "AUROC_reader_reject": auroc,
                    "AUROC_status": auroc_status,
                    "weighted_kappa": cohen_kappa_score(
                        data["class_ordinal"], data["verdict"].map({"FAIL": 0, "REVIEW": 1, "PASS": 2}), weights="quadratic"
                    ),
                }
            )
    reader_comparison = pd.DataFrame(reader_comparison)
    reader_comparison.to_csv(TABLES / "reader_eyeqc_comparison_v3.csv", index=False)
    print(reader_comparison.round(3).to_string(index=False))

    reader_thresholds = reader_operating_thresholds(reader_long_exact)
    reader_thresholds.to_csv(TABLES / "reader_operating_thresholds_v3.csv", index=False)

    v1_comparison_path = V1_TABLES / "reader_eyeqc_comparison.csv"
    if v1_comparison_path.exists():
        v1_comparison = pd.read_csv(v1_comparison_path)
        merged = reader_comparison.merge(
            v1_comparison[["linkage", "reader", "Spearman_quality", "weighted_kappa"]],
            on=["linkage", "reader"], suffixes=("_v3", "_v1"),
        )
        print("\nv1 vs v3 reader concordance:")
        print(merged[["linkage", "reader", "Spearman_quality_v1", "Spearman_quality_v3", "weighted_kappa_v1", "weighted_kappa_v3"]].round(3).to_string(index=False))
        merged.to_csv(TABLES / "v1_vs_v3_reader_comparison.csv", index=False)

    print(f"\nwrote reader-agreement tables to {TABLES}")


if __name__ == "__main__":
    main()
