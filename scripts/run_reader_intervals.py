"""Reader-study table: quadratic-weighted kappa, Spearman correlation with the grader's 0-100 quality score, AUROC for
the grader's FAIL calls, and the difference from inter-grader agreement, each with 95% participant-bootstrap intervals
(2,000 resamples).

Needs the reader-study data, which come from UK Biobank (restricted access) and are not distributed.  Outputs stay in
results/generated/ and must not be redistributed.

usage: python scripts/run_reader_intervals.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LINKS = REPO_ROOT / "data_final" / "source_labels" / "reader_case_level_exact_links.parquet"
OUTPUT = REPO_ROOT / "results" / "generated"
BOOTSTRAPS, SEED, MARGIN = 2000, 20261024, 0.15


def participant_rows(participants, rng: np.random.Generator) -> np.ndarray:
    """Row indices of one resample of whole participants."""
    _, index = np.unique(np.asarray(participants).astype(str), return_inverse=True)
    count = index.max() + 1
    times = np.bincount(rng.integers(0, count, count), minlength=count)[index]
    return np.repeat(np.arange(len(index)), times)


def kappa(a, b) -> float:
    return cohen_kappa_score(a, b, weights="quadratic")


def summarise(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``long``: one row per photograph and grader with ``class_ordinal`` (FAIL 0, REVIEW 1, PASS 2),
    ``quality_score_reader``, ``cluster_id``, ``risk`` and ``verdict`` (same codes)."""
    agreement = []
    for reader, g in long.groupby("reader"):
        grade, model = g.class_ordinal.to_numpy(int), g.verdict.to_numpy(int)
        risk, score = g.risk.to_numpy(float), g.quality_score_reader.to_numpy(float)
        rng, draws = np.random.default_rng(SEED + 600), []
        for _ in range(BOOTSTRAPS):
            i = participant_rows(g.cluster_id, rng)
            if len(np.unique(grade[i] == 0)) < 2:
                continue
            draws.append([kappa(grade[i], model[i]), spearmanr(score[i], 1 - risk[i], nan_policy="omit").statistic,
                          roc_auc_score(grade[i] == 0, risk[i])])
        low, high = np.quantile(draws, [0.025, 0.975], axis=0)
        agreement.append({"reader": reader, "n": len(g), "weighted_kappa": kappa(grade, model), "kappa_lo": low[0], "kappa_hi": high[0],
                          "Spearman": spearmanr(score, 1 - risk, nan_policy="omit").statistic, "Spearman_lo": low[1], "Spearman_hi": high[1],
                          "FAIL_AUROC": roc_auc_score(grade == 0, risk), "AUROC_lo": low[2], "AUROC_hi": high[2]})
    wide = long.pivot(index="image_id", columns="reader", values="class_ordinal")
    wide = wide.join(long.drop_duplicates("image_id").set_index("image_id")[["cluster_id", "verdict"]])
    readers = sorted(long.reader.unique())
    differences = []
    for reader in readers:
        peers = [other for other in readers if other != reader]

        def delta(frame):
            return kappa(frame[reader], frame.verdict) - np.mean([kappa(frame[reader], frame[other]) for other in peers])

        rng = np.random.default_rng(SEED + 601)
        low, high = np.quantile([delta(wide.iloc[participant_rows(wide.cluster_id, rng)]) for _ in range(BOOTSTRAPS)], [0.025, 0.975])
        differences.append({"reader": reader, "delta": delta(wide), "ci_lo": low, "ci_hi": high, "margin": MARGIN,
                            "equivalent": bool(low > -MARGIN and high < MARGIN)})
    return pd.DataFrame(agreement), pd.DataFrame(differences)


def main() -> None:
    from eyeqc.backbone import embed_images
    from eyeqc.core import FEATURES
    from eyeqc.model import load_model

    long = pd.read_parquet(LINKS)
    cases = long.drop_duplicates("image_id").reset_index(drop=True)
    embeddings, ok = embed_images(cases.image_path.tolist(), batch_size=16)
    if not ok.all():
        raise SystemExit("Some reader-study photographs could not be read.")
    model = load_model()
    scored = [model.score(row._asdict(), embedding) for row, embedding in zip(cases[list(FEATURES)].itertuples(index=False), embeddings)]
    cases["risk"] = [s["risk_bad"] for s in scored]
    cases["verdict"] = [{"FAIL": 0, "REVIEW": 1, "PASS": 2}[s["verdict"]] for s in scored]
    long = long.merge(cases[["image_id", "risk", "verdict"]], on="image_id", validate="many_to_one")
    agreement, differences = summarise(long)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    agreement.to_csv(OUTPUT / "reader_agreement_intervals.csv", index=False)
    differences.to_csv(OUTPUT / "reader_agreement_differences.csv", index=False)
    print(agreement.round(3).to_string(index=False))
    print(differences.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
