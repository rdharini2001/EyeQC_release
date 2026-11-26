"""Train MCF-Net (Fu et al. 2019 MICCAI) on EyeQC's EyeQ-train development
cohort and evaluate it with EyeQC's own development-only-threshold protocol,
for a direct baseline comparison against EyeQC.

Protocol (mirrors train_model.py / run_validation.py in scripts):
  1. 5-fold StratifiedGroupKFold (grouped by patient id) on EyeQ-train. Train
     one model per fold on the other 4 folds; predict on the held-out fold
     with a model that has never seen it. Concatenate -> honest OOF risk
     vector covering all of EyeQ-train.
  2. Fit pass/fail thresholds on that OOF vector with eyeqc.development's own
     _choose_fail_threshold / _choose_pass_threshold (target sensitivity 0.90,
     target PASS NPV 0.98), gt_bad = (quality == 2) i.e. Reject-only poor.
  3. Train one final model on 100% of EyeQ-train.
  4. Score the 5 external cohorts with the final model; apply the two fixed
     thresholds UNCHANGED (no recalibration on external data).
  5. Compute performance()/cluster_bootstrap_intervals() exactly as
     run_validation.py does, and write baseline_mcfnet_v3.csv.

This script only *adds* new files. It does not modify anything under
the repository, data_final, or results/generated except to add new baseline output
tables, and it does not modify the cloned MCF-Net repo.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import os

REPOSITORY = Path(__file__).resolve().parents[3]
DATASETS = Path(os.environ.get("EYEQC_DATASETS", REPOSITORY / "datasets"))
EXTERNAL = Path(os.environ.get("EYEQC_EXTERNAL", REPOSITORY / "external"))      # clones and weights of third-party methods

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mcfnet_common import (  # noqa: E402
    MCFNetInferenceDataset,
    MCFNetTrainDataset,
    build_model,
    eval_loss,
    one_hot,
    patient_group_id,
    predict_risk,
    train_one_epoch,
)

CODE_DIR = REPOSITORY
sys.path.insert(0, str(CODE_DIR))
from eyeqc.development import _choose_fail_threshold, _choose_pass_threshold  # noqa: E402
from eyeqc.statistics import cluster_bootstrap_intervals, performance  # noqa: E402

BOOTSTRAPS = 1000

TRAIN_LABEL_CSV = (DATASETS / "EyePACS/Label_EyeQ_train.csv")
CACHE_ROOT = EXTERNAL / "mcfnet_cache"
TABLES_DIR = (REPOSITORY / "data_final/tables")
OUT_DATA = TABLES_DIR / "baseline_mcfnet_v3.csv"
OUT_RESULTS = (REPOSITORY / "results/generated/baseline_mcfnet_v3.csv")
CKPT_DIR = HERE / "checkpoints"
LOGS_DIR = HERE / "logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

COHORTS = ["EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD"]
SEED = 20260822
TARGET_BAD_SENSITIVITY = 0.90
TARGET_PASS_NPV = 0.98
LOSS_W = [0.1, 0.1, 0.1, 0.1, 0.6]  # same weights as Main_EyeQuality.py

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_development_frame() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_LABEL_CSV)
    train_dir = CACHE_ROOT / "train"
    # one directory scan + set membership instead of 12.5k individual
    # Path.exists() syscalls -- much faster under concurrent disk I/O load
    cached_stems = {p.stem for p in train_dir.glob("*.png")}
    df["png_path"] = df["image"].apply(lambda name: str(train_dir / (Path(name).stem + ".png")))
    exists = df["image"].apply(lambda name: Path(name).stem in cached_stems)
    dropped = int((~exists).sum())
    if dropped:
        print(f"[dev] dropping {dropped} rows with missing/failed-preprocess PNGs", flush=True)
    df = df[exists].reset_index(drop=True)
    df["group"] = df["image"].apply(patient_group_id)
    return df


def make_loaders(df: pd.DataFrame, train_idx: np.ndarray, holdout_idx: np.ndarray, batch_size: int, val_frac: float = 0.1):
    """Carve a small *internal* validation split (grouped, stratified) out of
    train_idx only -- for early stopping -- so holdout_idx stays completely
    unseen until final OOF prediction."""
    sub = df.iloc[train_idx].reset_index(drop=True)
    y_sub = sub["quality"].to_numpy()
    g_sub = sub["group"].to_numpy()
    n_splits = max(int(round(1 / val_frac)), 3)
    try:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        inner_train, inner_val = next(cv.split(np.zeros(len(sub)), y_sub, g_sub))
    except ValueError:
        # fallback: not enough groups per class for that many splits
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
        inner_train, inner_val = next(cv.split(np.zeros(len(sub)), y_sub, g_sub))

    train_df = sub.iloc[inner_train].reset_index(drop=True)
    val_df = sub.iloc[inner_val].reset_index(drop=True)

    train_ds = MCFNetTrainDataset(train_df["png_path"].tolist(), one_hot(train_df["quality"].to_numpy()), train=True)
    val_ds = MCFNetTrainDataset(val_df["png_path"].tolist(), one_hot(val_df["quality"].to_numpy()), train=False)

    # num_workers=0 (no forked/spawned worker processes): this machine is a
    # shared desktop running several other concurrent jobs and was observed to
    # be down to ~134MB free system RAM; spawning extra python+torch worker
    # processes was intermittently failing with WinError 1450 (insufficient
    # system resources). Single-process loading is slower but reliable here.
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    holdout_df = df.iloc[holdout_idx].reset_index(drop=True)
    holdout_ds = MCFNetInferenceDataset(holdout_df["png_path"].tolist())
    holdout_loader = torch.utils.data.DataLoader(holdout_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, holdout_loader, len(holdout_df)


def train_model(train_loader, val_loader, max_epochs: int, patience: int, lr: float, tag: str) -> tuple[torch.nn.Module, dict]:
    model = build_model(DEVICE)
    criterion = torch.nn.BCELoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    best_val = np.inf
    best_state = None
    best_epoch = -1
    history = []
    bad_epochs = 0

    for epoch in range(max_epochs):
        t0 = time.time()
        train_loss = train_one_epoch(train_loader, model, optimizer, criterion, LOSS_W, DEVICE)
        val_loss = eval_loss(val_loader, model, criterion, DEVICE)
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "seconds": dt})
        print(f"[{tag}] epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} ({dt:.1f}s)")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[{tag}] early stopping at epoch {epoch} (best epoch {best_epoch}, best val_loss {best_val:.4f})")
                break

    converged = best_epoch < max_epochs - 1  # stopped before exhausting budget -> validation loss had turned up
    if best_state is not None:
        model.load_state_dict(best_state)
    info = {"tag": tag, "best_epoch": best_epoch, "best_val_loss": float(best_val), "history": history,
            "epochs_run": len(history), "max_epochs": max_epochs, "early_stopped": converged}
    return model, info


def predict_for_rows(model, png_paths: list[str], batch_size: int) -> np.ndarray:
    ds = MCFNetInferenceDataset(png_paths)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return predict_risk(loader, model, DEVICE, len(ds))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--stage", choices=["folds", "final", "score", "all"], default="all")
    args = ap.parse_args()

    df = load_development_frame()
    y = df["quality"].to_numpy()
    groups = df["group"].to_numpy()
    print(f"development set: {len(df)} images, {len(set(groups))} patients, "
          f"quality dist: {np.bincount(y)} (0=Good,1=Usable,2=Reject)")

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(cv.split(np.zeros(len(y)), y, groups))

    run_log = {}
    t_start_all = time.time()

    oof_risk = np.full(len(df), np.nan, dtype="float32")
    oof_fold = np.full(len(df), -1, dtype="int32")

    if args.stage in ("folds", "all"):
        for fold_i, (train_idx, holdout_idx) in enumerate(folds):
            ckpt_path = CKPT_DIR / f"fold{fold_i}.pt"
            if ckpt_path.exists():
                print(f"[fold {fold_i}] checkpoint exists, loading and re-predicting OOF only")
                model = build_model(DEVICE)
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            else:
                train_loader, val_loader, _holdout_loader, _n = make_loaders(df, train_idx, holdout_idx, args.batch_size)
                t0 = time.time()
                model, info = train_model(train_loader, val_loader, args.max_epochs, args.patience, args.lr, f"fold{fold_i}")
                info["train_seconds"] = time.time() - t0
                run_log[f"fold{fold_i}"] = info
                torch.save(model.state_dict(), ckpt_path)
                (LOGS_DIR / f"fold{fold_i}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

            holdout_df = df.iloc[holdout_idx].reset_index(drop=True)
            risk = predict_for_rows(model, holdout_df["png_path"].tolist(), args.batch_size)
            oof_risk[holdout_idx] = risk
            oof_fold[holdout_idx] = fold_i
            del model
            torch.cuda.empty_cache()

        df["oof_risk_bad"] = oof_risk
        df["oof_fold"] = oof_fold
        df.to_parquet(HERE / "oof_predictions.parquet", index=False)
        print(f"wrote {HERE / 'oof_predictions.parquet'}")

    if args.stage == "folds":
        print(f"total fold stage time: {time.time() - t_start_all:.1f}s")
        return

    # ---- final model on 100% of development data ----
    final_ckpt = CKPT_DIR / "final.pt"
    if args.stage in ("final", "all") and not final_ckpt.exists():
        all_idx = np.arange(len(df))
        # carve internal val the same way, from the full set, purely for early stopping
        train_loader, val_loader, _unused_loader, _n = make_loaders(
            df, all_idx, np.array([], dtype=int), args.batch_size
        )
        t0 = time.time()
        model, info = train_model(train_loader, val_loader, args.max_epochs, args.patience, args.lr, "final")
        info["train_seconds"] = time.time() - t0
        run_log["final"] = info
        torch.save(model.state_dict(), final_ckpt)
        (LOGS_DIR / "final.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        del model
        torch.cuda.empty_cache()

    if run_log:
        (LOGS_DIR / "run_summary.json").write_text(json.dumps(run_log, indent=2), encoding="utf-8")

    if args.stage == "final":
        print(f"total time: {time.time() - t_start_all:.1f}s")
        return

    # ---- thresholds from OOF (recompute if we have oof file but didn't just run folds) ----
    oof_path = HERE / "oof_predictions.parquet"
    if not oof_path.exists():
        raise RuntimeError("OOF predictions not found; run --stage folds first")
    oof_df = pd.read_parquet(oof_path)
    y_oof = (oof_df["quality"] == 2).astype(int).to_numpy()
    risk_oof = oof_df["oof_risk_bad"].to_numpy(dtype=float)
    oof_auroc = None
    from sklearn.metrics import roc_auc_score
    oof_auroc = roc_auc_score(y_oof, risk_oof)
    fail_t = _choose_fail_threshold(y_oof, risk_oof, TARGET_BAD_SENSITIVITY)
    pass_t = _choose_pass_threshold(y_oof, risk_oof, fail_t, TARGET_PASS_NPV)
    print(f"\n=== development OOF (n={len(oof_df)}) ===")
    print(f"OOF AUROC (gt_bad=Reject-only) = {oof_auroc:.4f}")
    print(f"fail_threshold (risk >= this -> FAIL) = {fail_t:.6f}")
    print(f"pass_threshold (risk <= this -> PASS) = {pass_t:.6f}")

    thresholds = {
        "development_oof_auroc": float(oof_auroc),
        "fail_threshold": float(fail_t),
        "pass_threshold": float(pass_t),
        "target_bad_sensitivity": TARGET_BAD_SENSITIVITY,
        "target_pass_npv": TARGET_PASS_NPV,
        "n_development_images": int(len(oof_df)),
        "n_development_patients": int(oof_df["group"].nunique()),
    }
    (HERE / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    print(f"wrote {HERE / 'thresholds.json'}")

    # ---- score external cohorts with the final model ----
    final_model = build_model(DEVICE)
    final_model.load_state_dict(torch.load(final_ckpt, map_location=DEVICE))

    rows = []
    for cohort_i, cohort in enumerate(COHORTS):
        cohort_df = pd.read_parquet(TABLES_DIR / f"image_level_v3_{cohort}.parquet")
        cache_dir = CACHE_ROOT / cohort
        import re
        def sanitize(name):
            return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))
        cached_stems = {p.stem for p in cache_dir.glob("*.png")}
        cohort_df["png_path"] = cohort_df["image_id"].apply(lambda i: str(cache_dir / (sanitize(i) + ".png")))
        exists = cohort_df["image_id"].apply(lambda i: sanitize(i) in cached_stems)
        n_total = len(cohort_df)
        n_dropped = int((~exists).sum())
        cohort_df = cohort_df[exists].reset_index(drop=True)
        print(f"[{cohort}] {n_total} total, {n_dropped} dropped (missing cached PNG), {len(cohort_df)} scored", flush=True)

        risk = predict_for_rows(final_model, cohort_df["png_path"].tolist(), args.batch_size)
        cohort_df["risk_bad"] = risk
        cohort_df[["image_id", "cluster_id", "gt_bad", "risk_bad"]].to_parquet(
            HERE / f"scored_{cohort}.parquet", index=False
        )

        eval_frame = pd.DataFrame({
            "gt_bad": cohort_df["gt_bad"],
            "risk_bad": cohort_df["risk_bad"],
            "cluster_id": cohort_df["cluster_id"],
        })
        point = performance(eval_frame, pass_t, fail_t)
        intervals = cluster_bootstrap_intervals(
            eval_frame, pass_t, fail_t, bootstraps=BOOTSTRAPS, workers=4, seed=SEED + cohort_i * 10_000
        )
        row = {
            "cohort": cohort,
            "N": len(cohort_df),
            "poor_N": int(cohort_df["gt_bad"].sum()),
            "AUROC": point["AUROC"],
            "AUPRC": point["AUPRC"],
            "Brier": point["Brier"],
            "FAIL_sensitivity": point["FAIL_sensitivity"],
            "PASS_NPV": point["PASS_NPV"],
            "n_dropped_missing_png": n_dropped,
        }
        for metric, (low, high) in intervals.items():
            row[f"{metric}_lo"] = low
            row[f"{metric}_hi"] = high
        rows.append(row)
        print(f"[{cohort}] N={len(cohort_df)} AUROC={point['AUROC']:.3f} "
              f"FAIL_sens={point['FAIL_sensitivity']:.3f} PASS_NPV={point['PASS_NPV']:.3f}")

    table = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DATA, index=False)
    table.to_csv(OUT_RESULTS, index=False)
    print(f"\nwrote {OUT_DATA}\nwrote {OUT_RESULTS}")
    print("\n", table.round(4).to_string(index=False))
    print(f"\ntotal wall time this invocation: {time.time() - t_start_all:.1f}s")


if __name__ == "__main__":
    main()
