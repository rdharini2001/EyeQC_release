"""The research applications of the paper, applied to the user's own photographs and tables (no layout code here).

Robustness to degradations, spatial localisation, consistency between two photographs of one eye, quality phenotypes,
cohort quality profiles, batch variation and harmonization, local validation and recalibration, and reader agreement.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from eyeqc.agreement import EQUIVALENCE_MARGIN, reader_report, to_codes
from eyeqc.batch_effects import (apply_combat, fit_combat, pathology_quality_disentanglement, separability_permutation_test,
                                 variance_decomposition)
from eyeqc.contract import FAIL_THRESHOLD, PASS_THRESHOLD
from eyeqc.core import DISPLAY_NAMES, FEATURES, extract_features
from eyeqc.experiments import DEGRADATIONS, apply_degradation, domain_separability, robust_location_scale_harmonize
from eyeqc.maps import GRID, MAP_KEYS, lowest_region, lowest_tile_hit, pixel_auc, quality_maps, tile_grid
from eyeqc.phenotypes import cohort_profiles, phenotype_composition, quality_phenotype
from eyeqc.recalibration import fit_recalibration, labelled_participants, local_pass_threshold, operating_metrics
from eyeqc.statistics import cluster_bootstrap_intervals, performance

import charts
import engine
from engine import MODEL, read_photo

MAX_BATCH = 60
RECALIBRATION_SPLITS = 100
DOSES = (0.0, 0.25, 0.5, 0.75, 1.0)
DEGRADATION_NAMES = {kind: ("JPEG compression" if kind == "jpeg" else kind.replace("_", " ").capitalize()) for kind in DEGRADATIONS}
DEGRADATION_LABELS = {name: kind for kind, name in DEGRADATION_NAMES.items()}
LOCAL_DEFECTS = {"Glare": "glare", "Shadow": "shadow", "Local blur": "local_blur"}
RISK_SEED = 20260822


def _save_csv(frame: pd.DataFrame, name: str, index: bool = False) -> str:
    target = Path(tempfile.mkdtemp(prefix="eyeqc_")) / f"{name}.csv"
    frame.to_csv(target, index=index)
    return str(target)


def _read_table(csv_path: str | None, needed: tuple[str, ...] = ()) -> pd.DataFrame:
    if not csv_path:
        raise ValueError("Choose a CSV file first.")
    frame = pd.read_csv(csv_path)
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise ValueError("The CSV needs the columns: " + ", ".join(missing))
    return frame


# ---------------------------------------------------------------- robustness
def _degraded_scores(rgb: np.ndarray, kinds: list[str]) -> tuple[pd.DataFrame, dict]:
    rows, images = [], {}
    original = MODEL.analyze(rgb)
    for kind in kinds:
        for dose in DOSES:
            if dose == 0:
                changed, result = rgb, original
            else:
                changed, _ = apply_degradation(rgb, kind, dose, np.random.default_rng(RISK_SEED))
                result = MODEL.analyze(changed)
            images[kind, dose] = changed
            rows.append({"degradation": kind, "severity": dose, "risk": result["risk_bad"], "verdict": result["verdict"],
                         **{name: float(result["features"][name]) for name in FEATURES}})
    return pd.DataFrame(rows), images


def degradation_response(path: str, label: str):
    """EyeQC risk, decision and measurements as one simulated defect increases."""
    kind = DEGRADATION_LABELS[label]
    table, images = _degraded_scores(read_photo(path), [kind])
    gallery = [(images[kind, row.severity], f"{row.severity:.0%} · {row.verdict} · {row.risk:.2f}") for row in table.itertuples()]
    shown = table.drop(columns="degradation").rename(columns={"severity": "Severity", "risk": "Poor-quality risk", "verdict": "Decision", **DISPLAY_NAMES})
    return charts.dose_response(table, label), gallery, shown.round(3), _save_csv(table, f"eyeqc_{kind}")


def degradation_all(path: str):
    """All nine simulated defects at five severities."""
    table, _ = _degraded_scores(read_photo(path), list(DEGRADATIONS))
    summary = []
    for kind, group in table.groupby("degradation", sort=False):
        risk = group.sort_values("severity")["risk"].to_numpy()
        summary.append({"Defect": DEGRADATION_NAMES[kind], "Risk at 0%": round(risk[0], 3), "Risk at 100%": round(risk[-1], 3),
                        "Change": round(risk[-1] - risk[0], 3), "Never decreases": "yes" if np.all(np.diff(risk) >= -1e-6) else "no",
                        "Decision at 100%": group.sort_values("severity")["verdict"].iloc[-1]})
    return charts.dose_grid(table, DEGRADATION_NAMES), pd.DataFrame(summary), _save_csv(table, "eyeqc_all_degradations")


def localisation(path: str, label: str, severity: float, placement: int, style: str = "spectral"):
    """One local defect at a known place: the maps, and how well each of them finds it."""
    kind = LOCAL_DEFECTS[label]
    changed, mask = apply_degradation(read_photo(path), kind, float(severity), np.random.default_rng(int(placement)))
    if mask is None:
        raise ValueError("Choose a severity above zero.")
    features = extract_features(changed)
    maps = quality_maps(features)
    truth = cv2.resize(mask.astype(np.uint8), features["inner_mask"].shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    coverage = cv2.resize(truth.astype(np.float32), (GRID, GRID), interpolation=cv2.INTER_AREA) >= 0.25
    box = engine.square_box(features["support_mask"])
    tiles, rows = {}, []
    for key in MAP_KEYS:
        auc = pixel_auc(maps[key], truth)
        lowest = lowest_region(maps[key])
        grid = tile_grid(features, key)
        hit = lowest_tile_hit(grid, coverage & np.isfinite(grid))
        tiles[key] = engine.tile(features, maps[key], style, box, "–" if np.isnan(auc) else f"{auc:.2f}", truth)
        rows.append({"Map": "Combined quality" if key == "quality" else key.capitalize(),
                     "Pixel AUC (0.5 = chance)": None if np.isnan(auc) else round(float(auc), 3),
                     "Lowest region inside defect": "no" if lowest is None else "yes" if truth[int(lowest[0]), int(lowest[1])] else "no",
                     "Lowest 6×6 tile inside defect": "–" if np.isnan(hit) else ("yes" if hit >= 0.5 else "no" if hit == 0 else f"tie {hit:.2f}")})
    photo = engine.crop(features["analysis_rgb"], box)
    result = MODEL.analyze(changed)
    return photo, tiles, truth, box, pd.DataFrame(rows), result


# ---------------------------------------------------------------- two photographs of one eye
def compare(path_a: str | None, path_b: str | None):
    if not path_a or not path_b:
        raise ValueError("Choose both photographs.")
    results = [MODEL.analyze(read_photo(path)) for path in (path_a, path_b)]
    first, second = results
    table = pd.DataFrame([{"Measurement": DISPLAY_NAMES[name], "A": round(float(first["features"][name]), 3),
                           "B": round(float(second["features"][name]), 3),
                           "B − A": round(float(second["features"][name] - first["features"][name]), 3)} for name in FEATURES])
    return results, charts.dumbbell(first["features"], second["features"]), charts.contribution_difference(first, second), table


# ---------------------------------------------------------------- batch scoring, phenotypes, cohort profiles
def score_batch(paths: list[str], cohort_text: str, progress=None):
    """Score photographs; ``cohort_text`` is one label for all, or one comma/newline-separated label per photograph."""
    if not paths:
        raise ValueError("Choose one or more photographs.")
    if len(paths) > MAX_BATCH:
        raise ValueError(f"At most {MAX_BATCH} photographs per run.")
    labels = [part.strip() for part in (cohort_text or "").replace("\n", ",").split(",") if part.strip()]
    if len(labels) not in (0, 1, len(paths)):
        raise ValueError(f"Give one cohort label for all photographs, or exactly {len(paths)} labels.")
    labels = labels or ["upload"]
    labels = labels * len(paths) if len(labels) == 1 else labels
    rows = []
    steps = zip(paths, labels) if progress is None else progress.tqdm(list(zip(paths, labels)), desc="Scoring")
    for path, cohort in steps:
        row = {"file": Path(path).name, "cohort": cohort}
        try:
            result = MODEL.analyze(read_photo(path))
            row.update(risk=round(result["risk_bad"], 4), verdict=result["verdict"],
                       phenotype=quality_phenotype(result["axis_contributions"]) if result["verdict"] != "PASS" else "",
                       **{name: round(float(result["features"][name]), 4) for name in FEATURES}, error="")
        except Exception as problem:  # one unreadable file must not stop the batch
            row.update(error=str(problem)[:120])
        rows.append(row)
    table = pd.DataFrame(rows)
    scored = table[table["error"] == ""] if "risk" in table else table.iloc[0:0]
    if scored.empty:
        raise ValueError("None of the photographs could be scored.")
    shown = table.assign(phenotype=table["phenotype"].map(lambda name: DISPLAY_NAMES.get(name, name)))
    shown = shown.rename(columns={**DISPLAY_NAMES, "file": "Photograph", "cohort": "Cohort", "risk": "Poor-quality risk", "verdict": "Decision",
                                  "phenotype": "Phenotype", "error": "Problem"})
    if not shown["Problem"].astype(bool).any():
        shown = shown.drop(columns="Problem")
    flagged = scored[scored["verdict"] != "PASS"]
    phenotype_chart = (charts.phenotypes(phenotype_composition(scored)) if not flagged.empty
                       else charts.empty("Every photograph passed, so there are no quality phenotypes to show."))
    counts = scored["verdict"].value_counts()
    return (shown, _save_csv(table, "eyeqc_batch"), phenotype_chart, charts.profiles(cohort_profiles(scored)),
            {"photographs": len(scored), **{verdict: int(counts.get(verdict, 0)) for verdict in ("PASS", "REVIEW", "FAIL")}})


# ---------------------------------------------------------------- batch variation and harmonization
def harmonize(csv_path: str, method: str, grade_column: str = ""):
    """Separability of cohorts from the ten measurements, before and after harmonization (CSV from the cohort tab)."""
    frame = _read_table(csv_path, ("cohort", *FEATURES)).dropna(subset=["cohort", *FEATURES]).copy()
    frame["cohort"] = frame["cohort"].astype(str)
    counts = frame["cohort"].value_counts()
    if len(counts) < 2 or counts.min() < 5:
        raise ValueError("Need at least two cohorts with at least five photographs each.")
    test = separability_permutation_test(frame, n_permutations=100)
    combat = apply_combat(frame, fit_combat(frame))
    location_scale = robust_location_scale_harmonize(frame, frame)
    versions = {"uncorrected": frame, "location-scale": location_scale, "ComBat": combat}
    separability = pd.DataFrame([{"version": name, "balanced_accuracy": domain_separability(data)} for name, data in versions.items()])
    chance = float(np.mean(test["null_distribution"]))
    chosen = combat if method == "ComBat" else location_scale
    shift = (cohort_profiles(chosen) - cohort_profiles(frame)).abs().to_numpy().max()
    note = (f"The ten measurements predict the cohort with balanced accuracy **{test['observed_separability']:.2f}** "
            f"(chance by permutation {chance:.2f}, p = {test['p_value']:.3f}). Largest change of a cohort median after {method}: **{shift:.2f}**. "
            "Corrections are fitted and checked on the same rows here. Harmonization can remove real quality differences, so check that "
            "defect responses survive before relying on corrected values; EyeQC scores always use the uncorrected measurements.")
    association = charts.empty("Add a grade column (for example disease severity) and name it to see its association with the measurements.", 160)
    grade_column = (grade_column or "").strip()
    if grade_column:
        if grade_column not in frame.columns:
            raise ValueError(f"The CSV has no column named '{grade_column}'.")
        graded = frame.dropna(subset=[grade_column])
        if graded[grade_column].nunique() < 2:
            raise ValueError("The grade column needs at least two different values.")
        found = pathology_quality_disentanglement(graded, grade_column, batch_column="cohort", n_permutations=500)
        association = charts.association(found["entanglement"], f"η² with '{grade_column}' (cohort-centred)")
    return (note, charts.separability(separability, chance), charts.profiles(cohort_profiles(frame), "Cohort profiles: uncorrected"),
            charts.profiles(cohort_profiles(chosen), f"Cohort profiles: after {method}"),
            charts.variance(variance_decomposition(frame, inner=None)), association, _save_csv(chosen, "eyeqc_harmonized"))


# ---------------------------------------------------------------- local validation and recalibration
METRIC_NAMES = {"ECE": "Expected calibration error", "Brier": "Brier score", "false_pass_rate": "Poor photographs passed",
                "pass_share": "All photographs passed", "pass_npv": "PASS negative predictive value", "fail_sensitivity": "FAIL sensitivity",
                "review_share": "REVIEW share", "AUROC": "AUROC"}


def _labelled_table(csv_path: str):
    frame = _read_table(csv_path, ("risk", "is_poor")).dropna(subset=["risk", "is_poor"]).reset_index(drop=True)
    risk, labels = frame.risk.to_numpy(float), frame.is_poor.to_numpy(float)
    if not np.isin(labels, (0, 1)).all() or np.unique(labels).size < 2:
        raise ValueError("is_poor must be 1 (poor) or 0 (acceptable), with both values present.")
    if ((risk < 0) | (risk > 1)).any():
        raise ValueError("risk must lie between 0 and 1.")
    groups = (frame.participant if "participant" in frame.columns else pd.Series(frame.index)).astype(str).to_numpy()
    return risk, labels.astype(int), groups


def validate(csv_path: str):
    """Discrimination, calibration and operating performance of the frozen thresholds on locally labelled photographs."""
    risk, labels, groups = _labelled_table(csv_path)
    if len(risk) < 20:
        raise ValueError("Need at least 20 labelled photographs.")
    frame = pd.DataFrame({"gt_bad": labels, "risk_bad": risk, "cluster_id": groups})
    point = performance(frame, PASS_THRESHOLD, FAIL_THRESHOLD)
    intervals = cluster_bootstrap_intervals(frame, PASS_THRESHOLD, FAIL_THRESHOLD, bootstraps=500)
    calibration = operating_metrics(labels, risk)

    def row(name, value, key=None, invert=False):
        interval = ""
        if key in intervals:
            low, high = intervals[key]
            low, high = (1 - high, 1 - low) if invert else (low, high)
            interval = f"{low:.3f} to {high:.3f}"
        return {"Metric": name, "Value": round(float(value), 3), "95% CI": interval}

    table = pd.DataFrame([row("AUROC", point["AUROC"], "AUROC"), row("AUPRC", point["AUPRC"], "AUPRC"), row("Brier score", point["Brier"]),
                          row("Expected calibration error", calibration["ECE"]), row("PASS negative predictive value", point["PASS_NPV"], "PASS_NPV"),
                          row("Poor photographs passed", 1 - point["poor_capture_nonPASS"], "poor_capture_nonPASS", invert=True),
                          row("All photographs passed", point["PASS_coverage"], "PASS_coverage"),
                          row("FAIL sensitivity", point["FAIL_sensitivity"]), row("FAIL specificity", point["FAIL_specificity"]),
                          row("REVIEW share", point["REVIEW_fraction"])])
    note = (f"**{len(risk):,}** photographs ({int(labels.sum()):,} poor), frozen thresholds. Intervals: 500 bootstrap resamples of "
            f"{'participants' if len(np.unique(groups)) < len(risk) else 'photographs'}.")
    return note, table, charts.reliability({"EyeQC (frozen)": (risk, labels)})


def recalibrate(csv_path: str, labelled: int, repeats: int = RECALIBRATION_SPLITS, seed: int = 20261024):
    """As in the study: draw whole participants into a labelled set of at most ``labelled`` photographs, fit the correction
    there, evaluate on everyone else, repeat, and report medians with the central 95% of the splits."""
    risk, labels, groups = _labelled_table(csv_path)
    splits = []
    for repeat in range(repeats):
        fit_rows = labelled_participants(groups, labelled, np.random.default_rng(seed + repeat))
        held = ~fit_rows
        if fit_rows.sum() < 10 or held.sum() < 10 or np.unique(labels[fit_rows]).size < 2:
            continue
        mapping = fit_recalibration(risk[fit_rows], labels[fit_rows])
        cut = local_pass_threshold(risk[fit_rows], labels[fit_rows])
        passed = risk[held] <= cut
        poor = labels[held] == 1
        splits.append(dict(mapping=mapping, held=held, fitted=int(fit_rows.sum()), cut=cut,
                           before=operating_metrics(labels[held], risk[held]), after=operating_metrics(labels[held], mapping(risk[held])),
                           threshold={"false_pass_rate": passed[poor].mean() if poor.any() else np.nan, "pass_share": passed.mean()}))
    if not splits:
        raise ValueError("Need at least 10 labelled photographs containing both labels, and at least 10 held-out photographs.")
    before = pd.DataFrame([split["before"] for split in splits])
    after = pd.DataFrame([split["after"] for split in splits])
    cut_only = pd.DataFrame([split["threshold"] for split in splits])

    def spread(values):
        return f"{values.quantile(0.025):.3f} to {values.quantile(0.975):.3f}" if values.notna().any() else ""

    table = pd.DataFrame([{"Metric": name, "Frozen": round(before[key].median(), 3), "Recalibrated": round(after[key].median(), 3),
                           "Recalibrated: 95% of splits": spread(after[key]),
                           "Local PASS cut-off only": f"{cut_only[key].median():.3f}" if key in cut_only else "–"}
                          for key, name in METRIC_NAMES.items()])
    # a reliability curve shows one split: the one whose corrected calibration error is the median
    typical = splits[int(np.argsort(after["ECE"].to_numpy(), kind="stable")[len(splits) // 2])]
    mapping, held = typical["mapping"], typical["held"]
    curve = charts.reliability({"frozen": (risk[held], labels[held]), "recalibrated": (mapping(risk[held]), labels[held])},
                               "Held-out photographs, median split")
    boxes = charts.split_spread(before, after, {"ECE": "Calibration error", "false_pass_rate": "Poor passed", "pass_share": "All passed"})
    sizes = sorted({split["fitted"] for split in splits})
    fitted = f"{sizes[0]}" if len(sizes) == 1 else f"{sizes[0]} to {sizes[-1]}"
    cut = float(np.median([split["cut"] for split in splits]))
    note = (f"**{len(splits)}** random splits keeping participants separate, each fitted on **{fitted}** labelled photographs and evaluated "
            f"on the others; the table gives medians. Correction in the median split: risk' = sigmoid({mapping.intercept:.3f} + "
            f"{mapping.slope:.3f} · logit(risk)). The exploratory *PASS cut-off only* option keeps the frozen risk and passes only below a "
            f"locally chosen value (median {cut:.3f}) that holds back 90% of the labelled poor photographs. With few labels, a single split "
            "can differ widely from the median.")
    return note, table, curve, boxes


# ---------------------------------------------------------------- reader agreement
EYEQC_COLUMNS = ("eyeqc", "EyeQC", "verdict", "Decision")
RISK_COLUMNS = ("risk", "Poor-quality risk", "eyeqc_risk")
PARTICIPANT_COLUMNS = ("participant", "participant_id", "cluster_id")


def readers(csv_path: str, grader_text: str = ""):
    """Agreement between graders and EyeQC for the user's own reader study."""
    frame = _read_table(csv_path)
    model_column = next((c for c in EYEQC_COLUMNS if c in frame.columns), None)
    if model_column is None:
        raise ValueError("The CSV needs a column 'eyeqc' (or 'verdict') with EyeQC's PASS / REVIEW / FAIL.")
    risk_column = next((c for c in RISK_COLUMNS if c in frame.columns), None)
    group_column = next((c for c in PARTICIPANT_COLUMNS if c in frame.columns), None)
    graders = [part.strip() for part in (grader_text or "").split(",") if part.strip()]
    if not graders:
        for column in frame.columns:
            if column in (model_column, risk_column, group_column, *FEATURES):
                continue
            try:
                to_codes(frame[column])
                graders.append(column)
            except ValueError:
                pass
    missing = [c for c in graders if c not in frame.columns]
    if missing:
        raise ValueError("No such column: " + ", ".join(missing))
    if len(graders) < 2:
        raise ValueError("Need at least two grader columns containing PASS / REVIEW / FAIL.")
    report = reader_report(frame[graders], frame[model_column], risk=None if risk_column is None else frame[risk_column],
                           participants=None if group_column is None else frame[group_column])
    per = report["per_grader"]
    shown = per.rename(columns={"grader": "Grader", "photographs": "Photographs", "kappa_with_EyeQC": "κ with EyeQC",
                                "mean_kappa_with_other_graders": "Mean κ with other graders", "difference": "Difference",
                                "ci_low": "CI low", "ci_high": "CI high", "equivalent_within_margin": "Equivalent (±0.15)",
                                "spearman_risk_vs_category": "Spearman ρ, risk vs category", "auroc_for_grader_FAIL": "AUROC for FAIL calls"}).round(3)
    shown["Equivalent (±0.15)"] = np.where(shown["Equivalent (±0.15)"], "yes", "no")
    group = report["group"].rename(columns={"group": "Raters", "raters": "Number", "fleiss_kappa": "Fleiss κ",
                                            "krippendorff_alpha_ordinal": "Krippendorff α (ordinal)", "icc_3_1": "ICC(3,1)",
                                            "icc_3_k": "ICC(3,k)", "unanimous_share": "Unanimous"}).round(3)
    note = (f"**{len(frame):,}** photographs, graders: {', '.join(graders)}. Quadratic-weighted κ on FAIL < REVIEW < PASS. "
            "Differences use a paired bootstrap (2,000 resamples); equivalence needs the whole 95% interval inside ±0.15.")
    return (note, charts.kappa_matrix(report["kappa_matrix"]), shown, charts.kappa_difference(per, EQUIVALENCE_MARGIN), group,
            _save_csv(report["per_grader"], "eyeqc_reader_agreement"))


# ---------------------------------------------------------------- CSV templates (illustrative values, clearly not data)
def template(kind: str) -> str:
    if kind == "labels":
        frame = pd.DataFrame({"participant": ["p001", "p001", "p002"], "risk": [0.04, 0.37, 0.85], "is_poor": [0, 0, 1]})
    elif kind == "readers":
        frame = pd.DataFrame({"participant": ["p001", "p002", "p003"], "grader_1": ["PASS", "REVIEW", "FAIL"],
                              "grader_2": ["PASS", "PASS", "FAIL"], "eyeqc": ["PASS", "REVIEW", "FAIL"], "risk": [0.04, 0.37, 0.85]})
    else:
        frame = pd.DataFrame([{"cohort": name, **{f: 1.0 for f in FEATURES}, "grade": 0} for name in ("camera_A", "camera_B")])
    return _save_csv(frame, f"eyeqc_template_{kind}")
