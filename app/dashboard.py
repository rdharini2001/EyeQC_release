"""Study results: charts and tables read from the aggregate files in ``results/`` (nothing is computed from uploads)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from eyeqc.core import DISPLAY_NAMES, FEATURES

import charts

HERE = Path(__file__).resolve().parent
RESULTS = next((path for path in (HERE / "results", HERE.parent / "results") if path.is_dir()), HERE / "results")
COHORTS = ["EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD"]
READER_SUMMARY = {"eyeqc_kappa": "0.70–0.78", "grader_kappa": "0.77–0.80", "auroc": "0.95–0.97", "equivalence": "1 of 3"}   # as published
METHODS = {"quickqual": "QuickQual", "retfound": "RETFound", "automorph": "AutoMorph", "efiqacp": "EFIQA-CP", "fthnet": "FTHNet",
           "hyperiqa": "HyperIQA", "brisque": "BRISQUE (adapted)", "mcfnet": "MCF-Net (adapted)"}
ABLATION = {"v3_hybrid": "EyeQC (measurements + DINOv2)", "v2_dinov2_only": "DINOv2 only", "v1_handcrafted_only": "Measurements only"}
DEFECTS = {"glare": "Glare", "shadow": "Shadow", "local_blur": "Local blur"}


def _read(name: str, **options) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"{name}.csv", **options)


def tiles(items: list[tuple[str, str, str]]) -> str:
    """Headline numbers: (value, label, context)."""
    cells = "".join(f"<div class='eq-kpi'><b>{value}</b><span>{label}</span><i>{context}</i></div>" for value, label, context in items)
    return f"<div class='eq-kpis'>{cells}</div>"


def headline() -> str:
    cohorts = _read("cohort_validation")
    latency = _read("throughput").set_index("metric")["value"]
    return tiles([
        (f"{int(cohorts.N.sum()):,}", "photographs evaluated", "five cohorts, one frozen model"),
        (f"{cohorts.AUROC.min():.3f}–{cohorts.AUROC.max():.3f}", "AUROC", "lowest under distribution shift (DeepDRiD)"),
        (READER_SUMMARY["eyeqc_kappa"], "weighted κ with graders", "80 photographs, 3 ophthalmic graders"),
        (f"{latency['latency_p50_ms']:.0f} ms", "median CPU time", f"99th percentile {latency['latency_p99_ms']:.0f} ms"),
    ])


# ---------------------------------------------------------------- validation
def validation():
    table = _read("cohort_validation")
    table["poor_passed"] = 1 - table["poor_capture_nonPASS"]
    forest = charts.auroc_forest(table)
    bars = charts.cohort_bars(table, {"PASS_NPV": "PASS negative predictive value", "FAIL_sensitivity": "FAIL sensitivity",
                                      "poor_passed": "Poor photographs passed"}, "Fixed thresholds")
    shown = pd.DataFrame({
        "Cohort": table.cohort, "Photographs": table.N, "Poor": table.poor_N,
        "AUROC (95% CI)": [f"{r.AUROC:.3f} ({r.AUROC_lo:.3f}–{r.AUROC_hi:.3f})" for r in table.itertuples()],
        "AUPRC": table.AUPRC.round(3), "Brier": table.Brier.round(3), "PASS NPV": table.PASS_NPV.round(3),
        "Poor passed": table.poor_passed.round(3), "FAIL sensitivity": table.FAIL_sensitivity.round(3), "REVIEW share": table.REVIEW_fraction.round(3)})
    return forest, bars, shown


def representation():
    ablation = _read("ablation").assign(method=lambda d: d.model.map(ABLATION))
    order = list(ABLATION.values())
    auroc = charts.method_dots(ablation, "AUROC", "What each representation contributes: AUROC", order, order[0], "AUROC", (0.55, 1.02))
    brier = charts.method_dots(ablation, "Brier", "Probability error (lower is better)", order, order[0], "Brier score")
    probes = _read("concept_probes").sort_values("mean_held_out_r2_fullscale", ascending=False)
    recovered = charts.horizontal(probes.display_name.tolist(), probes.mean_held_out_r2_fullscale.tolist(),
                                  "How much of each measurement DINOv2 already encodes", "held-out R² of a linear probe")
    return auroc, brier, recovered


def comparisons():
    columns = {"EyeQC": _read("cohort_validation").set_index("cohort").AUROC}
    for key, name in METHODS.items():
        columns[name] = _read(f"baseline_{key}").set_index("cohort").AUROC
    table = pd.DataFrame(columns).reindex(COHORTS).T
    return charts.matrix_heatmap(table, "AUROC of EyeQC and available comparison methods", 0.5, 1.0, "{:.3f}")


# ---------------------------------------------------------------- readers
def reader_study() -> str:
    """The reader study used UK Biobank photographs (restricted access): only the published summary is shown."""
    cards = tiles([(READER_SUMMARY["eyeqc_kappa"], "EyeQC vs each grader", "quadratic-weighted κ"),
                   (READER_SUMMARY["grader_kappa"], "grader vs grader", "quadratic-weighted κ"),
                   (READER_SUMMARY["auroc"], "AUROC for a grader's FAIL calls", "from the EyeQC risk"),
                   (READER_SUMMARY["equivalence"], "graders meeting equivalence", "margin ±0.15")])
    return (cards + "<div class='eq-card'><p><b>Restricted access.</b> The 80 photographs and the graders' assessments come from UK Biobank. "
            "No photographs, participant identifiers, individual grades or result tables from this study are distributed here; access to UK "
            "Biobank data is by application to UK Biobank. The <b>Reader agreement</b> tab computes the same statistics for your own graders.</p></div>")


# ---------------------------------------------------------------- shift, recalibration, same eye
def shift():
    table = _read("local_recalibration").query("cohort == 'DeepDRiD'")
    error = charts.budget_curves(table, "ECE", "DeepDRiD calibration error", "expected calibration error")
    passed = charts.budget_curves(table, "false_PASS_rate", "Poor photographs passed", "share of poor photographs")
    pairs = _read("same_eye_pairs").set_index("quantity")["value"]
    strip = tiles([
        (f"{int(pairs['pairs'])}", "same-eye pairs", "eyes with exactly two photographs"),
        (f"{pairs['same_category']:.1%}", "same decision", f"opposite PASS/FAIL in {pairs['opposite_PASS_FAIL']:.1%}"),
        (f"{pairs['deepdrid_artifact_higher_risk_for_worse']:.1%}", "higher risk for worse artifacts",
         f"{int(pairs['deepdrid_artifact_pairs'])} pairs that differ"),
        (f"{pairs['deepdrid_clarity_higher_risk_for_worse']:.1%} · {pairs['deepdrid_field_definition_higher_risk_for_worse']:.1%}",
         "same for clarity and field", "pairs that differ"),
    ])
    return error, passed, strip


# ---------------------------------------------------------------- degradations and maps
def defects():
    response = _read("degradation_response").sort_values("monotonic_fraction", ascending=False)
    names = ["JPEG compression" if k == "jpeg" else k.replace("_", " ").capitalize() for k in response.degradation]
    monotonic = charts.horizontal(names, response.monotonic_fraction.tolist(), "Risk never decreases as severity increases",
                                  "share of 2,294 photographs", percent=True)
    maps = _read("map_localization").query("dose == 1.0")
    rows = {"exposure": "Exposure", "detail": "Detail", "quality": "Combined", "contrast": "Contrast"}

    def grid(frame, value, title, low, high, fmt):
        table = frame.pivot_table(index="map", columns="degradation", values=value, aggfunc="mean").reindex(index=list(rows), columns=list(DEFECTS))
        return charts.matrix_heatmap(table.rename(index=rows, columns=DEFECTS), title, low, high, fmt)

    dense = grid(maps.query("readout == 'window_8' and half == 'held_out'"), "auc", "Dense maps: pixel AUC (held-out photographs)", 0.45, 1.0, "{:.2f}")
    tiled = grid(maps.query("readout == 'tile'"), "point_hit", "6 × 6 maps: lowest tile inside the defect (chance ≈ 22%)", 0.0, 1.0, "{:.0%}")
    return monotonic, dense, tiled


def cohort_variation():
    percent = _read("quality_phenotypes_percent", index_col=0)
    phenotype = charts.phenotypes(percent.reindex(columns=COHORTS).T / 100)
    separability = _read("cohort_separability")
    names = {"raw": "uncorrected", "location_scale": "location-scale", "combat": "ComBat"}
    bars = charts.separability(pd.DataFrame({"version": separability.feature_set.map(names),
                                             "balanced_accuracy": separability.cohort_separability_fullscale}), 0.2)
    retention = _read("harmonization_retention").sort_values("dose_response_retention", ascending=False)
    kept = charts.horizontal([k.replace("_", " ").capitalize() for k in retention.degradation], retention.dose_response_retention.tolist(),
                             "Defect response kept after ComBat (criterion ≥ 90%)", "share of original median response", percent=True)
    disease = charts.association(_read("disease_association"), "η² with diabetic retinopathy grade (DeepDRiD)")
    return phenotype, bars, kept, disease


def speed() -> str:
    latency = _read("throughput").set_index("metric")["value"]
    return tiles([(f"{latency['latency_p50_ms']:.0f} ms", "median per photograph", "complete CPU pipeline"),
                  (f"{latency['latency_p99_ms']:.0f} ms", "99th percentile", f"{int(latency['latency_photographs'])} photographs timed"),
                  (f"{latency['images_per_second_1_workers']:.2f} /s", "throughput, 1 worker", ""),
                  (f"{latency['images_per_second_8_workers']:.2f} /s", "throughput, 8 workers", "")])


def order_check() -> None:
    """Fail early if a measurement name in the result files is unknown."""
    unknown = set(_read("concept_probes").feature) - set(FEATURES)
    if unknown:
        raise ValueError(f"unknown measurements in results: {sorted(unknown)} (known: {list(DISPLAY_NAMES)})")
