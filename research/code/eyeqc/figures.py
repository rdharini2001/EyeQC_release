"""Publication and GUI figures using one Nature-inspired colour system."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .core import DISPLAY_NAMES, FEATURES
from .experiments import tile_quality_map
from .pipeline import read_rgb


NATURE = {
    "red": "#E64B35",
    "blue": "#4DBBD5",
    "green": "#00A087",
    "navy": "#3C5488",
    "salmon": "#F39B7F",
    "slate": "#8491B4",
    "mint": "#91D1C2",
    "dark_red": "#DC0000",
    "brown": "#7E6148",
    "sand": "#B09C85",
    "ink": "#252525",
    "light": "#F6F7F8",
}

COHORT_COLORS = [NATURE["navy"], NATURE["blue"], NATURE["green"], NATURE["red"], NATURE["brown"]]
ROUTE_COLORS = {"PASS": NATURE["green"], "REVIEW": "#E6A700", "FAIL": NATURE["red"]}


def set_nature_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "font.family": "Arial",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "grid.color": "#D9DDE1",
            "grid.linewidth": 0.6,
        }
    )


def save_figure(figure: plt.Figure, path_without_suffix) -> None:
    figure.savefig(f"{path_without_suffix}.png", bbox_inches="tight", facecolor="white")
    figure.savefig(f"{path_without_suffix}.pdf", bbox_inches="tight", facecolor="white")


def external_validation_figure(performance: pd.DataFrame, scored: pd.DataFrame) -> plt.Figure:
    set_nature_style()
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    cohorts = performance.sort_values("AUROC").reset_index(drop=True)
    y = np.arange(len(cohorts))

    axes[0, 0].errorbar(
        cohorts["AUROC"],
        y,
        xerr=[cohorts["AUROC"] - cohorts["AUROC_lo"], cohorts["AUROC_hi"] - cohorts["AUROC"]],
        fmt="o",
        color=NATURE["navy"],
        ecolor=NATURE["slate"],
        capsize=3,
    )
    axes[0, 0].set(yticks=y, yticklabels=cohorts["cohort"], xlim=(0.45, 1.01), xlabel="AUROC (95% CI)")
    axes[0, 0].axvline(0.5, color="#999999", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("a  Multi-cohort discrimination", loc="left", fontweight="bold")

    for color, (cohort, group) in zip(COHORT_COLORS, scored.groupby("cohort")):
        ordered = group.sort_values("risk_bad")
        bins = pd.qcut(ordered["risk_bad"], 8, duplicates="drop")
        calibration = ordered.groupby(bins, observed=True).agg(predicted=("risk_bad", "mean"), observed=("gt_bad", "mean"))
        axes[0, 1].plot(calibration["predicted"], calibration["observed"], "o-", color=color, label=cohort, linewidth=1.2, markersize=4)
    axes[0, 1].plot([0, 1], [0, 1], "--", color="#888888", linewidth=0.8)
    axes[0, 1].set(xlabel="Predicted poor-quality risk", ylabel="Observed poor-quality fraction")
    axes[0, 1].set_title("b  Calibration transport", loc="left", fontweight="bold")
    axes[0, 1].legend(ncol=2)

    disposition = scored.groupby(["cohort", "verdict"]).size().unstack(fill_value=0)
    disposition = disposition.div(disposition.sum(axis=1), axis=0)[["PASS", "REVIEW", "FAIL"]]
    bottom = np.zeros(len(disposition))
    for route in ("PASS", "REVIEW", "FAIL"):
        axes[1, 0].bar(disposition.index, disposition[route] * 100, bottom=bottom * 100, color=ROUTE_COLORS[route], label=route)
        bottom += disposition[route].to_numpy()
    axes[1, 0].set(ylabel="Images (%)")
    axes[1, 0].tick_params(axis="x", rotation=25)
    axes[1, 0].set_title("c  Frozen workflow disposition", loc="left", fontweight="bold")
    axes[1, 0].legend(ncol=3)

    fingerprints = scored.groupby("cohort")[list(FEATURES)].median()
    image = axes[1, 1].imshow(fingerprints, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    axes[1, 1].set(yticks=np.arange(len(fingerprints)), yticklabels=fingerprints.index)
    axes[1, 1].set(xticks=np.arange(len(FEATURES)), xticklabels=[DISPLAY_NAMES[name] for name in FEATURES])
    axes[1, 1].tick_params(axis="x", rotation=55)
    axes[1, 1].set_title("d  Cohort quality fingerprints", loc="left", fontweight="bold")
    figure.colorbar(image, ax=axes[1, 1], fraction=0.045, label="Median axis score")

    figure.tight_layout()
    return figure


def degradation_figure(results: pd.DataFrame) -> plt.Figure:
    set_nature_style()
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    summary = results.groupby(["degradation", "dose"])["risk_bad"].agg(["median", "sem"]).reset_index()
    for color, (kind, group) in zip(COHORT_COLORS + [NATURE["salmon"], NATURE["mint"], NATURE["dark_red"], NATURE["sand"]], summary.groupby("degradation")):
        axes[0].plot(group["dose"], group["median"], "o-", color=color, label=kind.replace("_", " "), linewidth=1.4, markersize=3.5)
    axes[0].set(xlabel="Degradation dose", ylabel="Median poor-quality risk")
    axes[0].set_title("a  Large-scale dose response", loc="left", fontweight="bold")
    axes[0].legend(ncol=2, fontsize=7)

    delta = results.copy()
    baseline = delta[delta["dose"].eq(0)].groupby(["image_id", "degradation"])[list(FEATURES)].first()
    severe = delta[delta["dose"].eq(1)].set_index(["image_id", "degradation"])
    effects = (severe[list(FEATURES)] - baseline).groupby("degradation").median()
    image = axes[1].imshow(effects, aspect="auto", cmap="coolwarm", vmin=-0.5, vmax=0.5)
    axes[1].set(yticks=np.arange(len(effects)), yticklabels=effects.index.str.replace("_", " "))
    axes[1].set(xticks=np.arange(len(FEATURES)), xticklabels=[DISPLAY_NAMES[name] for name in FEATURES])
    axes[1].tick_params(axis="x", rotation=60)
    axes[1].set_title("b  Mechanistic axis specificity", loc="left", fontweight="bold")
    figure.colorbar(image, ax=axes[1], fraction=0.046, label="Median axis change")

    localized = results.dropna(subset=["localization_point_hit"])
    localization = localized.groupby(["degradation", "dose"])["localization_point_hit"].mean().reset_index()
    for color, (kind, group) in zip((NATURE["red"], NATURE["blue"], NATURE["navy"]), localization.groupby("degradation")):
        axes[2].plot(group["dose"], group["localization_point_hit"] * 100, "o-", color=color, label=kind.replace("_", " "))
    axes[2].axhline(20, linestyle="--", color="#999999", linewidth=0.8, label="chance")
    axes[2].set(xlabel="Degradation dose", ylabel="Defect-localization hit rate (%)", ylim=(0, 105))
    axes[2].set_title("c  Quantitative map validation", loc="left", fontweight="bold")
    axes[2].legend()
    figure.tight_layout()
    return figure


def quality_radar_figure(analysis: dict) -> plt.Figure:
    """Polar 'quality fingerprint' of the ten axes for one photograph.

    A radar shape is easier for a clinician to pattern-match at a glance across
    repeat visits than a ranked bar list: a symmetric, near-circular fingerprint
    reads as a technically uniform capture, while a fingerprint with a deep
    notch on one axis reads as a single dominant acquisition problem.
    """
    set_nature_style()
    values = [analysis["features"][name] for name in FEATURES]
    angles = np.linspace(0, 2 * np.pi, len(FEATURES), endpoint=False).tolist()
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]

    figure = plt.figure(figsize=(4.6, 4.6))
    axis = figure.add_subplot(111, polar=True)
    axis.plot(angles_closed, values_closed, color=NATURE["navy"], linewidth=1.8)
    axis.fill(angles_closed, values_closed, color=NATURE["blue"], alpha=0.25)
    axis.plot(angles_closed, [0.45] * len(angles_closed), "--", color=NATURE["red"], linewidth=1.0, label="Typical concern level")
    axis.set(xticks=angles, xticklabels=[DISPLAY_NAMES[name] for name in FEATURES], ylim=(0, 1))
    axis.set_yticklabels([])
    axis.tick_params(axis="x", pad=8, labelsize=7.5)
    axis.set_title(f"Quality fingerprint · {analysis['verdict']}", fontsize=10, fontweight="bold", pad=20)
    axis.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=7)
    figure.tight_layout()
    return figure


def risk_gauge_figure(analysis: dict, pass_threshold: float, fail_threshold: float) -> plt.Figure:
    """Semicircular risk gauge against the frozen PASS/REVIEW/FAIL operating bands."""
    set_nature_style()
    figure = plt.figure(figsize=(4.6, 2.9))
    axis = figure.add_subplot(111, polar=True)
    axis.set_theta_zero_location("W")
    axis.set_theta_direction(-1)
    axis.set_thetamin(0)
    axis.set_thetamax(180)
    axis.set_ylim(0, 1)
    axis.set_yticks([])
    axis.set_xticks([])
    axis.spines["polar"].set_visible(False)
    axis.grid(False)

    bands = (
        (0.0, pass_threshold, NATURE["green"]),
        (pass_threshold, fail_threshold, "#E6A700"),
        (fail_threshold, 1.0, NATURE["red"]),
    )
    for low, high, color in bands:
        theta = np.linspace(low * np.pi, high * np.pi, 40)
        axis.fill_between(theta, 0, 1, color=color, alpha=0.85, linewidth=0)

    needle = float(np.clip(analysis["risk_bad"], 0, 1)) * np.pi
    axis.plot([needle, needle], [0, 1], color=NATURE["ink"], linewidth=3)
    axis.scatter([needle], [0.03], color=NATURE["ink"], s=45, zorder=5)
    axis.set_title(f"{analysis['verdict']} · poor-quality risk {analysis['risk_bad']:.2f}", fontsize=10, fontweight="bold", pad=10)
    figure.tight_layout()
    return figure


def batch_dashboard_figure(batch_table: pd.DataFrame, throughput: dict) -> plt.Figure:
    """Clinic-session summary: disposition, dominant failure axis, and measured throughput.

    The throughput panel follows the translational pattern used by high-throughput
    digital-pathology tooling (for example PatchSorter): report the measured
    images-per-second rate as a first-class deployment figure, not a footnote.
    """
    set_nature_style()
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))

    present = [route for route in ("PASS", "REVIEW", "FAIL", "ERROR") if route in batch_table["verdict"].unique()]
    counts = batch_table["verdict"].value_counts().reindex(present, fill_value=0)
    colors = [ROUTE_COLORS.get(route, NATURE["slate"]) for route in present]
    axes[0].bar(counts.index, counts.values, color=colors)
    for index, value in enumerate(counts.values):
        axes[0].text(index, value, str(int(value)), ha="center", va="bottom", fontsize=9)
    axes[0].set(ylabel="Images", title="")
    axes[0].set_title("a  Workflow disposition", loc="left", fontweight="bold")

    non_pass = batch_table[batch_table["verdict"].isin(("REVIEW", "FAIL")) & batch_table["weakest_axis"].notna()]
    if len(non_pass):
        axis_share = non_pass["weakest_axis"].value_counts(normalize=True).sort_values() * 100
        axes[1].barh(axis_share.index, axis_share.values, color=NATURE["red"])
        axes[1].set(xlabel="Share of REVIEW/FAIL images (%)")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No REVIEW or FAIL images", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_title("b  Dominant failure axis", loc="left", fontweight="bold")

    rate = throughput.get("images_per_sec", float("nan"))
    axes[2].bar([f"{throughput.get('workers', 1)} worker(s)"], [rate], color=NATURE["navy"], width=0.5)
    axes[2].set(ylabel="Images processed per second")
    if rate == rate:
        axes[2].text(0, rate, f"{rate:.2f}/s", ha="center", va="bottom", fontsize=9)
    axes[2].set_title("c  Measured batch throughput", loc="left", fontweight="bold")

    figure.tight_layout()
    return figure


def batch_quality_strip_figure(batch_table: pd.DataFrame) -> plt.Figure:
    """Group-level box-and-strip view of every axis across an uploaded batch.

    This follows MRIQC's group-report pattern: one box-plus-jittered-strip per
    quality metric across the whole cohort, coloured by outcome, so a clinic can
    scan an entire capture session at once and spot the images (and the axis)
    responsible for outliers rather than reading a table row by row.
    """
    set_nature_style()
    valid = batch_table[batch_table["verdict"].isin(("PASS", "REVIEW", "FAIL")) & batch_table[list(FEATURES)].notna().all(axis=1)]
    figure, axis = plt.subplots(figsize=(11.8, 4.3))
    if valid.empty:
        axis.axis("off")
        axis.text(0.5, 0.5, "No scored images", ha="center", va="center")
        return figure

    rng = np.random.default_rng(20260822)
    positions = np.arange(len(FEATURES))
    axis.boxplot(
        [valid[name].to_numpy() for name in FEATURES],
        positions=positions,
        widths=0.5,
        showfliers=False,
        boxprops={"color": NATURE["slate"]},
        medianprops={"color": NATURE["ink"]},
        whiskerprops={"color": NATURE["slate"]},
    )
    for position, name in zip(positions, FEATURES):
        column = valid[name]
        jitter = position + rng.uniform(-0.18, 0.18, size=len(column))
        colors = valid["verdict"].map(ROUTE_COLORS).fillna(NATURE["slate"])
        axis.scatter(jitter, column.to_numpy(), s=18, c=colors, alpha=0.75, linewidths=0)

    axis.set(xticks=positions, xticklabels=[DISPLAY_NAMES[name] for name in FEATURES], ylim=(-0.02, 1.02), ylabel="Axis score")
    axis.tick_params(axis="x", rotation=25)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=route) for route, color in ROUTE_COLORS.items()]
    axis.legend(handles=handles, ncol=3, loc="lower right", fontsize=8)
    axis.set_title("Batch quality distribution across the ten axes", loc="left", fontweight="bold")
    figure.tight_layout()
    return figure


def site_comparison_figure(
    diagnostic: dict,
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    label_a: str,
    label_b: str,
) -> plt.Figure:
    """Two-batch harmonization diagnostic: fingerprints, root-cause shift, and per-image profiles.

    The third panel is a HistoQC-style parallel-coordinates plot (one faint line
    per image across the ten axes) overlaid with each site's median profile, so
    a systematic scanner or protocol difference is visible as a consistent
    vertical offset rather than only as a single separability number.
    """
    set_nature_style()
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.1))

    fingerprint_a = diagnostic["fingerprint_a"][list(FEATURES)]
    fingerprint_b = diagnostic["fingerprint_b"][list(FEATURES)]
    y = np.arange(len(FEATURES))
    axes[0].barh(y - 0.19, fingerprint_a.values, height=0.38, color=NATURE["navy"], label=label_a)
    axes[0].barh(y + 0.19, fingerprint_b.values, height=0.38, color=NATURE["red"], label=label_b)
    axes[0].set(yticks=y, yticklabels=[DISPLAY_NAMES[name] for name in FEATURES], xlim=(0, 1), xlabel="Median axis score")
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=7)
    axes[0].set_title("a  Site quality fingerprints", loc="left", fontweight="bold")

    shift = diagnostic["axis_shift"]
    peak = shift.abs().max() if len(shift) else 0.0
    colors = [NATURE["red"] if abs(value) == peak else NATURE["slate"] for value in shift.values]
    axes[1].barh([DISPLAY_NAMES[name] for name in shift.index], shift.values, color=colors)
    axes[1].axvline(0, color="#999999", linewidth=0.8)
    axes[1].invert_yaxis()
    axes[1].set(xlim=(-1, 1), xlabel=f"Median axis-score shift\n({label_a} minus {label_b})")
    separability = diagnostic["domain_separability"]
    sep_text = f"{separability:.2f}" if separability == separability else "n/a (too few images)"
    axes[1].set_title(f"b  Root-cause axis shift · separability {sep_text}", loc="left", fontweight="bold")

    x = np.arange(len(FEATURES))
    for _, row in frame_a[list(FEATURES)].iterrows():
        axes[2].plot(x, row.to_numpy(), color=NATURE["navy"], alpha=0.22, linewidth=0.8)
    for _, row in frame_b[list(FEATURES)].iterrows():
        axes[2].plot(x, row.to_numpy(), color=NATURE["red"], alpha=0.22, linewidth=0.8)
    axes[2].plot(x, fingerprint_a.values, color=NATURE["navy"], linewidth=2.6, label=f"{label_a} median")
    axes[2].plot(x, fingerprint_b.values, color=NATURE["red"], linewidth=2.6, label=f"{label_b} median")
    axes[2].set(xticks=x, xticklabels=[DISPLAY_NAMES[name] for name in FEATURES], ylim=(0, 1))
    axes[2].tick_params(axis="x", rotation=55)
    axes[2].legend(fontsize=7)
    axes[2].set_title("c  Per-image quality profiles", loc="left", fontweight="bold")

    figure.tight_layout()
    return figure


def failure_taxonomy_figure(subset: pd.DataFrame, composition: pd.DataFrame, dominant_shift: pd.DataFrame) -> plt.Figure:
    """Failure-mode taxonomy across scanners and institutions.

    Panel a shows where each recurring failure phenotype (from
    :func:`eyeqc.experiments.failure_phenotype_taxonomy`) concentrates by
    cohort and scanner/subcohort. Panel b is a two-component PCA of the same
    axis-contribution space used to cluster the phenotypes, following the
    cohort-level PCA/scatter pattern used by pathology and radiology QC
    dashboards (HistoQC, MRQy) to expose batch structure without a black-box
    embedding. Panel c reports the raw feature-level shift most responsible
    for each cohort's harmonization correction, i.e. a root-cause ranking
    rather than a single aggregate batch-effect score.
    """
    set_nature_style()
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    palette = COHORT_COLORS + [NATURE["salmon"], NATURE["mint"], NATURE["dark_red"], NATURE["sand"], NATURE["slate"]]

    group_column = "subcohort" if "subcohort" in composition else "cohort"
    pivot = composition.pivot_table(index=group_column, columns="failure_phenotype", values="fraction", fill_value=0)
    bottom = np.zeros(len(pivot))
    for color, phenotype in zip(palette, pivot.columns):
        axes[0].bar(pivot.index, pivot[phenotype] * 100, bottom=bottom * 100, color=color, label=DISPLAY_NAMES.get(phenotype, phenotype))
        bottom += pivot[phenotype].to_numpy()
    axes[0].set(ylabel="REVIEW/FAIL images (%)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.02))
    axes[0].set_title("a  Failure phenotype composition", loc="left", fontweight="bold")

    contribution_columns = [f"contribution_{axis}" for axis in FEATURES]
    coordinates = PCA(n_components=2, random_state=20260822).fit_transform(StandardScaler().fit_transform(subset[contribution_columns]))
    projected = subset.assign(pc1=coordinates[:, 0], pc2=coordinates[:, 1])
    for color, (phenotype, group) in zip(palette, projected.groupby("failure_phenotype")):
        axes[1].scatter(group["pc1"], group["pc2"], s=8, alpha=0.45, color=color, label=DISPLAY_NAMES.get(phenotype, phenotype), linewidths=0)
    axes[1].set(xlabel="PC1 of axis contributions", ylabel="PC2 of axis contributions")
    axes[1].set_title("b  Failure-mode structure (PCA)", loc="left", fontweight="bold")

    ranked = dominant_shift.sort_values("dominant_shift_magnitude", key=np.abs)
    labels = ranked["cohort"] + " · " + ranked["dominant_shift_axis"].map(lambda axis: DISPLAY_NAMES.get(axis, axis))
    bar_colors = [NATURE["red"] if value < 0 else NATURE["green"] for value in ranked["dominant_shift_magnitude"]]
    axes[2].barh(labels, ranked["dominant_shift_magnitude"], color=bar_colors)
    axes[2].axvline(0, color="#999999", linewidth=0.8)
    axes[2].set(xlim=(-1, 1), xlabel="Median axis-score shift vs. development reference")
    axes[2].set_title("c  Root cause of each cohort's batch effect", loc="left", fontweight="bold")

    figure.tight_layout()
    return figure


def n_batch_fingerprint_figure(diagnostic: dict) -> plt.Figure:
    """Grouped-bar quality fingerprint across any number of named batches, with IQR error bars."""

    set_nature_style()
    fingerprints = diagnostic["fingerprints"][list(FEATURES)]
    q1 = diagnostic.get("fingerprints_q1")
    q3 = diagnostic.get("fingerprints_q3")
    palette = COHORT_COLORS + [NATURE["salmon"], NATURE["mint"], NATURE["dark_red"], NATURE["sand"]]
    figure, axis = plt.subplots(figsize=(11.0, 4.8))
    x = np.arange(len(FEATURES))
    n = len(fingerprints)
    width = 0.8 / max(n, 1)
    for index, (label, row) in enumerate(fingerprints.iterrows()):
        count = diagnostic["counts"].get(label, 0)
        positions = x + (index - (n - 1) / 2) * width
        yerr = None
        if q1 is not None and q3 is not None:
            lower = (row.values - q1.loc[label, list(FEATURES)].values).clip(min=0)
            upper = (q3.loc[label, list(FEATURES)].values - row.values).clip(min=0)
            yerr = np.vstack([lower, upper])
        axis.bar(positions, row.values, width=width, color=palette[index % len(palette)], label=f"{label} (n={count})")
        if yerr is not None:
            axis.errorbar(positions, row.values, yerr=yerr, fmt="none", ecolor=NATURE["ink"], elinewidth=0.8, capsize=2, alpha=0.6)
    axis.set(xticks=x, xticklabels=[DISPLAY_NAMES[name] for name in FEATURES], ylim=(0, 1.05), ylabel="Median axis score (bars: IQR)")
    axis.tick_params(axis="x", rotation=45)
    separability = diagnostic["domain_separability"]
    p_value = diagnostic.get("domain_separability_p_value", float("nan"))
    if separability == separability:
        p_text = f", permutation p{'<0.002' if p_value == p_value and p_value < 0.002 else f'={p_value:.3f}' if p_value == p_value else ' n/a'}"
        sep_text = f"{separability:.2f}{p_text}"
    else:
        sep_text = "n/a (too few images)"
    axis.set_title(f"Batch quality fingerprints · separability {sep_text}", loc="left", fontweight="bold")
    axis.legend(ncol=min(max(n, 1), 4), fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    figure.tight_layout()
    return figure


def variance_decomposition_figure(decomposition: pd.DataFrame) -> plt.Figure:
    """Inter-batch vs. intra-batch vs. residual variance share per quality axis.

    Reads as a HistoQC/MRQy-style batch-structure panel: an axis with a tall
    navy segment is unsafe to pool across cohorts without correction; a tall
    blue segment means even a single cohort is internally heterogeneous
    (multiple institutions, disease subgroups, or protocol variants).
    """
    set_nature_style()
    ordered = decomposition.sort_values("inter_batch_fraction")
    labels = [DISPLAY_NAMES.get(name, name) for name in ordered["feature"]]
    y = np.arange(len(ordered))

    figure, axis = plt.subplots(figsize=(8.6, 4.6))
    axis.barh(y, ordered["inter_batch_fraction"] * 100, color=NATURE["navy"], label="Inter-batch (cohort)")
    left = ordered["inter_batch_fraction"] * 100
    axis.barh(y, ordered["intra_batch_fraction"] * 100, left=left, color=NATURE["blue"], label="Intra-batch (sub-batch)")
    left = left + ordered["intra_batch_fraction"] * 100
    axis.barh(y, ordered["residual_fraction"] * 100, left=left, color=NATURE["light"], edgecolor=NATURE["slate"], linewidth=0.6, label="Residual (image-level)")
    axis.set(yticks=y, yticklabels=labels, xlim=(0, 100), xlabel="Share of total variance (%)")
    axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14), fontsize=8)
    axis.set_title("Nested batch-effect variance decomposition", loc="left", fontweight="bold")
    figure.tight_layout()
    return figure


def combat_correction_figure(result: dict, batch_column: str = "cohort") -> plt.Figure:
    """Before/after ComBat fingerprints, separability, and the governance verdict."""

    set_nature_style()
    before = result["fingerprints_before"][list(FEATURES)]
    after = result["fingerprints_after"][list(FEATURES)]
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    image_b = axes[0].imshow(before, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    axes[0].set(yticks=np.arange(len(before)), yticklabels=before.index)
    axes[0].set(xticks=np.arange(len(FEATURES)), xticklabels=[DISPLAY_NAMES[name] for name in FEATURES])
    axes[0].tick_params(axis="x", rotation=55)
    axes[0].set_title("a  Fingerprints before correction", loc="left", fontweight="bold")
    figure.colorbar(image_b, ax=axes[0], fraction=0.045)

    image_a = axes[1].imshow(after, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    axes[1].set(yticks=np.arange(len(after)), yticklabels=after.index)
    axes[1].set(xticks=np.arange(len(FEATURES)), xticklabels=[DISPLAY_NAMES[name] for name in FEATURES])
    axes[1].tick_params(axis="x", rotation=55)
    axes[1].set_title("b  Fingerprints after ComBat", loc="left", fontweight="bold")
    figure.colorbar(image_a, ax=axes[1], fraction=0.045)

    before_sep = result["separability_before"]
    after_sep = result["separability_after"]
    before_p = result.get("separability_before_p_value", float("nan"))
    bars = ["Before", "After"]
    values = [before_sep, after_sep]
    colors = [NATURE["red"], NATURE["green"] if result["accepted"] else "#E6A700"]
    axes[2].bar(bars, values, color=colors, width=0.5)
    axes[2].axhline(0.5, color="#999999", linestyle="--", linewidth=0.8, label="chance")
    for index, value in enumerate(values):
        if value == value:
            axes[2].text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    if before_p == before_p:
        p_label = "p<0.01" if before_p < 0.01 else f"p={before_p:.2f}"
        axes[2].text(0, -0.06, f"permutation {p_label}", ha="center", va="top", fontsize=7, color=NATURE["ink"], transform=axes[2].get_xaxis_transform())
    axes[2].set(ylim=(0, 1.05), ylabel="Batch-classifier balanced accuracy")
    verdict = "ACCEPTED" if result["accepted"] else "REJECTED"
    axes[2].set_title(f"c  Separability · {verdict}", loc="left", fontweight="bold")
    axes[2].legend(fontsize=7)

    figure.tight_layout()
    return figure


def disentanglement_figure(diagnostic: dict, caution_threshold: float = 0.10) -> plt.Figure:
    """Axis-wise pathology entanglement ranking plus per-disease-group fingerprints.

    Panel a ranks each quality axis by eta-squared against disease status
    (how much of that axis's variance tracks disease group rather than
    acquisition technique), with a permutation-test significance marker (*)
    on axes whose association is unlikely to be chance (p < 0.05); panel b
    shows the raw per-group fingerprint so a clinician can see, for example,
    that "contrast" or "detail uniformity" runs systematically lower in a
    disease group even on well-acquired images.
    """
    set_nature_style()
    entanglement = diagnostic["entanglement"].sort_values("eta_squared")
    fingerprints = diagnostic["group_fingerprints"][list(FEATURES)]
    has_p_values = "p_value" in entanglement.columns

    figure, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    labels = [DISPLAY_NAMES.get(name, name) for name in entanglement["feature"]]
    colors = [NATURE["red"] if value >= caution_threshold else NATURE["slate"] for value in entanglement["eta_squared"]]
    bars = axes[0].barh(labels, entanglement["eta_squared"], color=colors)
    if has_p_values:
        for bar, p_value in zip(bars, entanglement["p_value"]):
            if p_value == p_value and p_value < 0.05:
                axes[0].text(bar.get_width(), bar.get_y() + bar.get_height() / 2, "  *", va="center", fontsize=11, fontweight="bold", color=NATURE["red"])
    axes[0].axvline(caution_threshold, color=NATURE["red"], linestyle="--", linewidth=0.8, label=f"Caution threshold ({caution_threshold:.2f})")
    axes[0].set(xlabel=r"$\eta^2$ (variance explained by disease group)   * = permutation p<0.05", xlim=(0, max(caution_threshold * 1.5, float(entanglement["eta_squared"].max()) * 1.15)))
    axes[0].legend(fontsize=7)
    axes[0].set_title("a  Pathology–quality entanglement ranking", loc="left", fontweight="bold")

    image = axes[1].imshow(fingerprints, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    axes[1].set(yticks=np.arange(len(fingerprints)), yticklabels=fingerprints.index)
    axes[1].set(xticks=np.arange(len(FEATURES)), xticklabels=[DISPLAY_NAMES[name] for name in FEATURES])
    axes[1].tick_params(axis="x", rotation=55)
    axes[1].set_title("b  Quality fingerprint by disease group", loc="left", fontweight="bold")
    figure.colorbar(image, ax=axes[1], fraction=0.045, label="Median axis score")

    figure.tight_layout()
    return figure


def harmonization_safety_figure(safety: pd.DataFrame, retention: pd.DataFrame) -> plt.Figure:
    """Notebook firewall panel: AUROC change vs. residual separability, and dose-response retention."""

    set_nature_style()
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    axes[0].scatter(safety["domain_balanced_accuracy_after"], safety["AUROC_change"], s=60, color=NATURE["navy"])
    for _, row in safety.iterrows():
        axes[0].annotate(row["cohort"], (row["domain_balanced_accuracy_after"], row["AUROC_change"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
    axes[0].axhline(-0.02, color=NATURE["red"], linestyle="--", linewidth=0.9, label="AUROC firewall (-0.02)")
    axes[0].set(xlabel="Residual domain balanced accuracy", ylabel="AUROC change after correction")
    axes[0].legend(fontsize=7)
    axes[0].set_title("a  Discrimination-retention firewall", loc="left", fontweight="bold")

    ordered = retention.dropna(subset=["dose_response_retention"]).sort_values("dose_response_retention")
    colors = [NATURE["green"] if passed else NATURE["red"] for passed in ordered["passes_dose_response_firewall"]]
    axes[1].barh(ordered["degradation"].str.replace("_", " "), ordered["dose_response_retention"] * 100, color=colors)
    axes[1].axvline(90, color="#999999", linestyle="--", linewidth=0.8, label="90% retention required")
    axes[1].set(xlabel="Dose-response signal retained after correction (%)")
    axes[1].legend(fontsize=7)
    axes[1].set_title("b  Degradation-sensitivity firewall", loc="left", fontweight="bold")

    figure.tight_layout()
    return figure


def dose_response_trace_figure(dose_grid: np.ndarray, risk_values: np.ndarray, quality_values: np.ndarray, kind: str, current_dose: float) -> plt.Figure:
    """This photograph's own risk/quality curve as one synthetic degradation is dialled up.

    This is the interactive-explainer counterpart to the paper's large-scale
    dose-response validation (:func:`degradation_figure`): instead of a cohort
    median curve, it traces the *current* photograph, so moving the dose slider
    shows exactly why the composite risk score moves the way it does — the same
    "turn one dial, watch the model respond" idea used by interactive ML
    explainers such as GAN Lab, grounded here in EyeQC's own validated
    degradation mechanism rather than a toy simulation.
    """
    set_nature_style()
    figure, axis = plt.subplots(figsize=(6.0, 3.6))
    axis.plot(dose_grid, risk_values, "o-", color=NATURE["red"], linewidth=1.8, markersize=4, label="Poor-quality risk")
    axis.plot(dose_grid, quality_values, "o-", color=NATURE["green"], linewidth=1.8, markersize=4, label="Quality score")
    axis.axvline(current_dose, color=NATURE["ink"], linestyle="--", linewidth=1.0)
    axis.scatter([current_dose], np.interp([current_dose], dose_grid, risk_values), color=NATURE["ink"], s=55, zorder=5)
    axis.set(xlabel=f"{kind.replace('_', ' ').title()} dose (0 = untouched, 1 = severe)", ylabel="Score", ylim=(-0.02, 1.02), xlim=(-0.02, 1.02))
    axis.legend(loc="lower left", fontsize=8)
    axis.set_title("This photograph's dose–response curve", loc="left", fontweight="bold")
    figure.tight_layout()
    return figure


def batch_outlier_scatter_figure(batch_table: pd.DataFrame) -> plt.Figure:
    """MRQy-style cohort scatter: every image in this batch placed by overall quality profile.

    Each photograph's ten-axis feature vector is projected to two dimensions
    with PCA, coloured by workflow route. Images that sit apart from the main
    cluster are the ones worth a manual look first — the same "does this scan
    belong with the rest of the cohort" question MRQy answers with a t-SNE
    plot of its noise/information measures, adapted here to EyeQC's
    interpretable quality axes so the two plot dimensions stay inspectable
    rather than an opaque embedding.
    """
    set_nature_style()
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    valid = batch_table[batch_table["verdict"].isin(("PASS", "REVIEW", "FAIL")) & batch_table[list(FEATURES)].notna().all(axis=1)]
    if len(valid) < 3:
        axis.axis("off")
        axis.text(0.5, 0.5, "Upload at least 3 readable images\nto see the cohort scatter.", ha="center", va="center", transform=axis.transAxes)
        return figure

    coordinates = PCA(n_components=2, random_state=20260822).fit_transform(StandardScaler().fit_transform(valid[list(FEATURES)]))
    projected = valid.assign(pc1=coordinates[:, 0], pc2=coordinates[:, 1])
    for route, group in projected.groupby("verdict"):
        axis.scatter(group["pc1"], group["pc2"], s=42, alpha=0.85, color=ROUTE_COLORS.get(route, NATURE["slate"]), edgecolors="white", linewidths=0.6, label=route)

    centroid = projected[["pc1", "pc2"]].mean()
    distance = np.hypot(projected["pc1"] - centroid.pc1, projected["pc2"] - centroid.pc2)
    for _, row in projected.loc[distance.nlargest(min(5, len(projected))).index].iterrows():
        axis.annotate(row["filename"], (row["pc1"], row["pc2"]), xytext=(5, 4), textcoords="offset points", fontsize=6.5, color=NATURE["ink"])

    axis.set(xlabel="PC1 of the ten quality axes", ylabel="PC2 of the ten quality axes")
    axis.legend(fontsize=8, loc="best")
    axis.set_title("Where each photograph sits in this batch's quality space", loc="left", fontweight="bold")
    figure.tight_layout()
    return figure


def batch_thumbnail_montage_figure(paths: list, batch_table: pd.DataFrame, max_images: int = 48) -> plt.Figure:
    """HistoQC-style contact sheet: every photograph at a glance, bordered by its route.

    HistoQC's core review habit is scanning a coloured thumbnail gallery rather
    than reading a table row by row; this reproduces that for a fundus batch —
    a thin green/amber/red frame around each thumbnail so the clinic can spot
    which photographs need a second look without opening any of them.
    """
    set_nature_style()
    by_name = {path.name: path for path in paths}
    shown = batch_table[batch_table["verdict"].isin(("PASS", "REVIEW", "FAIL"))].head(max_images)
    truncated = len(batch_table) > len(shown)

    n = max(len(shown), 1)
    columns = min(8, n)
    rows = int(np.ceil(n / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 1.55, rows * 1.75))
    axes = np.atleast_1d(axes).reshape(rows, columns)

    for index, ax in enumerate(axes.flat):
        if index >= len(shown):
            ax.axis("off")
            continue
        row = shown.iloc[index]
        color = ROUTE_COLORS.get(row["verdict"], NATURE["slate"])
        try:
            thumbnail = read_rgb(by_name[row["filename"]])
            scale = 220 / max(thumbnail.shape[:2])
            thumbnail = cv2.resize(thumbnail, (int(thumbnail.shape[1] * scale), int(thumbnail.shape[0] * scale)))
            ax.imshow(thumbnail)
        except Exception:  # noqa: BLE001 - unreadable file still gets its verdict-coloured frame
            ax.text(0.5, 0.5, "unreadable", ha="center", va="center", transform=ax.transAxes, fontsize=7)
        # Keep the axes frame (do not call ax.axis("off")) so the verdict-coloured
        # border below stays visible; only the ticks and their labels are hidden.
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3.5)
        ax.set_title(row["filename"], fontsize=6.5, color=NATURE["ink"])

    title = "Batch contact sheet · frame colour = workflow route"
    if truncated:
        title += f" (first {max_images} of {len(batch_table)} shown)"
    figure.suptitle(title, fontsize=10, fontweight="bold", x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return figure


def batch_thumbnail_montage_from_images(images: dict, batch_table: pd.DataFrame, max_images: int = 48) -> plt.Figure:
    """Same contact sheet as :func:`batch_thumbnail_montage_figure`, but from
    already-decoded in-memory RGB arrays (``images``: filename -> array)
    rather than file paths, so an upload never has to touch disk."""
    set_nature_style()
    shown = batch_table[batch_table["verdict"].isin(("PASS", "REVIEW", "FAIL"))].head(max_images)
    truncated = len(batch_table) > len(shown)

    n = max(len(shown), 1)
    columns = min(8, n)
    rows = int(np.ceil(n / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 1.55, rows * 1.75))
    axes = np.atleast_1d(axes).reshape(rows, columns)

    for index, ax in enumerate(axes.flat):
        if index >= len(shown):
            ax.axis("off")
            continue
        row = shown.iloc[index]
        color = ROUTE_COLORS.get(row["verdict"], NATURE["slate"])
        try:
            thumbnail = images[row["filename"]]
            scale = 220 / max(thumbnail.shape[:2])
            thumbnail = cv2.resize(thumbnail, (int(thumbnail.shape[1] * scale), int(thumbnail.shape[0] * scale)))
            ax.imshow(thumbnail)
        except Exception:  # noqa: BLE001 - unreadable file still gets its verdict-coloured frame
            ax.text(0.5, 0.5, "unreadable", ha="center", va="center", transform=ax.transAxes, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3.5)
        ax.set_title(row["filename"], fontsize=6.5, color=NATURE["ink"])

    title = "Batch contact sheet · frame colour = workflow route"
    if truncated:
        title += f" (first {max_images} of {len(batch_table)} shown)"
    figure.suptitle(title, fontsize=10, fontweight="bold", x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return figure


def decision_support_figure(rgb: np.ndarray, analysis: dict) -> plt.Figure:
    set_nature_style()
    features = analysis["features"]
    grid = tile_quality_map(features)
    heatmap = np.kron(1 - grid, np.ones((40, 40)))
    heatmap = np.array(
        plt.get_cmap("magma")(heatmap)[..., :3] * 255,
        dtype=np.uint8,
    )
    heatmap = cv2.resize(heatmap, (rgb.shape[1], rgb.shape[0]))
    overlay = np.clip(0.68 * rgb + 0.32 * heatmap, 0, 255).astype(np.uint8)

    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    axes[0].imshow(rgb)
    axes[0].set_title("Source photograph")
    axes[1].imshow(overlay)
    axes[1].set_title("Local quality map")
    values = [features[name] for name in FEATURES]
    colors = [NATURE["red"] if value < 0.45 else NATURE["green"] for value in values]
    axes[2].barh(np.arange(len(FEATURES)), values, color=colors)
    axes[2].set(yticks=np.arange(len(FEATURES)), yticklabels=[DISPLAY_NAMES[name] for name in FEATURES], xlim=(0, 1), xlabel="Axis score")
    axes[2].invert_yaxis()
    axes[2].set_title(f"{analysis['verdict']} · quality {analysis['quality_score'] * 100:.0f}/100")
    for axis in axes[:2]:
        axis.axis("off")
    figure.tight_layout()
    return figure
