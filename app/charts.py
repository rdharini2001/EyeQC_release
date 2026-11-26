"""Every chart of the interface: interactive Plotly figures drawn for the dark surface.

Colours: the categorical order is validated for colour-vision deficiency and contrast on the panel colour; status colours
are used only for PASS / REVIEW / FAIL; magnitudes use one teal ramp.  Backgrounds are transparent, so figures take the
colour of the panel that holds them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from eyeqc.contract import FAIL_THRESHOLD, PASS_THRESHOLD
from eyeqc.core import DISPLAY_NAMES, FEATURES

TEXT, SOFT, GRID, SURFACE = "#EEF3F4", "#A9B7BF", "#37434D", "#232C35"
TEAL, MUTED = "#65DBC1", "#6F808B"
SERIES = ["#338ced", "#e76624", "#00a888", "#d28e05", "#e24d8b", "#0aac23", "#9887ef", "#eb6263"]
STATUS = {"PASS": "#43B581", "REVIEW": "#FBC82B", "FAIL": "#F4686C"}
RAMP = [[0.0, "#26333C"], [0.35, "#1F5B59"], [0.7, "#1F9E8B"], [1.0, "#9AF0DB"]]
OTHER = "#8A969E"
CONFIG_HEIGHT = 340


def _style(figure: go.Figure, height: int = CONFIG_HEIGHT, *, title: str | None = None, legend: bool = True) -> go.Figure:
    figure.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=SERIES,
        font=dict(family="Inter, Source Sans Pro, system-ui, sans-serif", size=13, color=TEXT),
        margin=dict(l=16, r=18, t=56 if title else 28, b=16), showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0, font=dict(color=SOFT, size=12), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#11181E", bordercolor=GRID, font=dict(color=TEXT, size=12)),
    )
    if title:
        figure.update_layout(title=dict(text=title, x=0, xref="paper", xanchor="left", y=0.98, yanchor="top", font=dict(size=15, color=TEXT)))
    axis = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickcolor=GRID, tickfont=dict(color=SOFT, size=12),
                title_font=dict(color=SOFT, size=12.5), automargin=True)
    figure.update_xaxes(**axis)
    figure.update_yaxes(**axis)
    return figure


def label(name: str) -> str:
    return DISPLAY_NAMES.get(name, name.replace("_", " ").capitalize())


def empty(message: str, height: int = 220) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper", font=dict(color=SOFT, size=13))
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return _style(figure, height, legend=False)


def _risk_band(figure: go.Figure, **where) -> None:
    figure.add_hrect(y0=0, y1=PASS_THRESHOLD, fillcolor=STATUS["PASS"], opacity=0.07, line_width=0, layer="below", **where)
    figure.add_hrect(y0=PASS_THRESHOLD, y1=FAIL_THRESHOLD, fillcolor=STATUS["REVIEW"], opacity=0.35, line_width=0, layer="below", **where)
    figure.add_hrect(y0=FAIL_THRESHOLD, y1=1, fillcolor=STATUS["FAIL"], opacity=0.07, line_width=0, layer="below", **where)


def _verdict_trace(x, risk, verdicts, size: int = 11, legend: bool = False, hover: str = "severity %{x:.0%}") -> go.Scatter:
    return go.Scatter(x=x, y=risk, mode="lines+markers", showlegend=legend, line=dict(color=TEAL, width=2),
                      marker=dict(size=size, color=[STATUS[v] for v in verdicts], line=dict(color=SURFACE, width=2)),
                      customdata=verdicts, hovertemplate=hover + "<br>risk %{y:.3f} · %{customdata}<extra></extra>")


# ---------------------------------------------------------------- robustness
def dose_response(table: pd.DataFrame, defect: str) -> go.Figure:
    figure = go.Figure(_verdict_trace(table["severity"], table["risk"], table["verdict"], size=13))
    _risk_band(figure)
    figure.update_xaxes(title=f"{defect}: simulated severity", tickformat=".0%", tickvals=sorted(table["severity"].unique()), range=[-0.04, 1.04])
    figure.update_yaxes(title="poor-quality risk", range=[-0.03, 1.03])
    for verdict, y in (("PASS", PASS_THRESHOLD / 2), ("FAIL", 0.94)):
        figure.add_annotation(x=1.04, y=y, text=verdict, showarrow=False, xanchor="right", font=dict(color=STATUS[verdict], size=11))
    return _style(figure, 360, legend=False)


def dose_grid(table: pd.DataFrame, names: dict[str, str]) -> go.Figure:
    """Small multiples: one panel per defect, same axes."""
    kinds = list(names)
    columns = 3
    rows = int(np.ceil(len(kinds) / columns))
    figure = make_subplots(rows=rows, cols=columns, subplot_titles=[names[k] for k in kinds], shared_xaxes=True, shared_yaxes=True,
                           vertical_spacing=0.1, horizontal_spacing=0.04)
    for i, kind in enumerate(kinds):
        data = table[table["degradation"] == kind]
        row, col = i // columns + 1, i % columns + 1
        figure.add_trace(_verdict_trace(data["severity"], data["risk"], data["verdict"], size=8), row=row, col=col)
    figure.update_annotations(font=dict(color=TEXT, size=13))
    _risk_band(figure, row="all", col="all")
    figure.update_xaxes(tickformat=".0%", tickvals=[0, 0.5, 1], range=[-0.06, 1.06])
    figure.update_yaxes(range=[-0.04, 1.04], tickvals=[0, 0.5, 1])
    return _style(figure, 190 * rows + 40, legend=False)


# ---------------------------------------------------------------- one photograph against another
def dumbbell(first: dict, second: dict, names: tuple[str, str] = ("Photograph A", "Photograph B")) -> go.Figure:
    labels = [label(name) for name in FEATURES]
    figure = go.Figure()
    for name, a, b in zip(labels, (first[f] for f in FEATURES), (second[f] for f in FEATURES)):
        figure.add_trace(go.Scatter(x=[a, b], y=[name, name], mode="lines", line=dict(color=MUTED, width=2), showlegend=False, hoverinfo="skip"))
    for values, title, colour in ((first, names[0], SERIES[0]), (second, names[1], SERIES[1])):
        figure.add_trace(go.Scatter(x=[values[f] for f in FEATURES], y=labels, mode="markers", name=title,
                                    marker=dict(size=12, color=colour, line=dict(color=SURFACE, width=2)),
                                    hovertemplate="%{y}: %{x:.3f}<extra>" + title + "</extra>"))
    figure.update_xaxes(title="measurement (1 = best)", range=[-0.03, 1.03])
    figure.update_yaxes(autorange="reversed", showgrid=False)
    return _style(figure, 430)


def contribution_difference(first: dict, second: dict) -> go.Figure:
    """Change in each log-odds term from photograph A to photograph B."""
    labels = [label(name) for name in FEATURES] + ["Visual features"]
    change = [second["axis_contributions"][f] - first["axis_contributions"][f] for f in FEATURES]
    change.append(second["representation_contribution"] - first["representation_contribution"])
    figure = go.Figure(go.Bar(x=change, y=labels, orientation="h", marker=dict(color=[STATUS["FAIL"] if v > 0 else TEAL for v in change], cornerradius=4),
                              hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    figure.update_xaxes(title="change in log-odds, A → B (right = riskier)")
    figure.update_yaxes(autorange="reversed", showgrid=False)
    return _style(figure, 430, legend=False)


# ---------------------------------------------------------------- cohorts
def phenotypes(shares: pd.DataFrame, top: int = 6) -> go.Figure:
    """Stacked shares per cohort; phenotypes beyond the ``top`` most common are folded into "Other"."""
    order = shares.sum().sort_values(ascending=False).index.tolist()
    kept, rest = order[:top], order[top:]
    table = shares[kept].copy()
    if rest:
        table["Other"] = shares[rest].sum(axis=1)
    figure = go.Figure()
    for i, name in enumerate(table.columns):
        title = "No measurement raises risk" if name == "none" else label(name)
        figure.add_trace(go.Bar(y=[str(c) for c in table.index], x=table[name], orientation="h", name=title,
                                marker=dict(color=OTHER if name == "Other" else SERIES[i], line=dict(color=SURFACE, width=2)),
                                hovertemplate="%{y}<br>" + title + ": %{x:.1%}<extra></extra>"))
    figure.update_layout(barmode="stack", bargap=0.35, legend_traceorder="normal")
    figure.update_xaxes(tickformat=".0%", range=[0, 1], title="share of photographs not assigned PASS")
    figure.update_yaxes(autorange="reversed", showgrid=False)
    return _style(figure, 170 + 46 * len(table))


def profiles(medians: pd.DataFrame, title: str | None = None) -> go.Figure:
    values = medians[list(FEATURES)].to_numpy(float)
    figure = go.Figure(go.Heatmap(z=values, x=[label(f) for f in FEATURES], y=[str(c) for c in medians.index], zmin=0, zmax=1,
                                  colorscale=RAMP, xgap=2, ygap=2, colorbar=dict(title=dict(text="median", font=dict(color=SOFT)), thickness=10,
                                  tickfont=dict(color=SOFT), outlinewidth=0), hovertemplate="%{y} · %{x}: %{z:.3f}<extra></extra>"))
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            figure.add_annotation(x=j, y=i, text=f"{value:.2f}", showarrow=False, font=dict(size=11, color="#0E1A1F" if value > 0.72 else TEXT))
    figure.update_xaxes(tickangle=-35, showgrid=False)
    figure.update_yaxes(autorange="reversed", showgrid=False)
    return _style(figure, 190 + 42 * len(medians), title=title, legend=False)


def separability(rows: pd.DataFrame, chance: float) -> go.Figure:
    figure = go.Figure(go.Bar(x=rows["version"], y=rows["balanced_accuracy"], marker=dict(color=[MUTED, SERIES[0], SERIES[2]][:len(rows)], cornerradius=4),
                              width=0.45, text=[f"{v:.2f}" for v in rows["balanced_accuracy"]], textposition="outside",
                              hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    figure.add_hline(y=chance, line=dict(color=SOFT, width=1), annotation_text=f"chance {chance:.2f}", annotation_font_color=SOFT,
                     annotation_position="top right")
    figure.update_yaxes(title="cohort predicted from the ten measurements<br>(balanced accuracy)", range=[0, 1.08])
    figure.update_xaxes(showgrid=False)
    return _style(figure, 340, legend=False)


def variance(table: pd.DataFrame) -> go.Figure:
    names = [label(f) for f in table["feature"]]
    figure = go.Figure()
    for column, title, colour in (("inter_batch_fraction", "between cohorts", SERIES[0]), ("residual_fraction", "within cohorts", "#4A5A66")):
        figure.add_trace(go.Bar(y=names, x=table[column], orientation="h", name=title, marker=dict(color=colour, line=dict(color=SURFACE, width=2)),
                                hovertemplate="%{y}: %{x:.1%} " + title + "<extra></extra>"))
    figure.update_layout(barmode="stack", bargap=0.3)
    figure.update_xaxes(tickformat=".0%", range=[0, 1], title="share of each measurement's variance")
    figure.update_yaxes(autorange="reversed", showgrid=False)
    return _style(figure, 400)


def association(table: pd.DataFrame, title: str = "η² with the grade (cohort-centred)") -> go.Figure:
    data = table.sort_values("eta_squared")
    figure = go.Figure(go.Bar(y=[label(f) for f in data["feature"]], x=data["eta_squared"], orientation="h",
                              marker=dict(color=SERIES[0], cornerradius=4), customdata=data["p_value"],
                              hovertemplate="%{y}: η² %{x:.3f}<br>permutation p = %{customdata:.3f}<extra></extra>"))
    figure.update_xaxes(title=title, rangemode="tozero")
    figure.update_yaxes(showgrid=False)
    return _style(figure, 400, legend=False)


# ---------------------------------------------------------------- calibration and agreement
def reliability(curves: dict[str, tuple[np.ndarray, np.ndarray]], title: str | None = None) -> go.Figure:
    """``curves``: name -> (predicted risk, observed label) of the same photographs; ten equal-width bins."""
    figure = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=SOFT, width=1, dash="dot"), name="perfect", hoverinfo="skip"))
    for (name, (risk, labels)), colour in zip(curves.items(), (SERIES[1], TEAL, SERIES[0])):
        risk, labels = np.asarray(risk, float), np.asarray(labels, float)
        bins = np.minimum((risk * 10).astype(int), 9)
        used = [j for j in range(10) if (bins == j).any()]
        figure.add_trace(go.Scatter(x=[risk[bins == j].mean() for j in used], y=[labels[bins == j].mean() for j in used], mode="lines+markers",
                                    name=name, line=dict(color=colour, width=2), marker=dict(size=10, color=colour, line=dict(color=SURFACE, width=2)),
                                    customdata=[int((bins == j).sum()) for j in used],
                                    hovertemplate="predicted %{x:.2f}<br>observed %{y:.2f}<br>%{customdata} photographs<extra>" + name + "</extra>"))
    figure.update_xaxes(title="predicted poor-quality risk", range=[-0.03, 1.03])
    figure.update_yaxes(title="observed share poor", range=[-0.03, 1.03])
    return _style(figure, 420, title=title)


def split_spread(before: pd.DataFrame, after: pd.DataFrame, metrics: dict[str, str]) -> go.Figure:
    """Distribution over repeated splits: before and after recalibration, one panel per metric."""
    figure = make_subplots(rows=1, cols=len(metrics), subplot_titles=list(metrics.values()), horizontal_spacing=0.06)
    for i, key in enumerate(metrics):
        for frame, name, colour in ((before, "before", SERIES[1]), (after, "after", TEAL)):
            figure.add_trace(go.Box(y=frame[key], name=name, marker_color=colour, line=dict(color=colour, width=1.5), boxpoints="outliers",
                                    marker=dict(size=4), showlegend=False, hovertemplate="%{y:.3f}<extra>" + name + "</extra>"), row=1, col=i + 1)
    figure.update_annotations(font=dict(color=TEXT, size=13))
    figure.update_yaxes(rangemode="tozero")
    figure.update_xaxes(showgrid=False)
    return _style(figure, 300, legend=False)


def kappa_matrix(matrix: pd.DataFrame, title: str | None = None) -> go.Figure:
    names = [str(n) for n in matrix.index]
    values = matrix.to_numpy(float)
    shown = np.where(np.eye(len(names), dtype=bool), np.nan, values)
    low = float(np.nanmin(shown)) if np.isfinite(shown).any() else 0.0
    figure = go.Figure(go.Heatmap(z=shown, x=names, y=names, colorscale=RAMP, zmin=min(0.0, low), zmax=1, xgap=3, ygap=3, showscale=False,
                                  hovertemplate="%{y} vs %{x}: κ %{z:.3f}<extra></extra>"))
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                figure.add_annotation(x=names[j], y=names[i], text=f"{values[i, j]:.3f}", showarrow=False,
                                      font=dict(size=13, color="#0E1A1F" if values[i, j] > 0.72 else TEXT))
    figure.update_xaxes(showgrid=False, side="top")
    figure.update_yaxes(autorange="reversed", showgrid=False)
    return _style(figure, 120 + 70 * len(names), title=title, legend=False)


def kappa_difference(table: pd.DataFrame, margin: float) -> go.Figure:
    """EyeQC-grader kappa minus grader-grader kappa, with bootstrap interval and equivalence margin."""
    figure = go.Figure()
    figure.add_vrect(x0=-margin, x1=margin, fillcolor=TEAL, opacity=0.1, line_width=0, layer="below")
    figure.add_vline(x=0, line=dict(color=SOFT, width=1))
    figure.add_trace(go.Scatter(
        x=table["difference"], y=table["grader"].astype(str), mode="markers",
        marker=dict(size=12, color=TEXT, line=dict(color=SURFACE, width=2)),
        error_x=dict(type="data", symmetric=False, array=table["ci_high"] - table["difference"],
                     arrayminus=table["difference"] - table["ci_low"], color=TEXT, thickness=2, width=6),
        customdata=table[["ci_low", "ci_high"]], hovertemplate="%{y}: %{x:+.3f} (95% CI %{customdata[0]:+.3f} to %{customdata[1]:+.3f})<extra></extra>"))
    edge = max(margin, float(np.nanmax(np.abs(table[["ci_low", "ci_high"]].to_numpy())))) * 1.2
    figure.update_xaxes(title=f"κ(EyeQC, grader) − mean κ(grader, other graders); shaded = ±{margin:g}", range=[-edge, edge])
    figure.update_yaxes(autorange="reversed", showgrid=False)
    return _style(figure, 130 + 52 * len(table), legend=False)


# ---------------------------------------------------------------- study dashboard
def auroc_forest(table: pd.DataFrame) -> go.Figure:
    names = table["cohort"].tolist()
    figure = go.Figure(go.Scatter(
        x=table["AUROC"], y=names, mode="markers+text", text=[f"{v:.3f}" for v in table["AUROC"]], textposition="top center",
        textfont=dict(color=TEXT, size=12), marker=dict(size=13, color=TEAL, line=dict(color=SURFACE, width=2)),
        error_x=dict(type="data", symmetric=False, array=table["AUROC_hi"] - table["AUROC"], arrayminus=table["AUROC"] - table["AUROC_lo"],
                     color=TEAL, thickness=2, width=6),
        customdata=table[["N", "AUROC_lo", "AUROC_hi"]],
        hovertemplate="%{y}: %{x:.3f} (%{customdata[1]:.3f}–%{customdata[2]:.3f})<br>%{customdata[0]:,} photographs<extra></extra>"))
    figure.update_xaxes(title="AUROC (95% CI)", range=[0.7, 1.03])
    figure.update_yaxes(autorange="reversed", showgrid=False)
    return _style(figure, 330, title="Discrimination", legend=False)


def cohort_bars(table: pd.DataFrame, columns: dict[str, str], title: str) -> go.Figure:
    figure = go.Figure()
    for (column, name), colour in zip(columns.items(), SERIES):
        figure.add_trace(go.Bar(x=table["cohort"], y=table[column], name=name, marker=dict(color=colour, cornerradius=4),
                                hovertemplate="%{x}: %{y:.1%}<extra>" + name + "</extra>"))
    figure.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08)
    figure.update_yaxes(tickformat=".0%", range=[0, 1.04])
    figure.update_xaxes(showgrid=False)
    return _style(figure, 340, title=title)


def method_dots(table: pd.DataFrame, value: str, title: str, order: list[str], highlight: str | None = None,
                axis_title: str = "AUROC", axis_range: tuple[float, float] | None = None) -> go.Figure:
    """Cohorts on the vertical axis, one marker per method (at most eight)."""
    figure = go.Figure()
    symbols = ["circle", "diamond", "square", "triangle-up", "x", "star", "cross", "triangle-down"]
    for i, method in enumerate(order):
        data = table[table["method"] == method]
        main = method == highlight
        figure.add_trace(go.Scatter(x=data[value], y=data["cohort"], mode="markers", name=method,
                                    marker=dict(size=15 if main else 10, color=TEXT if main else SERIES[(i - 1 if highlight else i) % 8],
                                                symbol="circle" if main else symbols[(i) % 8], line=dict(color=SURFACE, width=1.5)),
                                    hovertemplate="%{y}: %{x:.3f}<extra>" + method + "</extra>"))
    figure.update_xaxes(title=axis_title, range=list(axis_range) if axis_range else None)
    figure.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=table["cohort"].unique().tolist())
    return _style(figure, 400, title=title)


def horizontal(names: list[str], values: list[float], title: str, axis_title: str, percent: bool = False) -> go.Figure:
    template = "%{y}: %{x:.1%}" if percent else "%{y}: %{x:.3f}"
    figure = go.Figure(go.Bar(x=values, y=names, orientation="h", marker=dict(color=SERIES[0], cornerradius=4),
                              text=[f"{v:.1%}" if percent else f"{v:.2f}" for v in values], textposition="outside",
                              textfont=dict(color=TEXT, size=12), cliponaxis=False, hovertemplate=template + "<extra></extra>"))
    figure.update_xaxes(title=axis_title, tickformat=".0%" if percent else None, range=[0, 1.12])
    figure.update_yaxes(autorange="reversed", showgrid=False)
    return _style(figure, 100 + 30 * len(names), title=title, legend=False)


def matrix_heatmap(table: pd.DataFrame, title: str, zmin: float, zmax: float, fmt: str = "{:.2f}") -> go.Figure:
    values = table.to_numpy(float)
    figure = go.Figure(go.Heatmap(z=values, x=[str(c) for c in table.columns], y=[str(i) for i in table.index], zmin=zmin, zmax=zmax,
                                  colorscale=RAMP, xgap=3, ygap=3, showscale=False, hovertemplate="%{y} · %{x}: %{z:.3f}<extra></extra>"))
    threshold = zmin + 0.72 * (zmax - zmin)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                figure.add_annotation(x=j, y=i, text=fmt.format(values[i, j]), showarrow=False,
                                      font=dict(size=12, color="#0E1A1F" if values[i, j] > threshold else TEXT))
    figure.update_xaxes(showgrid=False, side="top")
    figure.update_yaxes(autorange="reversed", showgrid=False)
    return _style(figure, 130 + 40 * len(table), title=None, legend=False).update_layout(
        title=dict(text=title, x=0, xref="paper", y=0.99, yanchor="top", font=dict(size=15, color=TEXT)), margin=dict(t=86))


def budget_curves(table: pd.DataFrame, value: str, title: str, axis_title: str) -> go.Figure:
    figure = go.Figure()
    for method, name, colour in (("frozen", "frozen", SERIES[1]), ("monotone_local_Platt", "recalibrated", TEAL)):
        data = table[table["method"] == method].sort_values("budget_images")
        figure.add_trace(go.Scatter(x=data["budget_images"], y=data[f"{value}_median"], mode="lines+markers", name=name,
                                    line=dict(color=colour, width=2), marker=dict(size=10, color=colour, line=dict(color=SURFACE, width=2)),
                                    error_y=dict(type="data", symmetric=False, array=data[f"{value}_split_hi"] - data[f"{value}_median"],
                                                 arrayminus=data[f"{value}_median"] - data[f"{value}_split_lo"], color=colour, thickness=1, width=4),
                                    hovertemplate="%{x} labels: %{y:.3f}<extra>" + name + "</extra>"))
    figure.update_xaxes(title="locally labelled photographs", tickvals=[25, 50, 100, 200])
    figure.update_yaxes(title=axis_title, rangemode="tozero")
    return _style(figure, 340, title=title)


def weights(rows: list[dict], key: str, value: str, title: str) -> go.Figure:
    names = [str(row[key]) for row in rows]
    shown = [name if len(name) <= 46 else name[:44] + "…" for name in names]
    figure = go.Figure(go.Bar(x=[row[value] for row in rows], y=shown, orientation="h", marker=dict(color=SERIES[6], cornerradius=4),
                              customdata=names, text=[f"{row[value]:.0%}" for row in rows], textposition="outside", cliponaxis=False,
                              textfont=dict(color=TEXT), hovertemplate="%{customdata}: %{x:.1%}<extra></extra>"))
    figure.update_xaxes(tickformat=".0%", range=[0, 1.12], title="relative weight among these prompts")
    figure.update_yaxes(autorange="reversed", showgrid=False)
    return _style(figure, 130 + 44 * len(rows), title=title, legend=False)
