"""EyeQC interface (Gradio).  Run from the repository root:  python app/app.py

The first tab is the EyeQC window of the paper (Figure 2); the following tabs apply the paper's analyses to your own
photographs and tables.  Every EyeQC number comes from the released model.  "Vessels & FLAIR" holds optional third-party
tools that are not part of the paper.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Many small linear-algebra calls: a few threads are much faster than one per core on shared machines.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
# EyeQC itself runs on CPU, as in the paper (and a ZeroGPU Space only allows CUDA inside the optional tools' GPU calls).
os.environ.setdefault("EYEQC_DEVICE", "cpu")

HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent, HERE / "third_party"):      # Space layout, repository layout, optional third-party tools
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import extras  # noqa: E402  (must precede torch so that ZeroGPU can patch CUDA initialisation)
import gradio as gr

from eyeqc.criteria import DEFAULT_TEXT, parse_criteria, predefined_criteria

import about
import charts
import dashboard
import engine
import research
import ui

EXAMPLES = json.loads((HERE / "examples" / "examples.json").read_text())
EXAMPLE_PATHS = [str(HERE / "examples" / item["file"]) for item in EXAMPLES]
PAIR = [str(HERE / "examples" / item["file"]) for item in json.loads((HERE / "examples" / "pair.json").read_text())]
DARK_MODE = "<script>document.addEventListener('DOMContentLoaded', () => document.body.classList.add('dark'));</script>"
STYLE_LABELS = {"Spectral (blue good, red poor)": "spectral", "Highlight low scores only": "drop"}
REQUESTS = ("Is it in focus?", "Check glare and shadows", "Is the whole field visible?", "Everything except colour", "Is this usable?")


# ---------------------------------------------------------------- workbench callbacks
def run_assessment(path, text, use_predefined, style_label, show_values, show_demo):
    style = STYLE_LABELS[style_label]
    if not path:
        return None, ui.report(None), ui.heatmaps(None, style), "", "", gr.update(value=None, interactive=False)
    criteria = predefined_criteria() if use_predefined else parse_criteria(text)
    try:
        assessment = engine.assess(path, criteria, style=style, demo=show_demo, request="" if use_predefined else (text or ""))
    except Exception as problem:
        raise gr.Error(f"This photograph could not be assessed: {problem}")
    return (assessment.photo, ui.report(assessment, show_values), ui.heatmaps(assessment, style), ui.reply(assessment),
            ui.decomposition(assessment), gr.update(value=engine.export_report(assessment), interactive=True))


def preview(text, use_predefined):
    return ui.interpretation(predefined_criteria() if use_predefined else parse_criteria(text))


# ---------------------------------------------------------------- research callbacks
def guard(function, *args):
    try:
        return function(*args)
    except ValueError as problem:
        raise gr.Error(str(problem))


def run_localisation(path, defect, severity, placement):
    photo, tiles, truth, box, table, result = guard(research.localisation, path, defect, severity, placement)
    return ui.localisation(photo, tiles, truth, box, result, defect), table


def run_compare(path_a, path_b):
    results, dumbbell, difference, table = guard(research.compare, path_a, path_b)
    return ui.comparison(results), dumbbell, difference, table


def run_batch(files, cohorts, progress=gr.Progress()):
    table, csv, phenotype, profile, counts = guard(research.score_batch, files, cohorts, progress)
    summary = dashboard.tiles([(str(counts["photographs"]), "photographs scored", ""), (str(counts["PASS"]), "PASS", ""),
                               (str(counts["REVIEW"]), "REVIEW", ""), (str(counts["FAIL"]), "FAIL", "")])
    return summary, table, csv, phenotype, profile


def example_batch():
    files = EXAMPLE_PATHS + PAIR
    return files, "example set A, example set A, example set A, example set B, example set B"


def plot(label: str | None = None, value=None):
    return gr.Plot(value=value, label=label, show_label=False, elem_classes="eq-plot")


def redraw_on_open(tab, *drawn) -> None:
    """Charts created while their tab is hidden get a default width: send them again when the tab opens so they fill it.

    ``drawn`` are (plot component, figure) pairs.
    """
    tab.select(lambda: [figure for _, figure in drawn], None, [component for component, _ in drawn], show_progress="hidden", queue=False)


def shown(figure) -> tuple:
    """A plot of a stored figure, returned with the figure so that it can be redrawn."""
    return plot(value=figure), figure


def table(columns: tuple[str, ...] | None = None, **options):
    if columns:
        options["headers"] = list(columns)
    return gr.Dataframe(interactive=False, wrap=True, elem_classes="eq-grid", **options)


def build() -> gr.Blocks:
    with gr.Blocks(title="EyeQC", fill_width=True) as demo:
        current = gr.State(EXAMPLE_PATHS[0])
        settings_open = gr.State(False)
        with gr.Tabs(elem_id="eq-tabs") as tabs:
            with gr.Tab("Overview", id="overview"):
                gr.HTML(about.landing_head(), padding=False)
                open_tool = gr.Button("Open the EyeQC tool", variant="primary", elem_id="eq-start")
                gr.HTML(about.landing_figures(), padding=False)
                open_tool.click(lambda: gr.Tabs(selected="eyeqc"), None, tabs)
            # ------------------------------------------------ the EyeQC window (Figure 2)
            with gr.Tab("EyeQC", id="eyeqc"):
                with gr.Column(elem_id="eq-window"):
                    with gr.Row(elem_id="eq-titlebar"):
                        gr.HTML(ui.brand())
                        open_button = gr.UploadButton("Open photograph", file_types=["image"], type="filepath", elem_id="eq-open", elem_classes="eq-icon")
                        download = gr.DownloadButton("Download report", elem_id="eq-download", elem_classes="eq-icon", interactive=False)
                        gear = gr.Button("Settings", elem_id="eq-gear", elem_classes="eq-icon")
                    with gr.Row(elem_id="eq-settings", visible=False) as settings:
                        style_choice = gr.Radio(list(STYLE_LABELS), value=list(STYLE_LABELS)[0], label="Map colours")
                        show_demo = gr.Checkbox(True, label="Show the simulated glare and shadow rows")
                        show_values = gr.Checkbox(False, label="Show measurement values")
                    with gr.Row(elem_id="eq-body", equal_height=False):
                        with gr.Column(elem_id="eq-input", scale=25, min_width=260):
                            with gr.Row(elem_id="eq-imagehead"):
                                gr.HTML("<div class='eq-title'>Fundus image</div>")
                                replace = gr.UploadButton("⤓ Replace", file_types=["image"], type="filepath", elem_id="eq-replace", elem_classes="eq-outlined", size="sm")
                            photo = gr.Image(elem_id="eq-photo", show_label=False, interactive=False, height=250, buttons=[])
                            upload = gr.UploadButton("⤒ Upload image", file_types=["image"], type="filepath", elem_id="eq-upload", elem_classes="eq-outlined")
                            text = gr.Textbox(DEFAULT_TEXT, label="Quality criteria", lines=2, max_lines=4, elem_id="eq-criteria",
                                              placeholder="Ask in your own words, e.g. is it blurry? any shadows? ignore colour")
                            chips = gr.HTML(preview(DEFAULT_TEXT, False), padding=False)
                            with gr.Row(elem_id="eq-requests"):
                                request_buttons = [gr.Button(example, size="sm", elem_classes="eq-request", min_width=60) for example in REQUESTS]
                            predefined = gr.Checkbox(False, label="Use predefined criteria\n10 quality attributes", elem_id="eq-predefined", show_label=False, container=False)
                            assess = gr.Button("Assess image", elem_id="eq-assess", variant="primary")
                        with gr.Column(elem_id="eq-report-col", scale=27, min_width=300):
                            report = gr.HTML(ui.report(None), padding=False)
                        with gr.Column(elem_id="eq-maps-col", scale=48, min_width=460):
                            maps = gr.HTML(ui.heatmaps(None), padding=False)
                    reply = gr.HTML(padding=False, elem_id="eq-reply")
                with gr.Accordion("Why this score: contribution of each measurement and of the visual features", open=False):
                    contributions = gr.HTML(padding=False)
                with gr.Row(elem_id="eq-examples"):
                    gr.HTML("<div class='eq-summary'><b>Examples</b> · DeepDRiD, CC BY-SA 4.0</div>", padding=False)
                    example_buttons = [gr.Button(item["label"], size="sm", elem_classes="eq-outlined") for item in EXAMPLES]

                inputs = [current, text, predefined, style_choice, show_values, show_demo]
                outputs = [photo, report, maps, reply, contributions, download]
                for button, path in zip(example_buttons, EXAMPLE_PATHS):
                    button.click(lambda chosen=path: chosen, None, current)
                for button in (open_button, replace, upload):
                    button.upload(lambda file: file, button, current)
                current.change(run_assessment, inputs, outputs)
                assess.click(run_assessment, inputs, outputs)
                text.submit(run_assessment, inputs, outputs)
                for control in (text, predefined):
                    control.change(preview, [text, predefined], chips, queue=False, show_progress="hidden")
                for button, example in zip(request_buttons, REQUESTS):
                    button.click(lambda chosen=example: (chosen, False), None, [text, predefined]).then(run_assessment, inputs, outputs)
                for control in (style_choice, show_values, show_demo):
                    control.change(run_assessment, inputs, outputs)
                gear.click(lambda state: (not state, gr.update(visible=not state)), settings_open, [settings_open, settings])
                demo.load(run_assessment, inputs, outputs)

            # ------------------------------------------------ two photographs of one eye
            with gr.Tab("Compare"):
                gr.HTML(ui.section("Two photographs of the same eye", "Which acquisition is better, and in which measurements do they differ? In the study, "
                                   "separate photographs of one eye received the same decision in 66.5% of 798 pairs and opposite PASS and FAIL in 29.9%.", True))
                with gr.Row():
                    first = gr.Image(type="filepath", label="Photograph A", height=250, elem_classes="eq-upload")
                    second = gr.Image(type="filepath", label="Photograph B", height=250, elem_classes="eq-upload")
                with gr.Row():
                    pair_button = gr.Button("Load an example pair", elem_classes="eq-outlined")
                    compare_button = gr.Button("Compare", variant="primary")
                verdicts = gr.HTML(padding=False)
                with gr.Row():
                    compare_measures = plot()
                    compare_terms = plot()
                compare_table = table(("Measurement", "A", "B", "B − A"))
                pair_button.click(lambda: tuple(PAIR), None, [first, second]).then(run_compare, [first, second], [verdicts, compare_measures, compare_terms, compare_table])
                compare_button.click(run_compare, [first, second], [verdicts, compare_measures, compare_terms, compare_table])

            # ------------------------------------------------ robustness and localisation
            with gr.Tab("Robustness"):
                gr.HTML(ui.section("Robustness to degradations", "Simulated acquisition defects applied to the photograph open in the EyeQC tab.", True))
                with gr.Tabs(elem_classes="eq-subtabs"):
                    with gr.Tab("Risk by severity"):
                        with gr.Row(equal_height=True):
                            defect = gr.Dropdown(list(research.DEGRADATION_LABELS), value="Shadow", label="Defect", scale=3)
                            run_one = gr.Button("Run this defect", variant="primary", scale=1)
                            run_all = gr.Button("Run all nine (about a minute on CPU)", elem_classes="eq-outlined", scale=1)
                        curve = plot()
                        frames = gr.Gallery(label="Degraded photographs", columns=5, height=230, object_fit="contain")
                        dose_table = table(("Severity", "Poor-quality risk", "Decision"))
                        dose_file = gr.File(label="Download values")
                        run_one.click(lambda p, k: guard(research.degradation_response, p, k), [current, defect], [curve, frames, dose_table, dose_file])

                        def all_defects(path):
                            figure, summary, csv = guard(research.degradation_all, path)
                            return figure, None, summary, csv

                        run_all.click(all_defects, current, [curve, frames, dose_table, dose_file])
                    with gr.Tab("Spatial localisation"):
                        gr.Markdown("A local defect is placed at a known position. Each map is then asked to find it: pixel AUC (0.5 = chance), and whether "
                                    "its lowest-scoring region falls inside the defect. Local blur is the hardest case.")
                        with gr.Row(equal_height=True):
                            local_defect = gr.Radio(list(research.LOCAL_DEFECTS), value="Glare", label="Defect")
                            severity = gr.Slider(0.25, 1.0, value=1.0, step=0.25, label="Severity")
                            placement = gr.Slider(0, 50, value=7, step=1, label="Position (random seed)")
                            run_local = gr.Button("Locate", variant="primary")
                        local_view = gr.HTML(padding=False)
                        local_table = table(("Map", "Pixel AUC (0.5 = chance)", "Lowest region inside defect", "Lowest 6×6 tile inside defect"))
                        run_local.click(run_localisation, [current, local_defect, severity, placement], [local_view, local_table])

            # ------------------------------------------------ cohorts
            with gr.Tab("Cohort analysis"):
                gr.HTML(ui.section("Quality-phenotype analysis and cohort quality profiles", f"Score up to {research.MAX_BATCH} photographs. For photographs not "
                                   "passed, the phenotype is the measurement that raises the risk most.", True))
                files = gr.File(file_count="multiple", file_types=["image"], type="filepath", label="Photographs")
                cohorts = gr.Textbox(label="Cohort labels: one for all, or one per photograph separated by commas", placeholder="Camera A")
                with gr.Row():
                    batch_examples = gr.Button("Load the example photographs", elem_classes="eq-outlined")
                    run_batch_button = gr.Button("Score photographs", variant="primary")
                batch_summary = gr.HTML(padding=False)
                with gr.Row():
                    phenotype_plot = plot()
                    profile_plot = plot()
                batch_table = table(("Photograph", "Cohort", "Poor-quality risk", "Decision", "Phenotype"), label="Results")
                batch_csv = gr.File(label="Download results (input for the Harmonization and Reader agreement tabs)")
                batch_examples.click(example_batch, None, [files, cohorts])
                run_batch_button.click(run_batch, [files, cohorts], [batch_summary, batch_table, batch_csv, phenotype_plot, profile_plot])

            with gr.Tab("Harmonization"):
                gr.HTML(ui.section("Batch variation and harmonization", "In the study the ten measurements predicted the cohort with balanced accuracy 0.760; ComBat "
                                   "lowered this to 0.380 but kept at least 90% of the defect response for only 3 of 9 defects, so EyeQC uses uncorrected measurements.", True))
                with gr.Row():
                    combat_csv = gr.File(file_types=[".csv"], type="filepath", label="CSV with cohort and the ten measurements (from Cohort analysis)", scale=3, elem_id="eq-combat-file")
                    with gr.Column(scale=2):
                        method = gr.Radio(["ComBat", "location-scale"], value="ComBat", label="Method")
                        grade_column = gr.Textbox(label="Optional grade column for disease association", placeholder="e.g. dr_grade")
                with gr.Row():
                    gr.DownloadButton("CSV template", value=research.template("cohorts"), elem_classes="eq-outlined")
                    run_combat = gr.Button("Analyse", variant="primary")
                combat_note = gr.Markdown()
                with gr.Row():
                    separability_plot = plot()
                    variance_plot = plot()
                with gr.Row():
                    before_plot = plot()
                    after_plot = plot()
                association_plot = plot()
                combat_out = gr.File(label="Download corrected measurements")
                run_combat.click(lambda f, m, g: guard(research.harmonize, f, m, g), [combat_csv, method, grade_column],
                                 [combat_note, separability_plot, before_plot, after_plot, variance_plot, association_plot, combat_out])

            # ------------------------------------------------ local validation and recalibration
            with gr.Tab("Recalibration"):
                gr.HTML(ui.section("Local validation and recalibration", "Upload risk (EyeQC poor-quality risk), is_poor (your reference: 1 poor, 0 acceptable) and "
                                   "optionally participant. Features, classifier and thresholds are never changed.", True))
                with gr.Row():
                    labels_csv = gr.File(file_types=[".csv"], type="filepath", label="Labelled CSV", scale=3, elem_id="eq-labels-file")
                    with gr.Column(scale=1):
                        gr.DownloadButton("CSV template", value=research.template("labels"), elem_classes="eq-outlined")
                with gr.Tabs(elem_classes="eq-subtabs"):
                    with gr.Tab("Validate the frozen model"):
                        run_validate = gr.Button("Validate", variant="primary")
                        validate_note = gr.Markdown()
                        with gr.Row():
                            validate_table = table(("Metric", "Value", "95% CI"), label="Performance on your photographs")
                            validate_plot = plot()
                        run_validate.click(lambda f: guard(research.validate, f), labels_csv, [validate_note, validate_table, validate_plot])
                    with gr.Tab("Recalibrate"):
                        with gr.Row(equal_height=True):
                            labelled = gr.Slider(25, 200, value=100, step=25, label="Locally labelled photographs used for fitting", scale=3)
                            run_recal = gr.Button("Recalibrate", variant="primary", scale=1)
                        recal_note = gr.Markdown()
                        recal_table = table(("Metric", "Frozen", "Recalibrated", "Recalibrated: 95% of splits", "Local PASS cut-off only"), label="Medians over repeated splits, held-out photographs")
                        with gr.Row():
                            recal_curve = plot()
                            recal_boxes = plot()
                        run_recal.click(lambda f, n: guard(research.recalibrate, f, int(n)), [labels_csv, labelled], [recal_note, recal_table, recal_curve, recal_boxes])

            with gr.Tab("Reader agreement"):
                gr.HTML(ui.section("Agreement with your graders", "One row per photograph: a column per grader (PASS / REVIEW / FAIL), a column eyeqc with EyeQC's "
                                   "decision, optionally risk and participant. The Cohort analysis download already contains eyeqc-compatible verdict and risk columns.", True))
                with gr.Row():
                    reader_csv = gr.File(file_types=[".csv"], type="filepath", label="Grades CSV", scale=3, elem_id="eq-reader-file")
                    with gr.Column(scale=2):
                        grader_columns = gr.Textbox(label="Grader columns (leave empty to detect them)", placeholder="grader_1, grader_2")
                        gr.DownloadButton("CSV template", value=research.template("readers"), elem_classes="eq-outlined")
                run_readers = gr.Button("Compute agreement", variant="primary")
                reader_note = gr.Markdown()
                with gr.Row():
                    reader_matrix = plot()
                    reader_difference = plot()
                reader_table = table(("Grader", "κ with EyeQC", "Mean κ with other graders", "Difference"), label="Each grader")
                reader_group = table(("Raters", "Fleiss κ", "Krippendorff α (ordinal)", "ICC(3,1)"), label="All raters together")
                reader_file = gr.File(label="Download")
                run_readers.click(lambda f, g: guard(research.readers, f, g), [reader_csv, grader_columns],
                                  [reader_note, reader_matrix, reader_table, reader_difference, reader_group, reader_file])

            # ------------------------------------------------ study results
            with gr.Tab("Study results") as study_tab:
                gr.HTML(ui.section("Results reported in the paper", "Stored aggregate results from the public cohorts; nothing here is computed from your uploads. "
                                   "Validation is retrospective.", True))
                gr.HTML(dashboard.headline(), padding=False)
                with gr.Tabs(elem_classes="eq-subtabs"):
                    with gr.Tab("Validation") as tab:
                        forest, bars, cohort_table = dashboard.validation()
                        with gr.Row():
                            first_charts = (shown(forest), shown(bars))
                        table(value=cohort_table)
                        redraw_on_open(tab, *first_charts)
                        redraw_on_open(study_tab, *first_charts)
                    with gr.Tab("Reader study"):
                        gr.HTML(dashboard.reader_study(), padding=False)
                    with gr.Tab("Representation & comparisons") as tab:
                        auroc, brier, probes = dashboard.representation()
                        with gr.Row():
                            pair = (shown(auroc), shown(brier))
                        redraw_on_open(tab, *pair, shown(probes), shown(dashboard.comparisons()))
                        gr.Markdown("Comparison methods were evaluated with their available implementations and predictions; they differ in training data and "
                                    "adaptation (see the paper's supplementary methods). MCF-Net did not cover every photograph in four cohorts.")
                    with gr.Tab("Distribution shift") as tab:
                        error, passed, pairs = dashboard.shift()
                        with gr.Row():
                            redraw_on_open(tab, shown(error), shown(passed))
                        gr.Markdown("Points are medians of 100 participant-separated splits; bars span 95% of the splits.")
                        gr.HTML(pairs, padding=False)
                    with gr.Tab("Degradations & maps") as tab:
                        monotonic, dense, tiled = dashboard.defects()
                        with gr.Row():
                            pair = (shown(dense), shown(tiled))
                        redraw_on_open(tab, *pair, shown(monotonic))
                        gr.Markdown("Maps were evaluated on 400 photographs with simulated defects at the highest severity; dense-map values come from the 200 "
                                    "photographs not used to choose the window size.")
                    with gr.Tab("Cohort variation & speed") as tab:
                        phenotype, separability, retention, disease = dashboard.cohort_variation()
                        top = shown(phenotype)
                        with gr.Row():
                            pair = (shown(separability), shown(retention))
                        redraw_on_open(tab, top, *pair, shown(disease))
                        gr.HTML(dashboard.speed(), padding=False)

            # ------------------------------------------------ optional third-party tools
            with gr.Tab("Vessels & FLAIR"):
                gr.HTML(ui.section("Optional research tools", "Third-party models applied to the photograph open in the EyeQC tab. They are not part of EyeQC, do not "
                                   "change its score and were not validated in the study.", False))
                with gr.Tabs(elem_classes="eq-subtabs"):
                    with gr.Tab("Retinal vessels (RRWNet)"):
                        run_vessel = gr.Button("Segment vessels", variant="primary")
                        vessel_note = gr.Markdown()
                        with gr.Row():
                            vessel_overlay = gr.Image(label="Arteries (red) and veins (blue)", height=380)
                            vessel_heat = gr.Image(label="Vessel probability", height=380)
                            vessel_lines = gr.Image(label="Centreline", height=380)
                        vessel_table = table(("Measurement", "Value"), label="Vessel measurements")
                        run_vessel.click(extras.vessel_analysis, current, [vessel_overlay, vessel_heat, vessel_lines, vessel_table, vessel_note])
                    with gr.Tab("FLAIR image-text prompts"):
                        run_categories = gr.Button("Score five fixed quality phrases", elem_classes="eq-outlined")
                        category_plot = plot()
                        category_note = gr.Markdown()
                        run_categories.click(extras.flair_categories, current, [category_plot, category_note])
                        with gr.Row():
                            quality_prompt = gr.Textbox("blurred retinal photograph with uneven illumination", label="Prompt 1 (quality)")
                            pathology_prompt = gr.Textbox("diabetic retinopathy with hemorrhages and exudates", label="Prompt 2 (pathology)")
                        more_prompts = gr.Textbox(label="More prompts, one per line (optional)", lines=2)
                        with gr.Row(equal_height=True):
                            explain = gr.Checkbox(True, label="Compute Grad-CAM maps for prompts 1 and 2")
                            run_flair = gr.Button("Compare prompts", variant="primary")
                        flair_note = gr.Markdown()
                        with gr.Row():
                            flair_plot = plot()
                            flair_table = table(("Prompt", "Relative weight", "Similarity logit"))
                        with gr.Row():
                            flair_quality = gr.Image(label="Grad-CAM, prompt 1")
                            flair_pathology = gr.Image(label="Grad-CAM, prompt 2")
                            flair_animation = gr.Image(label="Alternating between the two", format="gif")
                        run_flair.click(extras.flair_prompts, [current, quality_prompt, pathology_prompt, more_prompts, explain],
                                        [flair_plot, flair_table, flair_note, flair_quality, flair_pathology, flair_animation])

            with gr.Tab("About"):
                gr.HTML(about.page(), padding=False)
        gr.HTML("<div class='eq-disclaimer'>Research use only. EyeQC is not a medical device. Do not upload identifiable patient data.</div>")
    return demo


demo = build()

if __name__ == "__main__":
    engine.warm_up()
    demo.queue(default_concurrency_limit=1, max_size=16).launch(
        server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), ssr_mode=False, footer_links=[],
        allowed_paths=[str(HERE / "examples")],        # the example photographs, wherever the app is started from
        css=(HERE / "style.css").read_text(), theme=gr.themes.Base(primary_hue="teal"), head=DARK_MODE)
