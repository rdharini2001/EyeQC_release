# Example photographs

From the DeepDRiD dataset (regular fundus images), https://github.com/deepdrdoc/DeepDRiD, licensed CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/). The files are unmodified copies, renamed with a `deepdrid_` prefix.

Liu R, Wang X, Wu Q, et al. DeepDRiD: Diabetic Retinopathy - Grading and Image Quality Estimation Challenge.
Patterns. 2022;3:100512. https://doi.org/10.1016/j.patter.2022.100512

## How they were chosen

| Use | Rule | DeepDRiD overall label |
|---|---|---|
| PASS example (`348_r1`) | the photograph nearest the median risk among those labelled adequate, with no artifact, clarity and field ratings of at least 8 and no writing outside the retinal field | adequate |
| REVIEW example (`71_r2`) | illustrates the intermediate category | inadequate |
| FAIL example (`154_r2`) | illustrates a FAIL that agrees with the label | inadequate |
| Same-eye pair (`1_l1`, `1_l2`) | first participant whose two photographs of one eye differ in artifact rating and receive opposite decisions | both inadequate (artifact severity 4 and 0), so the PASS disagrees with the overall label |

They illustrate the interface and are not a sample of performance; DeepDRiD is the cohort in which EyeQC performed worst.

`examples.json` and `pair.json` store the risk and decision reported for each photograph in the study; `tests/test_model.py`
checks that this code reproduces them.
