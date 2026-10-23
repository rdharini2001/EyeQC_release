# EyeQC — precise manuscript additions

## Main text: within-eye acquisition comparison

In DeepDRiD, we further compared separate photographs of the same eye using image-specific artefact, clarity and field-definition annotations. Among pairs with differing annotations, EyeQC assigned higher poor-quality risk to the photograph with greater artefact severity in 83.9% of pairs, lower clarity in 71.0% and poorer field definition in 78.2%. An exploratory combined measure showed concordant ranking in 80.6% of informative pairs. These results support sensitivity to acquisition-level quality differences, although they do not establish improved clinical reacquisition decisions.

**Placement:** Replace the final sentence of the existing operating-threshold paragraph or place this after the independent quality-reference paragraph. Keep the 66.4% exact category agreement and 30.1% PASS↔FAIL changes in the Supplementary Information; they are important counterpoints.

## Supplementary Methods: within-eye quality ranking

We compared 792 pairs of separate same-eye photographs from DeepDRiD using image-specific artefact severity, clarity and field-definition scores. Among pairs with differing annotations, we tested whether EyeQC assigned greater poor-quality risk to the photograph with poorer annotated quality. Concordance was calculated separately for each measure and for an exploratory combined severity measure. Confidence intervals were estimated by participant-clustered bootstrap resampling. This analysis assessed acquisition-to-acquisition quality ranking, not prospective reacquisition decisions.

## Supplementary table

| Independent image-specific quality reference | Informative pairs | Concordant ranking | 95% participant-clustered CI |
|---|---:|---:|---:|
| Artefact severity | 465 | 83.9% | 80.4–87.1% |
| Clarity | 462 | 71.0% | 66.7–75.3% |
| Field definition | 418 | 78.2% | 74.1–82.1% |
| Exploratory combined severity | 728 | 80.6% | 77.7–83.3% |

Combined severity was calculated from standardized artefact severity minus standardized clarity and field-definition scores. Image order was imbalanced: the second photograph had better combined annotations in approximately 71% of informative pairs, so include the direction-balanced accuracy (76.8%) and order-adjusted comparator (9.6 percentage-point improvement) if reporting the combined measure. These estimates were produced in the prior audit and should be reproduced with a committed analysis script before submission.

## Limitations paragraph addition

Although EyeQC detected quality differences between separate acquisitions, these retrospective comparisons cannot establish whether its recommendations reduce unnecessary repeats, improve reader consistency or improve patient outcomes. These outcomes require prospective evaluation with independent assessments of image adequacy.

## Authentic spatial figure caption

**Image-derived spatial quality maps.** Local readability was calculated in six-by-six retinal tiles and interpolated for display over each fundus photograph. These maps show spatial variation in an interpretable image measurement. They are neither manually annotated pixel-level quality masks nor explanations of every feature contributing to the calibrated EyeQC risk.

## Two factual corrections

- The original experimental scripts refer to `eyeqc_v2.backbone` and `eyeqc_v2.hybrid_model`, which are missing from the supplied code ZIP. The measured ten-axis formulas are available; recover the frozen model's original embedding preprocessing before asserting exact reproducibility from source photographs.
- Keep FLAIR and vessel analysis as optional web modules, outside the paper's validated predictor. The QuickQual and EFIQA-CP comparisons do not demonstrate universal superiority; report cohort-specific rankings and differing baseline training protocols.

## Suggested next experiment

Evaluate exported EyeQC local maps on held-out real photographs with expert defect-region annotations, ideally alongside EFIQA-CP, with area under pixel-level precision–recall, intersection-over-union and failure examples. A randomized clinician-assisted reading study is required before claiming that EyeQC reduces between-reader variability.
