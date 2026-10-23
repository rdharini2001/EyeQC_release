# EyeQC: evidence audit of `data_final.zip`

**Purpose.** Assess what the delivered data can establish now, identify analyses that strengthen the clinical quality-control story, and distinguish those results from studies requiring new data. This report is a research audit, not a claim of prospective clinical effectiveness.

## 1. Archive inventory and limits

The archive contains 114 entries, including model JSON files, 47 table CSV files, parquet data and feature caches. The EyeQC v3 JSON supplies the trained logistic coefficients, normalization parameters, Platt parameters, development manifest hash and PASS/FAIL thresholds (0.3452816755/0.3860883789). It does **not** include original fundus pixels, per-pixel spatial maps, segmentation masks, inference/feature-extraction source code or prospective aided-reader outcomes. Some parquet files could not be decoded in the audit environment; the new analysis below uses directly readable CSV files. The embedded JSON incorrectly calls all five evaluation cohorts 'external'; EyeQ-test is held out within the development source dataset, leaving four independent external cohorts.

## 2. New quantitative analysis: same-eye ranking of image-specific quality

**Data:** `tables/deepdrid_test_retest_pairs_v3.csv`, 792 matched same-eye pairs from 396 participants. The binary dataset adequacy label is identical within every pair, so it is not a valid outcome for determining which of the two acquisitions is preferable. However, expert-assigned image-specific artifact-severity, clarity and field-definition annotations vary within many pairs.

We used the convention that higher artifact scores, lower clarity and lower field definition mean worse image quality. Within each pair, EyeQC was counted as concordant when the photograph with the worse expert annotation also had higher EyeQC predicted poor-quality risk. Pairs tied on a particular expert dimension were excluded from that dimension's comparison.

| Expert dimension | Informative pairs | Concordant pairs | Concordance | 95% patient-clustered bootstrap CI | Spearman rho between within-pair differences |
|---|---:|---:|---:|---|---:|
| Artifact severity | 465 | 390 | 83.9% | 80.4–87.1% | 0.660 |
| Clarity | 462 | 328 | 71.0% | 66.7–75.3% | 0.473 |
| Field definition | 418 | 327 | 78.2% | 74.1–82.1% | 0.285 |
| Exploratory composite | 728 | 587 | 80.6% | 77.7–83.3% | 0.609 |

The exploratory composite equals the change in standardized artifact severity minus the changes in standardized clarity and field definition, standardized using the pooled observations from the two images across the 792 pairs; it is **not** a validated clinical adequacy endpoint. Across the 728 non-tied composite pairs, the AUROC of the difference in EyeQC risk for identifying the worse-acquisition direction was **0.836 (95% participant-clustered bootstrap CI 0.801–0.869)**; balanced accuracy was **76.8% (73.2–80.3%)**. The second image had better composite annotations in 71.0% of pairs; therefore simply always choosing image 2 would score 71.0%, and EyeQC's 80.6% concordance improves on that index-only comparator by **9.6 percentage points (95% CI 5.4–13.8)**. Image order is imbalanced, and this benchmark must not be described as random chance.

**Interpretation:** This is meaningful image-level construct-validity evidence. EyeQC tracks clinically graded differences between separate photographs of the same eye even when the source dataset's binary adequacy label is the same. It supports an exploratory *quality-ranking* claim and a rationale for prospective acquisition trials; it does **not** prove readers act more consistently, reduce reacquisition, or improve patient outcomes.

**Figure:** `EyeQC_real_within_eye_quality_ranking.png` and editable `EyeQC_real_within_eye_quality_ranking.svg` were generated directly from the CSV data (not synthetic or AI-drawn fundus images).

## 3. What is demonstrated, and what is not

| Manuscript claim | What the archive provides | Remaining evidence to close the gap |
|---|---|---|
| Interpretable representation contributes beyond DINOv2 | Ten-measure vs DINOv2 vs hybrid ablations; better hybrid Brier score by 0.002–0.011 across cohorts, but not uniformly better AUROC. Some measures poorly predicted by embeddings. | Bootstrap matched per-cohort differences where individual predictions are available; demonstrate defect-specific utility without inferring causal interpretability from coefficients alone. |
| Spatial maps localize acquisition defects | Aggregate and raw summary outcomes for synthetic glare, shadow and local blur; severe point hit 94.8%, 95.0% and 35.2%, respectively. | Original retinal photographs, actual per-pixel EyeQC map arrays, defect masks or independently annotated clinical maps; paired Dice/IoU/AUPRC and appropriate comparators. Archive does not contain those arrays or photographs. |
| EyeQC reduces **between-reader** variability | 3 unaided graders, 80 cases, model-reader κ, group agreement with EyeQC treated as fourth rater. | Randomized/counterbalanced assisted-versus-unaided reader study using the **same graders**, an independent reference, and paired group-level disagreement and inappropriate-PASS outcomes. Adding the model as fourth rater is not an assistance experiment. |
| EyeQC reduces **between-institution** decision variability | Cohort classification from quality features, scanner subgroups, harmonization results. | Graders or imaging workflows at multiple institutions evaluating the same cases under usual versus EyeQC-assisted conditions, adjusting for case mix/camera/site. Statistical harmonization is not clinical standardization. |
| EyeQC improves reacquisition decisions | 792 real same-eye acquisition pairs and separate image-specific expert subscores; new observational ranking analysis above. | Verified acquisition order, per-image diagnostic adequacy/adjudication, technician decisions and preferably a prospective randomized assisted-acquisition pilot. Do not treat the shared eye-level label as independent per-image adequacy. |
| Improves patient care | No clinical interventional outcome data. | Prospective screened-encounter outcomes: adequate images on first visit, unnecessary repeats, false accepts, recall rates, time to diagnosis, and downstream detection safety. |
| Reproducible spatial model | Model JSON and DINOv2/RETFound embedding caches; numerical repeatability on 10 images. | Executable source defining ten measures and map computation, image preprocessing, weights/checksums, environment/hardware details, and source sample images. |

## 4. Additional significant limitations to report rather than hide

- DeepDRiD fixed PASS NPV is 64.0%; among 978 PASS images, approximately 352 are reference-poor. This is an important safety issue for autonomous acceptance; keep the intended use advisory and apply site-specific validation.
- The REVIEW band covers only 0.5–1.8% of images. It is an operating interval, not calibrated uncertainty or abstention.
- 30.1% of same-eye pairs switch directly PASS versus FAIL. This may reflect genuine acquisition differences, but the binary source labels are duplicated across pair photographs; the new subscore analysis helps contextualize the switches without resolving clinical diagnostic adequacy.
- Existing spatial localizations are convincing mainly for severe glare and shadow; weaker local blur, and low-severity results, must be presented rather than removed.
- Published baseline rankings vary by cohort; differences in native training protocols and quality endpoints limit claims of superiority.
- EyeQ-test shares the development data source and should not be counted among the four independent cohorts.

## 5. Submission-effective additions

**Suggested main-text sentence** (replace part of the current repeat-image passage rather than extend the 1,500-word Brief Communication):

> In an exploratory analysis of 792 pairs of separate same-eye photographs from DeepDRiD, EyeQC ranked the image with worse expert-assessed artifact severity as higher risk in 83.9% of 465 informative pairs. Across 728 pairs differing on a composite of image-specific quality annotations, directional agreement was 80.6% (95% participant-clustered bootstrap CI, 77.7–83.3%), supporting sensitivity to acquisition-level quality differences without establishing clinical reacquisition benefit (Supplementary Fig. Sx).

**Suggested Supplementary Methods:**

> Same-eye acquisition pairs were compared using the released image-specific artefact-severity, clarity and field-definition annotations. For each property, we determined whether the EyeQC risk difference agreed with the direction of poorer expert quality, excluding pairs with identical scores for that property. An exploratory combined severity difference was calculated as the sum of standardized artefact severity minus standardized clarity and field-definition values; feature standard deviations were estimated from the pooled paired observations. Concordance, balanced accuracy and rank AUROC were reported. Uncertainty was estimated using 5,000 bootstrap resamples clustered by the 396 participants. Because the binary quality label was identical between repeated images and capture order was not independently verified, this analysis was interpreted as image-quality ranking, not a test of clinical reacquisition decisions.

**Recommended prioritized studies:** (1) export true EyeQC spatial maps for a clinically annotated independent pixel-level benchmark; (2) independently adjudicate diagnostic adequacy for paired acquisitions and evaluate a safe image-selection rule; (3) run a counterbalanced assisted-versus-unaided reader study with safety outcomes; (4) move to a technician workflow pilot only after fixed-threshold acceptance safety is documented.

## 6. Key source files

- `data_final/tables/deepdrid_test_retest_pairs_v3.csv` (792 paired acquisition records and image-specific quality subscores)
- `data_final/tables/Table_1_external_validation_v3.csv` (cohort performance and PASS NPV)
- `data_final/tables/ablation_v1_v2_v3_full_v3.csv` (matched feature configurations)
- `data_final/tables/reader_equivalence_v3.csv`, `reader_statistics_v2_noninferiority.csv` (reader-level agreement)
- `data_final/tables/localization_full_v3_summary.csv` (dose-specific localization)
- `data_final/models/eyeqc_v3_model.json` (development details and thresholds)

Audit date: 2026-10-23. All new numbers in Section 2 were computed from uploaded CSV values; results should be checked by the study statistician and incorporated as exploratory analyses. No simulated image data or spatial maps were invented.
