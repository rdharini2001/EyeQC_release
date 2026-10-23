import pandas as pd
from pathlib import Path

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
T3 = Path(r"F:\EyeQC\EyeQC_npj\EyeQC_v2_robust\data_final\tables")


def show(title, name, cols=None):
    print(f"\n=== {title} ===")
    p = T3 / name
    if not p.exists():
        print("MISSING:", p)
        return
    df = pd.read_csv(p)
    if cols:
        df = df[cols]
    print(df.round(4).to_string(index=False))


show("reader_eyeqc_comparison_v3", "reader_eyeqc_comparison_v3.csv")
show("reader_operating_thresholds_v3", "reader_operating_thresholds_v3.csv")
show("construct_validity_v3", "construct_validity_v3.csv")
show("institution_scanner_subdomain_validation_v3", "institution_scanner_subdomain_validation_v3.csv")
show("deepdrid_stratum_auroc_v3", "deepdrid_stratum_auroc_v3.csv")
show("baseline_heuristic_comparison_v3", "baseline_heuristic_comparison_v3.csv")
show("decision_curve_policy_summary_v3", "decision_curve_policy_summary_v3.csv")
show("harmonization_safety_combat_v3", "harmonization_safety_combat_v3.csv")
show("harmonization_safety_location_scale_v3", "harmonization_safety_location_scale_v3.csv")
show("degradation_monotonicity_v3 (400-image)", "degradation_monotonicity_v3.csv")
show("throughput_benchmark_v3_gridformat", "throughput_benchmark_v3_gridformat.csv")
show("feature_latency_percentiles_v3", "feature_latency_percentiles_v3.csv")
show("compound_degradation_interactions_v3", "compound_degradation_interactions_v3.csv")

print("\n=== v1_vs_v3 false-fail/pass (from earlier session) ===")
try:
    print(pd.read_csv(T3.parent.parent / "results_final" / "tables" / "v1_v2_v3_false_fail_pass.csv").round(4).to_string(index=False))
except Exception as e:
    print("could not read:", e)
