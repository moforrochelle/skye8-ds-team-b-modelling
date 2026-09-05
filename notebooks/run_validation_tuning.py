import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from claims_fraud.validation_tuning import (
    evaluate_cv,
    last_20_gain,
    load_data,
    make_logistic_pipeline,
    make_xgb_pipeline,
    run_tuning,
    save_optimization_plot,
    summarize_validation,
    temporal_evaluation,
)

DATA = ROOT / "data" / "raw"

OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

bundle = load_data(DATA)

print("Rows after duplicate removal:", len(bundle.model_data))
print("Fraud prevalence:", bundle.y.mean())

logistic = make_logistic_pipeline(class_weight="balanced")
random_df = evaluate_cv(bundle.X, bundle.y, logistic, bundle.model_data, "random")
grouped_df = evaluate_cv(
    bundle.X,
    bundle.y,
    logistic,
    bundle.model_data,
    "grouped",
    groups=bundle.model_data["garage_id"],
)
temporal_df = temporal_evaluation(bundle, logistic)

random_df.to_csv(OUT / "random_cv_folds.csv", index=False)
grouped_df.to_csv(OUT / "grouped_cv_folds.csv", index=False)
temporal_df.to_csv(OUT / "temporal_evaluation.csv", index=False)
summary = summarize_validation(random_df, grouped_df, temporal_df)
summary.to_csv(OUT / "validation_comparison.csv", index=False)
print("\nVALIDATION COMPARISON")
print(summary.to_string(index=False))

# Tune only on the development period and using grouped validation.
study = run_tuning(bundle, OUT / "optuna_xgb_grouped.db", n_trials=60)
print("\nBest trial:", study.best_trial.number)
print("Best grouped PR-AUC:", study.best_value)
print("Best parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

save_optimization_plot(study, OUT / "optuna_best_score_by_trial.png")
gain = last_20_gain(study)
pd.DataFrame([gain]).to_csv(OUT / "optuna_last_20_gain.csv", index=False)
print("\nLast-20-trial marginal gain:", gain["final_20_marginal_gain"])

# Final tuned model can be reconstructed with study.best_params.
tuned_model = make_xgb_pipeline(study.best_params)
print("\nTuned model ready:", tuned_model)
