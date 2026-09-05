"""Validation and tuning utilities for Team B claims-fraud modelling.

Owner: Ngwa Densey - Validation and Tuning.
The implementation follows docs/protocol.md / feature_timing.md:
- reported_date is the scoring time;
- holder history is strictly prior-date and fold-safe;
- preprocessing is fitted inside each fold through sklearn Pipelines;
- PR-AUC is the headline metric;
- grouped validation uses garage_id, the identified clustering entity;
- the final holdout is never used for tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
POST_ASSESSMENT = {
    "investigation_opened",
    "days_to_settle",
    "amount_paid_xaf",
    "fraud_flag",
}
HISTORY_COLUMNS = [
    "holder_claim_count",
    "holder_history_days",
    "holder_claim_frequency",
]


@dataclass(frozen=True)
class DataBundle:
    model_data: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series


def load_data(data_dir: str | Path) -> DataBundle:
    """Load and merge the four raw project datasets and build the feature matrix."""
    data_dir = Path(data_dir)
    claims = pd.read_csv(data_dir / "claims.csv").drop_duplicates().copy()
    policies = pd.read_csv(data_dir / "policies.csv")
    garages = pd.read_csv(data_dir / "garages.csv")
    adjusters = pd.read_csv(data_dir / "adjusters.csv")

    model_data = claims.merge(policies, on="policy_id", how="left")
    model_data = model_data.merge(garages, on="garage_id", how="left")
    model_data = model_data.merge(adjusters, on="adjuster_id", how="left")

    for raw, clean in [
        ("incident_date", "incident_date_clean"),
        ("reported_date", "reported_date_clean"),
        ("policy_start", "policy_start_clean"),
    ]:
        model_data[clean] = pd.to_datetime(
            model_data[raw], format="mixed", dayfirst=True, errors="coerce"
        )

    X = pd.DataFrame(index=model_data.index)
    X["incident_type"] = model_data["incident_type"]
    X["incident_hour"] = model_data["incident_hour"]
    X["police_report"] = model_data["police_report"]
    X["witness_count"] = model_data["witness_count"]
    X["prior_claims_holder"] = model_data["prior_claims_holder"]
    X["vehicle_towed"] = model_data["vehicle_towed"]
    X["region"] = model_data["region_x"]
    X["vehicle_make"] = model_data["vehicle_make"]
    X["vehicle_year"] = model_data["vehicle_year"]
    X["cover_type"] = model_data["cover_type"]
    X["sum_insured_xaf"] = model_data["sum_insured_xaf"]
    X["annual_premium_xaf"] = model_data["annual_premium_xaf"]
    X["payment_frequency"] = model_data["payment_frequency"]
    X["town"] = model_data["town"]
    X["registered_year"] = model_data["registered_year"]
    X["bay_count"] = model_data["bay_count"]
    X["approved"] = model_data["approved"]
    X["adjuster_region"] = model_data["region_y"]
    X["hired_year"] = model_data["hired_year"]
    X["caseload_band"] = model_data["caseload_band"]

    X["claim_amount_clean"] = (
        model_data["claim_amount_xaf"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("XAF", "", regex=False)
        .str.strip()
        .replace("nan", np.nan)
        .astype(float)
    )
    X["claim_to_insured_ratio"] = (
        X["claim_amount_clean"] / model_data["sum_insured_xaf"]
    )
    X["reporting_delay"] = (
        model_data["reported_date_clean"] - model_data["incident_date_clean"]
    ).dt.days
    X["policy_age_days"] = (
        model_data["incident_date_clean"] - model_data["policy_start_clean"]
    ).dt.days
    X["vehicle_age"] = (
        model_data["incident_date_clean"].dt.year - model_data["vehicle_year"]
    )
    X["incident_month"] = model_data["incident_date_clean"].dt.month
    X["incident_dayofweek"] = model_data["incident_date_clean"].dt.dayofweek
    X["night_hour"] = (X["incident_hour"] < 6) | (X["incident_hour"] >= 22)
    X["late_report"] = X["reporting_delay"] > 7
    X["negative_reporting_delay"] = X["reporting_delay"] < 0
    X["negative_policy_age"] = X["policy_age_days"] < 0

    assert not POST_ASSESSMENT.intersection(X.columns), (
        "Post-assessment columns reached the model matrix: "
        f"{sorted(POST_ASSESSMENT.intersection(X.columns))}"
    )
    y = model_data["fraud_flag"].map({"NO": 0, "YES": 1}).astype(int)
    return DataBundle(model_data=model_data, X=X, y=y)


def add_holder_history_features(
    scoring_data: pd.DataFrame, history_source: pd.DataFrame
) -> pd.DataFrame:
    """Create holder history using only strictly earlier reported dates."""
    scoring = scoring_data[["holder_id", "reported_date_clean"]].copy()
    source = history_source[["holder_id", "reported_date_clean"]].copy()
    source = source[source["reported_date_clean"].notna()].copy()
    scoring["_original_index"] = scoring.index

    source_counts = (
        source.groupby(["holder_id", "reported_date_clean"])
        .size()
        .reset_index(name="claims_on_date")
        .sort_values(["reported_date_clean", "holder_id"])
    )
    source_counts["cumulative_claims"] = source_counts.groupby("holder_id")[
        "claims_on_date"
    ].cumsum()

    scoring_sorted = scoring.sort_values(["reported_date_clean", "holder_id"])
    source_sorted = source_counts.sort_values(["reported_date_clean", "holder_id"])
    result = pd.merge_asof(
        scoring_sorted,
        source_sorted[["holder_id", "reported_date_clean", "cumulative_claims"]],
        by="holder_id",
        on="reported_date_clean",
        direction="backward",
        allow_exact_matches=False,
    )
    result["holder_claim_count"] = result["cumulative_claims"].fillna(0)
    first_dates = (
        source.groupby("holder_id")["reported_date_clean"]
        .min()
        .rename("holder_first_prior_reported_date")
        .reset_index()
    )
    result = result.merge(first_dates, on="holder_id", how="left")
    result["holder_history_days"] = (
        result["reported_date_clean"] - result["holder_first_prior_reported_date"]
    ).dt.days
    no_prior_claims = result["holder_claim_count"] == 0
    result.loc[no_prior_claims, "holder_first_prior_reported_date"] = pd.NaT
    result.loc[no_prior_claims, "holder_history_days"] = np.nan
    result["holder_claim_frequency"] = result["holder_claim_count"] / result[
        "holder_history_days"
    ].replace(0, np.nan)
    result.loc[result["reported_date_clean"].isna(), HISTORY_COLUMNS] = np.nan
    return result.set_index("_original_index")[HISTORY_COLUMNS]


def make_logistic_pipeline(class_weight: str | None = "balanced") -> Pipeline:
    numeric = [
        "incident_hour",
        "witness_count",
        "prior_claims_holder",
        "vehicle_year",
        "sum_insured_xaf",
        "annual_premium_xaf",
        "registered_year",
        "bay_count",
        "hired_year",
        "claim_amount_clean",
        "claim_to_insured_ratio",
        "reporting_delay",
        "policy_age_days",
        "vehicle_age",
        "incident_month",
        "incident_dayofweek",
        *HISTORY_COLUMNS,
    ]
    categorical = [
        "incident_type",
        "police_report",
        "vehicle_towed",
        "region",
        "vehicle_make",
        "cover_type",
        "payment_frequency",
        "town",
        "approved",
        "adjuster_region",
        "caseload_band",
        "night_hour",
        "late_report",
        "negative_reporting_delay",
        "negative_policy_age",
    ]
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000, class_weight=class_weight, random_state=RANDOM_STATE
                ),
            ),
        ]
    )


def make_xgb_pipeline(params: dict[str, Any] | None = None) -> Pipeline:
    """XGBoost pipeline; hyperparameters are supplied by Optuna."""
    params = params or {}
    numeric = [
        "incident_hour",
        "witness_count",
        "prior_claims_holder",
        "vehicle_year",
        "sum_insured_xaf",
        "annual_premium_xaf",
        "registered_year",
        "bay_count",
        "hired_year",
        "claim_amount_clean",
        "claim_to_insured_ratio",
        "reporting_delay",
        "policy_age_days",
        "vehicle_age",
        "incident_month",
        "incident_dayofweek",
        *HISTORY_COLUMNS,
    ]
    categorical = [
        "incident_type",
        "police_report",
        "vehicle_towed",
        "region",
        "vehicle_make",
        "cover_type",
        "payment_frequency",
        "town",
        "approved",
        "adjuster_region",
        "caseload_band",
        "night_hour",
        "late_report",
        "negative_reporting_delay",
        "negative_policy_age",
    ]
    preprocessor = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=4,
        random_state=RANDOM_STATE,
        verbosity=0,
        **params,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _fold_history(
    X_data: pd.DataFrame,
    model_data: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = X_data.iloc[train_idx].copy()
    X_valid = X_data.iloc[valid_idx].copy()
    train_source = model_data.loc[X_train.index]
    valid_rows = model_data.loc[X_valid.index]
    train_history = add_holder_history_features(train_source, train_source)
    valid_history = add_holder_history_features(valid_rows, train_source)
    X_train[HISTORY_COLUMNS] = train_history[HISTORY_COLUMNS]
    X_valid[HISTORY_COLUMNS] = valid_history[HISTORY_COLUMNS]
    return X_train, X_valid


def evaluate_cv(
    X_data: pd.DataFrame,
    y: pd.Series,
    model: Pipeline,
    model_data: pd.DataFrame,
    scheme: str,
    groups: pd.Series | None = None,
    n_splits: int = 5,
) -> pd.DataFrame:
    if scheme == "random":
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
        )
        splits = splitter.split(X_data, y)
    elif scheme == "grouped":
        if groups is None:
            raise ValueError("groups are required for grouped validation")
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
        )
        splits = splitter.split(X_data, y, groups=groups)
    else:
        raise ValueError("scheme must be 'random' or 'grouped'")

    rows: list[dict[str, float | int | str]] = []
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        X_train, X_valid = _fold_history(X_data, model_data, train_idx, valid_idx)
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        fitted = clone(model).fit(X_train, y_train)
        p = fitted.predict_proba(X_valid)[:, 1]
        pred = (p >= 0.5).astype(int)
        rows.append(
            {
                "scheme": scheme,
                "fold": fold,
                "n_train": len(train_idx),
                "n_valid": len(valid_idx),
                "pr_auc": average_precision_score(y_valid, p),
                "roc_auc": roc_auc_score(y_valid, p),
                "precision_at_0_5": precision_score(y_valid, pred, zero_division=0),
                "recall_at_0_5": recall_score(y_valid, pred, zero_division=0),
                "prevalence": float(y_valid.mean()),
            }
        )
    return pd.DataFrame(rows)


def temporal_evaluation(bundle: DataBundle, model: Pipeline) -> pd.DataFrame:
    """Train on development (<2026-01-01), evaluate Jan-Jun 2026."""
    dates = bundle.model_data["reported_date_clean"]
    train_mask = dates < pd.Timestamp("2026-01-01")
    eval_mask = (dates >= pd.Timestamp("2026-01-01")) & (
        dates < pd.Timestamp("2026-07-01")
    )
    X_train = bundle.X.loc[train_mask].copy()
    X_eval = bundle.X.loc[eval_mask].copy()
    y_train = bundle.y.loc[train_mask]
    y_eval = bundle.y.loc[eval_mask]
    train_history = add_holder_history_features(
        bundle.model_data.loc[train_mask], bundle.model_data.loc[train_mask]
    )
    eval_history = add_holder_history_features(
        bundle.model_data.loc[eval_mask], bundle.model_data.loc[train_mask]
    )
    X_train[HISTORY_COLUMNS] = train_history[HISTORY_COLUMNS]
    X_eval[HISTORY_COLUMNS] = eval_history[HISTORY_COLUMNS]
    fitted = clone(model).fit(X_train, y_train)
    p = fitted.predict_proba(X_eval)[:, 1]
    pred = (p >= 0.5).astype(int)
    return pd.DataFrame(
        [
            {
                "scheme": "temporal",
                "fold": 1,
                "n_train": len(X_train),
                "n_valid": len(X_eval),
                "pr_auc": average_precision_score(y_eval, p),
                "roc_auc": roc_auc_score(y_eval, p),
                "precision_at_0_5": precision_score(y_eval, pred, zero_division=0),
                "recall_at_0_5": recall_score(y_eval, pred, zero_division=0),
                "prevalence": float(y_eval.mean()),
            }
        ]
    )


def summarize_validation(
    random_df: pd.DataFrame, grouped_df: pd.DataFrame, temporal_df: pd.DataFrame
) -> pd.DataFrame:
    def summary(df: pd.DataFrame, scheme: str) -> dict[str, Any]:
        return {
            "scheme": scheme,
            "PR-AUC": (
                f"{df.pr_auc.mean():.4f} ± {df.pr_auc.std(ddof=0):.4f}"
                if len(df) > 1
                else f"{df.pr_auc.iloc[0]:.4f}"
            ),
            "ROC-AUC": (
                f"{df.roc_auc.mean():.4f} ± {df.roc_auc.std(ddof=0):.4f}"
                if len(df) > 1
                else f"{df.roc_auc.iloc[0]:.4f}"
            ),
            "fraud prevalence": f"{df.prevalence.mean():.4f}",
        }

    return pd.DataFrame(
        [
            summary(random_df, "Random stratified 5-fold"),
            summary(grouped_df, "Grouped by garage 5-fold"),
            summary(temporal_df, "Temporal Jan-Jun 2026"),
        ]
    )


def make_optuna_objective(
    X_data: pd.DataFrame, y: pd.Series, model_data: pd.DataFrame, groups: pd.Series
) -> Any:
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 280),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 15.0),
        }
        scores: list[float] = []
        for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(X_data, y, groups=groups), start=1
        ):
            X_train, X_valid = _fold_history(X_data, model_data, train_idx, valid_idx)
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            model = make_xgb_pipeline(params)
            model.fit(X_train, y_train)
            p = model.predict_proba(X_valid)[:, 1]
            score = average_precision_score(y_valid, p)
            scores.append(score)
            trial.report(float(np.mean(scores)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    return objective


def run_tuning(
    bundle: DataBundle, study_path: str | Path, n_trials: int = 60
) -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_path = Path(study_path)
    study_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{study_path.resolve()}"
    study = optuna.create_study(
        study_name="team_b_xgb_grouped_pr_auc",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    dates = bundle.model_data["reported_date_clean"]
    dev_mask = dates < pd.Timestamp("2026-01-01")
    X_dev = bundle.X.loc[dev_mask].copy()
    y_dev = bundle.y.loc[dev_mask]
    groups = bundle.model_data.loc[dev_mask, "garage_id"]
    study.optimize(
        make_optuna_objective(X_dev, y_dev, bundle.model_data.loc[dev_mask], groups),
        n_trials=n_trials,
    )
    return study


def save_optimization_plot(study: optuna.Study, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trials = study.trials
    completed_values = [
        t.value
        for t in trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not completed_values:
        raise ValueError("No completed Optuna trials to plot")
    values = np.array(completed_values, dtype=float)
    best = np.maximum.accumulate(values)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(best) + 1), best, marker="o", markersize=2)
    plt.xlabel("Completed trial")
    plt.ylabel("Best grouped PR-AUC")
    plt.title("Optuna best score by trial")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def last_20_gain(study: optuna.Study) -> dict[str, float | int]:
    completed_values = [
        t.value
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if len(completed_values) < 20:
        raise ValueError("At least 20 completed trials are required")
    values = np.array(completed_values, dtype=float)
    overall_best = float(values.max())
    pre20_best = float(values[:-20].max()) if len(values) > 20 else float(values[0])
    return {
        "completed_trials": len(completed_values),
        "best_pr_auc": overall_best,
        "best_before_final_20": pre20_best,
        "final_20_marginal_gain": overall_best - pre20_best,
    }


if __name__ == "__main__":
    raise SystemExit(
        "Import this module from the project notebook/script; see notebooks/validation_tuning.ipynb."
    )


def collect_oof_predictions(
    X_data: pd.DataFrame,
    y: pd.Series,
    model: Pipeline,
    model_data: pd.DataFrame,
    scheme: str = "grouped",
    groups: pd.Series | None = None,
) -> pd.DataFrame:
    """Return out-of-fold probabilities for threshold/precision-recall analysis."""
    if scheme == "random":
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        splits = splitter.split(X_data, y)
    elif scheme == "grouped":
        if groups is None:
            raise ValueError("groups are required for grouped validation")
        splitter = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=RANDOM_STATE
        )
        splits = splitter.split(X_data, y, groups=groups)
    else:
        raise ValueError("scheme must be 'random' or 'grouped'")

    rows: list[dict[str, Any]] = []
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        X_train, X_valid = _fold_history(X_data, model_data, train_idx, valid_idx)
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        fitted = clone(model).fit(X_train, y_train)
        probabilities = fitted.predict_proba(X_valid)[:, 1]
        for idx, truth, probability in zip(X_valid.index, y_valid, probabilities):
            rows.append(
                {
                    "index": idx,
                    "fold": fold,
                    "y_true": int(truth),
                    "probability": float(probability),
                }
            )
    return pd.DataFrame(rows).set_index("index").sort_index()


def threshold_report(
    oof: pd.DataFrame, thresholds: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50)
) -> pd.DataFrame:
    """Report precision/recall at candidate operating thresholds."""
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        pred = (oof["probability"] >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(oof["y_true"], pred, zero_division=0),
                "recall": recall_score(oof["y_true"], pred, zero_division=0),
                "flag_rate": float(pred.mean()),
                "pr_auc": average_precision_score(oof["y_true"], oof["probability"]),
            }
        )
    return pd.DataFrame(rows)


def compare_class_weighting(
    X_data: pd.DataFrame,
    y: pd.Series,
    model_data: pd.DataFrame,
) -> pd.DataFrame:
    """Compare no class weighting versus balanced weighting under grouped CV."""
    rows = []
    for label, weight in [("none", None), ("balanced", "balanced")]:
        result = evaluate_cv(
            X_data,
            y,
            make_logistic_pipeline(class_weight=weight),
            model_data,
            "grouped",
            groups=model_data["garage_id"],
        )
        rows.append(
            {
                "class_weight": label,
                "PR-AUC": result["pr_auc"].mean(),
                "ROC-AUC": result["roc_auc"].mean(),
                "precision_at_0_5": result["precision_at_0_5"].mean(),
                "recall_at_0_5": result["recall_at_0_5"].mean(),
            }
        )
    return pd.DataFrame(rows)
