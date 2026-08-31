# Evaluation Protocol

## Purpose

This protocol defines how Team B will evaluate fraud-detection models before
results are examined. Any change requires a pull request with a written reason.

## Headline metric

The primary metric is PR-AUC, calculated as average precision. Every result must
show the fraud prevalence in its evaluation set beside PR-AUC.

Accuracy must not be used as a headline metric because fraud is rare.

Secondary metrics are ROC-AUC, precision, recall, and later precision at k.

## Data and leakage controls

- The target is `fraud_flag`, mapped from `YES`/`NO` to `1`/`0`.
- Dates will be parsed with explicit, tested rules for each raw date format.
  Unparseable or ambiguous values must be reported; they must not be silently
  coerced.
- Time-based evaluation uses `reported_date`, because it represents the point at
  which a new claim is available for scoring.
- All preprocessing, imputation, encoding, scaling, resampling, and target
  encoding must be fitted only on the training partition of each fold.
- Post-assessment fields identified in `docs/feature_timing.md` are excluded
  from every model matrix and protected by a test.

## Validation schemes

All schemes use identical cleaned features, target definition, and model settings.

1. **Random stratified cross-validation**
   - `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
   - Measures performance when claims are assumed independent.

2. **Grouped cross-validation**
   - `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`
   - Group: `garage_id`
   - Measures generalisation to garages whose claims were not seen in training.

3. **Temporal validation**
   - Claims are ordered by parsed `reported_date`.
   - The final six calendar months are held out and not used for model selection
     or tuning.
   - For development reporting, train on claims before the six months preceding
     that final holdout, then evaluate on that preceding six-month period.
   - After the model and threshold are frozen, retrain on all data before the
     final holdout and evaluate once on the final six months.

The three validation results will be reported side by side. The production-facing
estimate will be the final temporal-holdout PR-AUC, with grouped-CV performance
used as a required robustness check.

## Tuning protocol

- No tuning begins until baseline results exist for all three validation schemes.
- Baselines are prevalence, logistic regression, and a single decision tree.
- Optuna will run at least 60 persisted trials.
- The tuning objective is mean grouped-CV PR-AUC on pre-holdout training data.
- The selected configuration must also be evaluated under random, grouped, and
  temporal validation.
- We will report total compute, the best-score-by-trial plot, and the marginal
  gain from trials 41–60.

## Reproducibility

- Random seed: `42`.
- Evaluation code must run from a clean checkout through the documented command.
- Each result table records dataset period, split scheme, model version, feature
  version, and metrics.
- The Project 1 evaluation protocol is the default team standard. Any deviation
  from it will be added below with its justification.

## Approved deviations from Project 1 protocol

None recorded yet.