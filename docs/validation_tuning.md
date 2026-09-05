# Validation and Tuning 

## Scope

This document covers the Validation & Tuning vertical in Project Brief 2. It does not replace Rochelle's feature-engineering ownership or Rooney's ensemble/error-analysis ownership.

## Validation contract

The fraud target is approximately 3% of claims, so PR-AUC is the headline metric and the fraud prevalence is reported as the baseline. The same feature set and baseline model are evaluated under three schemes:

1. **Random stratified 5-fold** — estimates IID-style performance and is useful as a diagnostic, but it can be optimistic when repeated entities occur in both train and validation folds.
2. **Grouped 5-fold by `garage_id`** — keeps all claims from a garage in one fold and tests generalisation to unseen garages. This is the project's clustering stress test.
3. **Temporal evaluation** — train on development claims reported before 2026-01-01 and evaluate claims reported from 2026-01-01 through 2026-06-30.

Holder history is recalculated inside every fold. Training history uses only training rows, while validation history uses the training rows as its source. Same-day claims are not treated as earlier history.

All imputation, scaling and categorical encoding are inside the sklearn pipeline and therefore fitted only on each training fold.

## Reference results in the team protocol

`docs/feature_timing.md` records these results, regenerated from a clean run of the current pipeline:

| Scheme | PR-AUC | ROC-AUC |
|---|---:|---:|
| Random 5-fold | 0.1078 ± 0.0145 | 0.7748 ± 0.0156 |
| Grouped by garage 5-fold | 0.0544 ± 0.0357 | 0.5740 ± 0.1328 |
| Jan–Jun 2026 temporal | 0.1143 | 0.7539 |

These are **reference values from the team's protocol**, not values to copy blindly. The validation implementation should be run against the exact feature/model pipeline merged into the shared repository.

The grouped result is much lower than the random result, demonstrating that ordinary random CV can materially overstate performance when garage-level clustering is ignored. The temporal result is closer to the production question because it evaluates later claims using earlier claims for training.

## Tuning contract

Tuning starts only after the three-scheme validation table exists. The supplied implementation tunes XGBoost against **grouped validation PR-AUC**, using a persisted Optuna SQLite study. The default configuration is 60 trials. The search includes tree count, depth, learning rate, row/column subsampling, minimum child weight, gamma, L1/L2 regularisation and positive-class weighting.

The search uses a 3-fold grouped split to keep the search computationally practical. The winning configuration must then be re-evaluated using the project's full 5-fold grouped validation and temporal evaluation before being described as an improvement. This explicitly tests whether a tuning gain survives the stricter grouped scheme.

The study database is persisted and the best-score-by-trial plot is saved. `last_20_gain()` reports the marginal improvement obtained from the final twenty completed trials.

**Reproducibility note:** `XGBClassifier` is run with `n_jobs=4`. Multi-threaded XGBoost is not bit-for-bit deterministic across machines/core counts even with `random_state` fixed, because histogram summation order depends on thread scheduling. In a 60-trial study with a `MedianPruner`, this can be enough to flip individual prune/keep decisions, and because `TPESampler` conditions on trial history, one flipped decision changes every subsequent trial's suggested parameters. In practice this means **the exact best trial/hyperparameters found by a fresh 60-trial run are expected to differ between machines**, even though the code, seed, and data are identical — confirmed by comparing two clean runs (one landed on `n_estimators=113, max_depth=7`, PR-AUC 0.0444; the other on `n_estimators=91, max_depth=2`, PR-AUC 0.0512). The three-scheme validation table above is not affected the same way and does reproduce exactly. Whichever machine produces the tuned model that gets used downstream should have its `outputs/` committed as the source of truth, rather than expecting a rerun elsewhere to match it.

## Imbalance comparison

The implementation compares `class_weight=None` against `class_weight="balanced"` under grouped validation and reports PR-AUC, precision and recall. Threshold adjustment is handled separately with out-of-fold probabilities so that precision/recall can be compared across candidate thresholds without fitting on validation labels.

No resampling is performed outside a fold. If SMOTE is added later, it must be placed inside an imbalanced-learn pipeline and executed only on the training portion of each fold.

## Data coverage

`claims.csv` runs from 7 January 2023 to 8 December 2026, so the July–December 2026 final holdout described in the brief is fully populated (July through November are complete calendar months; December is complete through the 8th, the last date in the file). The final-holdout numbers reported in `docs/feature_timing.md` reflect this full window, not a partial slice.

## Deliverables produced

- `src/claims_fraud/validation_tuning.py`
- `notebooks/validation_tuning.ipynb`
- `notebooks/run_validation_tuning.py`
- `docs/validation_tuning.md`
- `outputs/` generated validation/tuning artifacts
