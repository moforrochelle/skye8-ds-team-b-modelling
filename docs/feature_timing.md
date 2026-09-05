# Feature Timing and Leakage Rules

## Scoring point

A claim is ready for fraud scoring when it is reported.

We use `reported_date` as the main time point for deciding which information can be used by the model.

A feature can be used only if the information would be known at or before `reported_date`.

Information that becomes available after the claim is reported must not be used for the first fraud score.

---

## Claim information

The following claim information is available when the claim is reported:

* `incident_type`
* `incident_hour`
* `police_report`
* `witness_count`
* `prior_claims_holder`
* `vehicle_towed`
* `claim_amount_xaf`

These features are used as claim-level inputs to the model.

---

## Policy information

Policy information is used only when it is available at the time of scoring.

The model uses:

* `region`
* `vehicle_make`
* `vehicle_year`
* `cover_type`
* `sum_insured_xaf`
* `annual_premium_xaf`
* `payment_frequency`
* `policy_start`

These values describe the policy that is linked to the claim.

---

## Holder history

Holder history is based on `reported_date`.

For a claim, only earlier claims from the same holder can be used.

A claim reported on the same date is not treated as an earlier claim.

The model uses:

* `holder_claim_count`
* `holder_history_days`
* `holder_claim_frequency`

During validation, history is calculated separately for each training and validation set.

For training claims, only other training claims are used to create the history.

For validation claims, only the training data is used as the history source.

This prevents validation claims from changing the history used by the model during training.

---

## Garage information

Garage information can be used only if the garage is already known when the claim is scored.

The dataset does not contain a timestamp showing when a garage was assigned to a claim. Because of this, we cannot directly confirm that the garage was known at `reported_date`.

The garage features used by the model are:

* `town`
* `registered_year`
* `bay_count`
* `approved`

`garage_id` is not used directly as a model feature.

`garage_name` was removed from the final feature set. It could act as a garage identifier, and the grouped-by-garage test showed that keeping it did not improve PR-AUC.

If the garage is assigned after `reported_date`, these garage features must not be used for the first fraud score.

---

## Adjuster information

Adjuster information can be used only if the adjuster is already known when the claim is scored.

The dataset does not contain a timestamp showing when an adjuster was assigned to a claim. Because of this, we cannot directly confirm that the adjuster was known at `reported_date`.

The model uses:

* `adjuster_region`
* `hired_year`
* `caseload_band`

`adjuster_id` is not used directly as a model feature.

If the adjuster is assigned after `reported_date`, these features must not be used for the first fraud score.

---

## Engineered features

The model also uses features created from information that should be available at scoring time.

These include:

* `claim_amount_clean`
* `claim_to_insured_ratio`
* `reporting_delay`
* `policy_age_days`
* `vehicle_age`
* `incident_month`
* `incident_dayofweek`
* `night_hour`
* `late_report`
* `negative_reporting_delay`
* `negative_policy_age`

These features are calculated from the claim, policy, and date information available to the model.

---

## Information that must not be used

The following fields are not used as model features because they are only known after the claim has been processed or investigated:

* `investigation_opened`
* `days_to_settle`
* `amount_paid_xaf`
* `fraud_flag`

`fraud_flag` is the target that the model is trying to predict.

These fields must not enter the model feature matrix.

---

## Data preparation during validation

All data preparation steps must be fitted using the training data only.

This includes:

* missing-value filling
* scaling
* categorical encoding
* any target encoding
* resampling
* model fitting

The validation data must not be used to fit these steps.

This keeps the validation results honest.

---

## Temporal evaluation

The model is evaluated using `reported_date` so that future claims are not used to predict earlier claims.

The data is divided into:

* Development period: January 2023 to December 2025
* Temporal evaluation: January 2026 to June 2026
* Final holdout: July 2026 to December 2026

The final holdout is kept separate until the final evaluation.

For the January–June 2026 evaluation, holder history is created using only claims from the development period.

For the July–December 2026 final holdout, holder history is created using claims available before July 2026.

The final holdout must not be used to fit the model or any preprocessing steps before the final score is calculated.

---

## Evaluation results

The model was checked in several ways. These numbers were regenerated from a clean run of the current pipeline (`notebooks/validation_tuning.ipynb`, balanced-class-weight logistic baseline, `scikit-learn==1.9.0` as pinned in `pyproject.toml`) — regenerate again before final submission if the feature set or model changes.

Random 5-fold cross-validation gave:

* PR-AUC: `0.1078 ± 0.0145`
* ROC-AUC: `0.7748 ± 0.0156`

Grouped-by-garage cross-validation gave:

* PR-AUC: `0.0544 ± 0.0357`
* ROC-AUC: `0.5740 ± 0.1328`

The January–June 2026 temporal evaluation gave:

* PR-AUC: `0.1143`
* ROC-AUC: `0.7539`

The final July–December 2026 holdout gave:

* PR-AUC: `0.1669`
* ROC-AUC: `0.7949`

The grouped-by-garage result is much lower than the random and temporal results. This shows that model performance is weaker when the model has to work with garages that were not seen during training.

The final holdout result is kept separate because it is the final test of the model on later claims.
