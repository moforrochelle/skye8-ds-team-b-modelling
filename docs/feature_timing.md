# Feature Timing

The model is meant to rank a claim when it is first filed. Because of this, I only want to use information that would be available at the time the claim is being scored.

| Column | Can we use it? | Reason |
|---|---|---|
| `claim_id` | No | It is just an identifier and does not provide useful information for the model. |
| `policy_id` | No | It is an identifier. We use the information from the policy instead. |
| `garage_id` | No as a feature | It is an identifier. We use the available garage information instead. |
| `adjuster_id` | No as a feature | It is an identifier. Adjuster information is only used if the adjuster has already been assigned. |
| `incident_type` | Yes | Known when the claim is filed. |
| `incident_date` | Yes | Known when the claim is filed. |
| `incident_hour` | Yes | Known when the claim is filed. |
| `reported_date` | Yes | Known when the claim is reported. |
| `claim_amount_xaf` | Yes | Known when the claim is filed. |
| `police_report` | Yes | Available as part of the claim information. |
| `witness_count` | Yes | Available when the claim is filed. |
| `prior_claims_holder` | Yes | This information is already available about the policy holder. |
| `vehicle_towed` | Yes | Known as part of the claim. |
| `investigation_opened` | No | This happens after the claim has entered the investigation process. |
| `days_to_settle` | No | We only know this after the claim has been settled. |
| `amount_paid_xaf` | No | We only know this after payment. |
| `fraud_flag` | No | This is the target we are trying to predict. |

## Policy information

Policy information can be used because it is already available when a claim is made.

Some of the policy information we use includes:

- vehicle make
- vehicle year
- cover type
- sum insured
- annual premium
- policy start date
- payment frequency
- registered year

We also use this information to create features such as policy age and vehicle age.

## Garage information

Garage information can be used if the garage is already known when the claim is scored.

We use information such as the garage name, town, approval status and number of bays. The `garage_id` itself is not used as a numeric feature.

## Adjuster information

Adjuster information is only used if the adjuster has already been assigned when the claim is scored.

We use information such as the adjuster's region, hiring year and caseload band. The `adjuster_id` itself is not used as a numeric feature.

If the adjuster is assigned later, that information should not be used for scoring the claim.

## Features we created

From the information available at claim time, we created features including:

- reporting delay
- policy age
- vehicle age
- claim amount to sum insured ratio
- incident month
- incident day of the week
- night-hour indicator
- late-report indicator
- negative reporting delay indicator
- negative policy age indicator

We also created historical features for repeated entities, such as the number of previous claims made by a policy holder and the holder's claim frequency.

## Entity history

Claims can be related because the same policy holder, garage or adjuster can appear in multiple claims.

For this reason, we use historical information such as previous claim counts and claim frequency. The important rule is that the history for a claim must only contain information that would have been available before that claim.

When training the model, the history features are calculated using the appropriate training data so that information from the validation data does not leak into the features.

## Leakage rule

The main rule I followed when creating the features is:

**If the information becomes available after the claim has been assessed, investigated or settled, it should not be used by the model.**

This means that `investigation_opened`, `days_to_settle`, `amount_paid_xaf` and `fraud_flag` are excluded from the model.

The preprocessing and encoding steps are also part of the training pipeline, so they are fitted on the training data rather than on the full dataset before the split.