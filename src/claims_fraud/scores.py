from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from claims_fraud.models import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch score claims for fraud risk.")
    parser.add_argument(
        "--input", type=str, help="Path to input claims CSV file", required=True
    )
    parser.add_argument(
        "--model", type=str, help="Path to saved model joblib file", required=True
    )
    parser.add_argument(
        "--output", type=str, help="Path to save output scored CSV", required=True
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Decision threshold for fraud classification",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load input data safely
    print(f"Loading input claims data from: {args.input}")
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input}'.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"Error reading input CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate presence of required identifier columns
    required_ids = ["claim_id", "policy_id", "garage_id", "adjuster_id"]
    for col in required_ids:
        if col not in df.columns:
            print(
                f"Error: Required column '{col}' is missing from input data.",
                file=sys.stderr,
            )
            sys.exit(1)

    # 2. Load model safely
    print(f"Loading model artifact from: {args.model}")
    try:
        model = load_model(args.model)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{args.model}'.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Enforce leakage boundary
    post_assessment_columns = [
        "investigation_opened",
        "days_to_settle",
        "amount_paid_xaf",
        "fraud_flag",
    ]
    print("Enforcing leakage boundary by dropping post-assessment columns...")
    features_df = df.drop(columns=post_assessment_columns, errors="ignore")

    # 4. Generate predictions
    print("Generating fraud risk probabilities...")
    try:
        prob = model.predict_proba(features_df)[:, 1]
    except Exception as e:  # noqa: BLE001
        print(f"Error during model inference: {e}", file=sys.stderr)
        sys.exit(1)

    fraud_flag = (prob >= args.threshold).astype(int)

    # 5. Build results table and export
    result = pd.DataFrame(
        {
            "claim_id": df["claim_id"],
            "policy_id": df["policy_id"],
            "garage_id": df["garage_id"],
            "adjuster_id": df["adjuster_id"],
            "fraud_prob": prob,
            "fraud_flag": fraud_flag,
        }
    )

    print(f"Exporting scored results to: {args.output}")
    try:
        # Ensure output directory exists if a nested path is provided
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result.to_csv(output_path, index=False)
        print("Batch scoring completed successfully!")
    except Exception as e:  # noqa: BLE001
        print(f"Error saving output CSV: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
