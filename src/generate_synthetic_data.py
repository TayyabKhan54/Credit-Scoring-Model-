"""Generate a synthetic credit dataset and save to CSV.

Usage:
    python src/generate_synthetic_data.py --output data/credit_data.csv
"""
import argparse
import numpy as np
import pandas as pd


def build_dataset(n=5000, random_state=42):
    rng = np.random.default_rng(random_state)
    income = np.exp(rng.normal(10.5, 0.8, n))  # positive, skewed
    age = rng.integers(21, 75, size=n)
    debts = np.round(rng.normal(20000, 15000, n).clip(0), 2)
    num_loans = rng.integers(0, 8, size=n)
    credit_history_years = rng.integers(0, 30, size=n)
    missed_payments = rng.poisson(0.3, size=n)
    payment_history_score = rng.integers(300, 851, size=n)
    savings = np.round(rng.normal(5000, 8000, n).clip(0), 2)
    employment_years = rng.integers(0, 40, size=n)
    housing_status = rng.choice(["rent", "own", "mortgage"], size=n, p=[0.4, 0.35, 0.25])

    # Risk heuristic -> probability of default
    dti = debts / (income + 1)
    score = (
        -0.8 * np.log1p(income)
        + 2.5 * dti
        + 0.3 * missed_payments
        - 0.01 * payment_history_score
        - 0.0001 * savings
        - 0.02 * employment_years
        + 0.5 * (housing_status == "rent")
    )

    prob_default = 1 / (1 + np.exp(-score))
    default = (rng.random(n) < prob_default).astype(int)

    df = pd.DataFrame(
        {
            "income": np.round(income, 2),
            "age": age,
            "debts": debts,
            "num_loans": num_loans,
            "credit_history_years": credit_history_years,
            "missed_payments": missed_payments,
            "payment_history_score": payment_history_score,
            "savings": savings,
            "employment_years": employment_years,
            "housing_status": housing_status,
            "default": default,
        }
    )

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/credit_data.csv")
    parser.add_argument("--rows", type=int, default=5000)
    args = parser.parse_args()

    df = build_dataset(n=args.rows)
    # ensure data dir exists
    df.to_csv(args.output, index=False)
    print(f"Saved synthetic dataset to {args.output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
