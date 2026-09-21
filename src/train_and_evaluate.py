"""Train classifiers and evaluate creditworthiness predictions.

Usage:
    python src/train_and_evaluate.py data/credit_data.csv

Outputs:
 - results/metrics.txt (text summary)
 - results/roc_curves.png
 - models/best_model.joblib
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["debt_to_income"] = df["debts"] / (df["income"] + 1)
    df["log_income"] = np.log1p(df["income"])
    df["log_debts"] = np.log1p(df["debts"])
    # One-hot housing
    df = pd.get_dummies(df, columns=["housing_status"], drop_first=True)
    return df


def train_and_evaluate(path: str):
    df = pd.read_csv(path)
    df = feature_engineer(df)

    X = df.drop(columns=["default"]) 
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": RandomForestClassifier(n_estimators=1, max_depth=5, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    roc_data = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        auc = roc_auc_score(y_test, probs)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        acc = accuracy_score(y_test, preds)

        results[name] = {
            "roc_auc": float(auc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "accuracy": float(acc),
        }

        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_data[name] = (fpr.tolist(), tpr.tolist())

    # pick best model by ROC AUC
    best_name = max(results.keys(), key=lambda k: results[k]["roc_auc"])
    best_model = models[best_name]

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    # Save metrics
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump({"results": results, "best_model": best_name}, fh, indent=2)

    # Save a human-readable summary
    with open(out_dir / "metrics.txt", "w") as fh:
        fh.write(f"Best model: {best_name}\n\n")
        for n, r in results.items():
            fh.write(f"Model: {n}\n")
            fh.write(json.dumps(r) + "\n\n")

    # ROC plot
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curves.png")

    # Save best model and scaler
    joblib.dump(best_model, Path("models") / "best_model.joblib")
    joblib.dump(scaler, Path("models") / "scaler.joblib")

    print(f"Saved metrics to {out_dir / 'metrics.json'} and ROC image.")
    print(f"Best model: {best_name} saved to models/best_model.joblib")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    args = parser.parse_args()

    train_and_evaluate(args.data_path)


if __name__ == "__main__":
    main()
