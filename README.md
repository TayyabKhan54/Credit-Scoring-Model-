# Creditworthiness Classification Demo

This small project generates a synthetic credit dataset, trains several classifiers (Logistic Regression, Decision Tree-ish RandomForest, and RandomForest), and evaluates them with Precision, Recall, F1, and ROC-AUC.

Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\Activate.ps1 or Activate.bat
pip install -r requirements.txt
```

2. Generate synthetic data:

```bash
python src/generate_synthetic_data.py --output data/credit_data.csv --rows 5000
```

3. Train and evaluate:

```bash
python src/train_and_evaluate.py data/credit_data.csv
```

Outputs will be saved under `results/` and the best model under `models/`.

Next steps you might ask for:
- Replace synthetic data with your real dataset (CSV with same columns).
- Add cross-validation and hyperparameter tuning (GridSearchCV) per model.
- Create a notebook with visual EDA and feature importance plots.
