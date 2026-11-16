# Student Retention ML

A compact applied-ML project for predicting student retention risk from tabular academic and engagement features.

## Focus

- reproducible preprocessing with scikit-learn pipelines
- mixed numeric/categorical features
- stratified cross-validation
- probability-based classification
- model evaluation beyond accuracy

## Run

```bash
pip install pandas numpy scikit-learn pytest
python projects/student-retention-ml/train.py
```

The demo uses a synthetic dataset generated in code so the repository stays reproducible and does not publish student data.

## ML workflow

1. Generate a small synthetic retention dataset.
2. Split numeric and categorical features.
3. Impute missing values and one-hot encode categories.
4. Train a class-weighted Random Forest.
5. Evaluate ROC AUC, F1, precision, and recall.

This project is intentionally small enough to inspect quickly while still showing the structure of a practical tabular ML workflow.
