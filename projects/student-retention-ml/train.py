from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.append(str(Path(__file__).parent))
from features import make_retention_dataset


NUMERIC = ["attendance", "average_grade", "study_hours", "missed_assignments"]
CATEGORICAL = ["program", "commuter"]


def build_pipeline() -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessing = ColumnTransformer(
        [("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)]
    )

    model = RandomForestClassifier(
        n_estimators=250,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([("preprocess", preprocessing), ("model", model)])


def train_and_evaluate(frame: pd.DataFrame) -> dict[str, float]:
    X = frame[NUMERIC + CATEGORICAL]
    y = frame["dropped_out"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = pipeline.predict(X_test)

    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    return {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "f1": float(report["1"]["f1-score"]),
        "precision": float(report["1"]["precision"]),
        "recall": float(report["1"]["recall"]),
    }


if __name__ == "__main__":
    metrics = train_and_evaluate(make_retention_dataset())
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")
