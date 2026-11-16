from __future__ import annotations

import numpy as np
import pandas as pd


def make_retention_dataset(n_samples: int = 800, seed: int = 42) -> pd.DataFrame:
    """Create a reproducible synthetic student-retention dataset."""
    rng = np.random.default_rng(seed)

    attendance = rng.normal(78, 12, n_samples).clip(20, 100)
    avg_grade = rng.normal(72, 11, n_samples).clip(35, 100)
    study_hours = rng.gamma(2.2, 2.0, n_samples).clip(0, 25)
    missed_assignments = rng.poisson(1.5, n_samples)
    program = rng.choice(["CS", "Business", "Science", "Arts"], n_samples)
    commuter = rng.choice(["yes", "no"], n_samples, p=[0.4, 0.6])

    risk_score = (
        -0.045 * (attendance - 75)
        -0.055 * (avg_grade - 70)
        -0.08 * study_hours
        + 0.35 * missed_assignments
        + 0.25 * (commuter == "yes")
        + rng.normal(0, 0.7, n_samples)
    )

    probability = 1 / (1 + np.exp(-risk_score))
    dropped_out = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "attendance": attendance.round(1),
            "average_grade": avg_grade.round(1),
            "study_hours": study_hours.round(1),
            "missed_assignments": missed_assignments,
            "program": program,
            "commuter": commuter,
            "dropped_out": dropped_out,
        }
    )
