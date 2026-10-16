"""Classical baselines, a feature-rule baseline, and patient-separated splits."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import CLASSES


def build_models(seed: int = 42) -> dict:
    return {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty="l1", solver="saga", max_iter=5000, random_state=seed),
        ),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }


def rule_predict(features: pd.DataFrame) -> np.ndarray:
    """Apply the coursework's feature thresholds independently to each breath.

    This implementation has no previous-row prediction propagation.
    """
    ratio = features.tve_tvi_ratio.to_numpy()
    dta = (features.e_time.to_numpy() < 0.3) & (
        (ratio < 0.25) | ((ratio < 0.5) & (features.tve.to_numpy() < 100))
    )
    return np.where(dta, 2, np.where(ratio < 0.9, 1, 0))


def patient_split(data: pd.DataFrame, seed: int = 42, test_size: float = 0.25):
    """Reserve entire patients, rather than splitting their breaths across sets."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if data.patient.nunique() < 4:
        raise ValueError("Patient-separated evaluation requires at least four patients")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train, test = next(splitter.split(data, data.target, groups=data.patient))
    for indices, name in [(train, "training"), (test, "validation")]:
        if set(data.iloc[indices].target.unique()) != set(CLASSES):
            raise ValueError(f"The {name} split must contain all three classes; check data coverage and split settings")
    if len(train) < 5:
        raise ValueError("KNN requires at least five training breaths")
    return train, test
