"""Evaluate every model on the same held-out patients and write a run record."""

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_score

from .data import CLASSES, CLASS_NAMES
from .features import FEATURES
from .models import build_models, patient_split, rule_predict


def evaluate(data, audit, out, seed=42, test_size=0.25, synthetic=False, cv_folds=5):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    train_ids, test_ids = patient_split(data, seed=seed, test_size=test_size)
    train, test = data.iloc[train_ids], data.iloc[test_ids]
    x_train, x_test = train[FEATURES], test[FEATURES]
    models = build_models(seed)
    reports = {}
    predictions = test[["patient", "breath_id", "target"]].copy()
    for name, estimator in models.items():
        estimator.fit(x_train, train.target)
        predictions[name] = estimator.predict(x_test)
    predictions["Feature Rules"] = rule_predict(x_test)
    for name in [*models, "Feature Rules"]:
        report = classification_report(
            test.target, predictions[name], labels=CLASSES, target_names=CLASS_NAMES,
            output_dict=True, zero_division=0,
        )
        reports[name] = {
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "dta_recall": report["DTA"]["recall"],
            "classification_report": report,
            "confusion_matrix": confusion_matrix(test.target, predictions[name], labels=CLASSES).tolist(),
        }
    cv = None
    if cv_folds:
        if not 2 <= cv_folds <= train.patient.nunique():
            raise ValueError("cv_folds must be 0 or between 2 and the number of training patients")
        folds = list(GroupKFold(n_splits=cv_folds).split(x_train, train.target, train.patient))
        for fold_train, fold_valid in folds:
            if set(train.iloc[fold_train].target.unique()) != set(CLASSES) or set(train.iloc[fold_valid].target.unique()) != set(CLASSES):
                raise ValueError("Each cross-validation fold needs all three classes; check patient coverage")
        scores = cross_val_score(
            models["Logistic Regression"], x_train, train.target, cv=folds,
            scoring="f1_macro", error_score="raise",
        )
        cv = {"model": "Logistic Regression", "method": "GroupKFold on training patients only",
              "folds": cv_folds, "macro_f1_by_fold": scores.tolist(), "mean_macro_f1": float(scores.mean())}
    record = {
        "data_kind": "synthetic demonstration" if synthetic else "user-supplied data",
        "seed": seed, "cleaning": audit, "features": FEATURES,
        "split": {
            "method": "GroupShuffleSplit by patient", "validation_patient_fraction": test_size,
            "training_breaths": len(train), "validation_breaths": len(test),
            "training_patients": train.patient.nunique(), "validation_patients": test.patient.nunique(),
            "patient_overlap": len(set(train.patient) & set(test.patient)),
            "validation_class_counts": {name: int((test.target == label).sum()) for label, name in zip(CLASSES, CLASS_NAMES)},
        },
        "prediction_adjustment": "none; each breath predicted independently",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
        "models": reports, "cross_validation": cv,
    }
    (out / "metrics.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    predictions.to_csv(out / "predictions.csv", index=False)
    return record
