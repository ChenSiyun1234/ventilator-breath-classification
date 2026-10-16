import json

import numpy as np
import pandas as pd
import pytest

from ventilator.evaluation import evaluate
from ventilator.features import FEATURES
from ventilator.models import build_models, patient_split, rule_predict


def test_patient_split_is_disjoint_and_reproducible(demo_data):
    data = demo_data[3]
    train, valid = patient_split(data, seed=42)
    assert not set(data.iloc[train].patient) & set(data.iloc[valid].patient)
    assert len(train) + len(valid) == len(data)
    assert np.array_equal(valid, patient_split(data, seed=42)[1])
    assert set(data.iloc[train].target) == {0, 1, 2}
    assert set(data.iloc[valid].target) == {0, 1, 2}


def test_scaler_uses_training_distribution_only(demo_data):
    data = demo_data[3]
    train, valid = patient_split(data)
    x_train = data.iloc[train][FEATURES].copy()
    x_valid = data.iloc[valid][FEATURES].copy() + 10000
    model = build_models()["Logistic Regression"]
    model.fit(x_train, data.iloc[train].target)
    mean_before = model[0].mean_.copy()
    model.predict(x_valid)
    np.testing.assert_allclose(mean_before, x_train.mean(axis=0))
    np.testing.assert_array_equal(model[0].mean_, mean_before)


def test_rule_thresholds_have_explicit_boundary_behavior():
    features = pd.DataFrame({
        "tve_tvi_ratio": [0.1, 0.4, 0.4, 0.9, 0.1, 0.5],
        "e_time": [0.2, 0.2, 0.2, 1, 0.3, 0.2],
        "tve": [200, 99, 100, 400, 20, 20],
    })
    np.testing.assert_array_equal(rule_predict(features), [2, 2, 1, 0, 1, 1])
    np.testing.assert_array_equal(rule_predict(features.iloc[::-1]), [1, 1, 0, 1, 2, 2])


def test_complete_evaluation_writes_consistent_reports(demo_data, tmp_path):
    record = evaluate(demo_data[3], demo_data[4], tmp_path, synthetic=True, cv_folds=3)
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved == record
    assert saved["data_kind"] == "synthetic demonstration"
    assert saved["split"]["patient_overlap"] == 0
    assert len(saved["cross_validation"]["macro_f1_by_fold"]) == 3
    predictions = pd.read_csv(tmp_path / "predictions.csv")
    assert len(predictions) == saved["split"]["validation_breaths"] == 24
    assert not {"patient", "breath_id", "target"} & set(saved["features"])
    for name, report in saved["models"].items():
        matrix = np.asarray(report["confusion_matrix"])
        assert matrix.sum() == len(predictions)
        manual_accuracy = np.trace(matrix) / matrix.sum()
        assert report["accuracy"] == pytest.approx(manual_accuracy)
        assert report["dta_recall"] == pytest.approx(matrix[2, 2] / matrix[2].sum())
        assert 0 <= report["macro_f1"] <= 1
    assert set(saved["models"]) == {"Random Forest", "Logistic Regression", "KNN", "Feature Rules"}


def test_missing_class_in_patient_split_is_rejected(demo_data):
    data = demo_data[3].query("target != 2")
    with pytest.raises(ValueError, match="all three classes"):
        patient_split(data)


def test_invalid_cv_fold_count_is_rejected(demo_data, tmp_path):
    with pytest.raises(ValueError, match="cv_folds"):
        evaluate(demo_data[3], demo_data[4], tmp_path, cv_folds=100)
