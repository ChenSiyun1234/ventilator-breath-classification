import numpy as np
import pandas as pd
import pytest

from ventilator.data import prepare_dataset


def test_shuffled_labels_keep_the_same_targets(demo_data):
    _, labels, features, expected, _ = demo_data
    actual, _ = prepare_dataset(features, labels.sample(frac=1, random_state=9))
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("side", ["features", "labels"])
def test_duplicate_identity_is_rejected(demo_data, side):
    _, labels, features, _, _ = demo_data
    features, labels = features.copy(), labels.copy()
    if side == "features":
        features = pd.concat([features, features.iloc[:1]])
    else:
        labels = pd.concat([labels, labels.iloc[:1]])
    with pytest.raises(ValueError, match="one row"):
        prepare_dataset(features, labels)


def test_missing_label_is_not_silently_dropped(demo_data):
    _, labels, features, _, _ = demo_data
    with pytest.raises(ValueError, match="keys do not match"):
        prepare_dataset(features, labels.iloc[1:])


def test_invalid_rows_share_the_filter_and_are_counted(demo_data):
    _, labels, features, expected, _ = demo_data
    features, labels = features.copy(), labels.copy()
    labels.loc[0, ["bsa", "dta"]] = 1
    features.loc[1, "tvi"] = np.nan
    features.loc[2, "rr"] = np.inf
    clean, audit = prepare_dataset(features, labels.sample(frac=1, random_state=4))
    assert audit == {
        "input_breaths": 96, "dual_label_rows": 1, "nonfinite_feature_rows": 2,
        "excluded_rows": 3, "retained_breaths": 93,
    }
    pd.testing.assert_frame_equal(clean, expected.iloc[3:].reset_index(drop=True))


def test_invalid_label_flags_are_rejected(demo_data):
    _, labels, features, _, _ = demo_data
    labels = labels.copy()
    labels.loc[0, "dta"] = 2
    with pytest.raises(ValueError, match="binary"):
        prepare_dataset(features, labels)
