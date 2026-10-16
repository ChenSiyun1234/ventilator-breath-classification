import numpy as np
import pandas as pd
import pytest

from ventilator.features import FEATURES, extract_breath, extract_table


def constant_breath():
    return pd.DataFrame({
        "time_s": np.arange(6) * 0.02,
        "flow_l_min": [30, 30, 30, -20, -20, -20],
        "pressure_cm_h2o": [10.0, 12.0, 14.0, 7.0, 6.0, 5.0],
        "inspiration": [1, 1, 1, 0, 0, 0],
    })


def test_feature_units_against_analytic_constant_signal():
    features = extract_breath(constant_breath())
    assert set(features) == set(FEATURES)
    assert features["tvi"] == pytest.approx(20)  # 30 L/min * .04 s * 1000/60
    assert features["tve"] == pytest.approx(40 / 3)
    assert features["tve_tvi_ratio"] == pytest.approx(2 / 3)
    assert features["i_time"] == pytest.approx(0.06)
    assert features["e_time"] == pytest.approx(0.06)
    assert features["rr"] == pytest.approx(500)
    assert features["pip"] == 14
    assert features["maw"] == 12
    assert features["peep"] == 6
    assert features["ip_auc"] == pytest.approx(0.48)
    assert features["ep_auc"] == pytest.approx(0.24)


@pytest.mark.parametrize("kind", ["irregular", "reversed", "nonfinite", "phase", "flow_sign", "zero_volume"])
def test_rejects_invalid_waveform_contract(kind):
    samples = constant_breath()
    if kind == "irregular":
        samples.loc[2, "time_s"] += 0.005
    elif kind == "reversed":
        samples = samples.iloc[::-1]
    elif kind == "nonfinite":
        samples.loc[0, "pressure_cm_h2o"] = np.inf
    elif kind == "phase":
        samples["inspiration"] = [1, 0, 1, 0, 0, 0]
    elif kind == "flow_sign":
        samples.loc[3, "flow_l_min"] = 20
    else:
        samples.loc[:2, "flow_l_min"] = 0
    with pytest.raises(ValueError):
        extract_breath(samples)


def test_breath_ids_are_scoped_to_patient():
    first = constant_breath().assign(patient="a", breath_id=1)
    second = constant_breath().assign(patient="b", breath_id=1)
    second["flow_l_min"] *= 2
    features = extract_table(pd.concat([first, second]))
    assert len(features) == 2
    assert features.tvi.tolist() == pytest.approx([20, 40])
