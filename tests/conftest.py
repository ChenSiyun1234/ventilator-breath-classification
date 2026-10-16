import pytest

from ventilator.data import prepare_dataset
from ventilator.features import extract_table
from ventilator.synthetic import make_demo


@pytest.fixture(scope="session")
def demo_data():
    waveforms, labels = make_demo(seed=7, patients=8, breaths_per_patient=12)
    features = extract_table(waveforms)
    data, audit = prepare_dataset(features, labels)
    return waveforms, labels, features, data, audit
