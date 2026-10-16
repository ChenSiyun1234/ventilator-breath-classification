"""Deterministic toy waveforms for exercising the software without patient data."""

import numpy as np
import pandas as pd
from scipy.integrate import simpson


def make_demo(seed: int = 42, patients: int = 24, breaths_per_patient: int = 40):
    """Generate labelled mathematical patterns, not physiological simulations.

    Classes deliberately differ in volume ratio and expiration duration.
    Classifier scores on these patterns measure demonstration behavior only.
    """
    if patients < 4 or breaths_per_patient < 10:
        raise ValueError("Use at least four synthetic patients and ten breaths each")
    rng = np.random.default_rng(seed)
    waves, labels = [], []
    dt = 0.02
    for patient_id in range(patients):
        patient = f"synthetic-{patient_id + 1:03d}"
        pressure_offset = rng.uniform(4, 8)
        targets = np.array([0, 1, 2] + rng.choice([0, 1, 2], breaths_per_patient - 3, p=[0.70, 0.22, 0.08]).tolist())
        rng.shuffle(targets)
        for breath_id, target in enumerate(targets, 1):
            n_in = int(rng.integers(25, 66))
            n_out = int(rng.integers(5, 15) if target == 2 else rng.integers(45, 111))
            ratio = rng.uniform(*[(0.86, 1.12), (0.45, 0.96), (0.12, 0.46)][target])
            amplitude = rng.uniform(16, 40)
            inspiratory = amplitude * (0.12 + np.sin(np.linspace(0, np.pi, n_in)))
            inspiratory *= rng.uniform(0.93, 1.07, n_in)
            expiratory_shape = 0.12 + np.exp(-np.linspace(0, 2.8, n_out))
            expiratory = -expiratory_shape * simpson(inspiratory, dx=dt) * ratio / simpson(expiratory_shape, dx=dt)
            flow = np.r_[inspiratory, expiratory]
            pressure = pressure_offset + np.r_[
                rng.uniform(9, 20) * np.sin(np.linspace(0, np.pi, n_in)),
                2 * np.exp(-np.linspace(0, 4, n_out)),
            ] + rng.normal(0, 0.18, len(flow))
            waves.append(pd.DataFrame({
                "patient": patient, "breath_id": breath_id,
                "time_s": np.arange(len(flow)) * dt,
                "flow_l_min": flow, "pressure_cm_h2o": pressure,
                "inspiration": np.r_[np.ones(n_in, dtype=int), np.zeros(n_out, dtype=int)],
            }))
            labels.append({"patient": patient, "breath_id": breath_id, "bsa": int(target == 1), "dta": int(target == 2)})
    return pd.concat(waves, ignore_index=True), pd.DataFrame(labels)
