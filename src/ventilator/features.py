"""Convert a segmented, uniformly sampled breath into 16 numeric features."""

import numpy as np
import pandas as pd
from scipy.integrate import simpson

KEYS = ["patient", "breath_id"]
FEATURES = [
    "i_time", "e_time", "i_e_ratio", "rr", "tvi", "tve", "tve_tvi_ratio",
    "max_flow", "min_flow", "max_pressure", "min_pressure", "pip", "maw",
    "peep", "ip_auc", "ep_auc",
]
WAVEFORM_COLUMNS = KEYS + [
    "time_s", "flow_l_min", "pressure_cm_h2o", "inspiration",
]


def extract_breath(samples: pd.DataFrame) -> dict[str, float]:
    """Extract features; phase boundaries must be supplied by the data adapter.

    Integration uses Simpson's rule over each phase's sample timestamps.
    Timing uses sample counts times the sampling interval. MAW follows the
    original experiment's inspiratory-mean definition; see docs/data.md.
    """
    required = WAVEFORM_COLUMNS[2:]
    missing = set(required) - set(samples.columns)
    if missing:
        raise ValueError(f"Missing waveform columns: {sorted(missing)}")
    values = samples[required].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Waveform samples must be finite")
    times = values.time_s.to_numpy(dtype=float)
    if len(times) < 4:
        raise ValueError("A breath needs at least two samples in each phase")
    intervals = np.diff(times)
    dt = float(intervals[0])
    if dt <= 0 or not np.allclose(intervals, dt, rtol=1e-4, atol=1e-8):
        raise ValueError("Timestamps must increase at a uniform sampling interval")
    phase = values.inspiration.to_numpy()
    if not np.isin(phase, [0, 1]).all():
        raise ValueError("inspiration must contain only 0 and 1")
    n_in = int(phase.sum())
    n_out = len(phase) - n_in
    if n_in < 2 or n_out < 2 or not np.array_equal(
        phase, np.r_[np.ones(n_in), np.zeros(n_out)]
    ):
        raise ValueError("Each breath must contain inspiration followed by expiration")
    flow = values.flow_l_min.to_numpy(dtype=float)
    pressure = values.pressure_cm_h2o.to_numpy(dtype=float)
    inflow, outflow = flow[:n_in], flow[n_in:]
    if np.any(inflow < 0) or np.any(outflow > 0):
        raise ValueError("Flow must be nonnegative during inspiration and nonpositive during expiration")
    i_time, e_time = n_in * dt, n_out * dt
    tvi = float(simpson(inflow, dx=dt) * 1000 / 60)
    tve = float(-simpson(outflow, dx=dt) * 1000 / 60)
    if tvi <= 0:
        raise ValueError("Inhaled volume must be positive")
    return {
        "i_time": i_time,
        "e_time": e_time,
        "i_e_ratio": i_time / e_time,
        "rr": 60 / (i_time + e_time),
        "tvi": tvi,
        "tve": tve,
        "tve_tvi_ratio": tve / tvi,
        "max_flow": float(flow.max()),
        "min_flow": float(flow.min()),
        "max_pressure": float(pressure.max()),
        "min_pressure": float(pressure.min()),
        "pip": float(pressure[:n_in].max()),
        "maw": float(pressure[:n_in].mean()),
        "peep": float(np.median(pressure[n_in:][-5:])),
        "ip_auc": float(simpson(pressure[:n_in], dx=dt)),
        "ep_auc": float(simpson(pressure[n_in:], dx=dt)),
    }


def extract_table(waveforms: pd.DataFrame) -> pd.DataFrame:
    """One output row per (patient, breath_id); input rows are time ordered."""
    missing = set(WAVEFORM_COLUMNS) - set(waveforms.columns)
    if missing:
        raise ValueError(f"Missing waveform columns: {sorted(missing)}")
    if waveforms.empty or waveforms[KEYS].isna().any().any():
        raise ValueError("Waveforms need nonmissing patient and breath identifiers")
    rows = []
    for (patient, breath_id), samples in waveforms.groupby(KEYS, sort=True):
        rows.append({"patient": patient, "breath_id": breath_id, **extract_breath(samples)})
    return pd.DataFrame(rows, columns=KEYS + FEATURES)
