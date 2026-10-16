"""Join labels by identity and keep a count of excluded rows."""

import numpy as np
import pandas as pd

from .features import FEATURES, KEYS

CLASS_NAMES = ["Non-PVA", "BSA", "DTA"]
CLASSES = [0, 1, 2]


def _check_keys(frame: pd.DataFrame, name: str) -> None:
    if frame[KEYS].isna().any().any():
        raise ValueError(f"{name} contains missing identifiers")
    if frame.patient.astype(str).str.strip().eq("").any():
        raise ValueError(f"{name} contains empty patient identifiers")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{name} must have one row per patient and breath_id")


def prepare_dataset(features: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Require complete one-to-one key coverage before filtering invalid rows."""
    for frame, required, name in [
        (features, KEYS + FEATURES, "features"),
        (labels, KEYS + ["bsa", "dta"], "labels"),
    ]:
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing {name} columns: {sorted(missing)}")
        _check_keys(frame, name)
    joined = features[KEYS + FEATURES].merge(
        labels[KEYS + ["bsa", "dta"]], on=KEYS, how="outer",
        validate="one_to_one", indicator=True,
    )
    if joined._merge.ne("both").any():
        raise ValueError("Feature and label keys do not match; align patient and breath_id first")
    flags = joined[["bsa", "dta"]].apply(pd.to_numeric, errors="raise")
    if not flags.isin([0, 1]).all().all():
        raise ValueError("bsa and dta must be binary 0/1 flags")
    numeric = joined[FEATURES].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    dual = flags.bsa.eq(1) & flags.dta.eq(1)
    keep = finite & ~dual
    audit = {
        "input_breaths": len(joined),
        "dual_label_rows": int(dual.sum()),
        "nonfinite_feature_rows": int((~finite).sum()),
        "excluded_rows": int((~keep).sum()),
        "retained_breaths": int(keep.sum()),
    }
    clean = joined.loc[keep, KEYS].copy()
    clean[FEATURES] = numeric.loc[keep]
    clean["target"] = (flags.bsa + 2 * flags.dta).loc[keep].astype(int)
    if clean.empty:
        raise ValueError("No valid breaths remain after cleaning")
    return clean.reset_index(drop=True), audit
