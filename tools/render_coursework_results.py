"""Regenerate the README figure using only retained aggregate metrics."""

import json
from pathlib import Path

from ventilator.plotting import plot_comparison

root = Path(__file__).resolve().parents[1]
record = json.loads((root / "results/coursework_metrics.json").read_text(encoding="utf-8"))
(root / "assets").mkdir(exist_ok=True)
plot_comparison(
    record["models"], root / "assets/coursework_results.png",
    "Ventilator breath classification | original coursework",
    "1,247 validation breaths | retained pipeline results, rounded to 2 decimals",
)
