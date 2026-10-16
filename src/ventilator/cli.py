"""Command line entry points; all generated data stays in the output folder."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import prepare_dataset
from .evaluation import evaluate
from .features import extract_table
from .plotting import write_demo_figures
from .synthetic import make_demo


def main():
    parser = argparse.ArgumentParser(description="Ventilator breath classification")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run the complete pipeline on synthetic waveforms")
    demo.add_argument("--patients", type=int, default=24)
    demo.add_argument("--breaths-per-patient", type=int, default=40)
    extract = commands.add_parser("extract", help="Convert canonical waveform CSV into feature CSV")
    extract.add_argument("--waveforms", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("evaluate", help="Evaluate a feature CSV with a keyed label CSV")
    run.add_argument("--features", type=Path, required=True)
    run.add_argument("--labels", type=Path, required=True)
    for command in [demo, run]:
        command.add_argument("--out", type=Path, default=Path("artifacts/demo" if command is demo else "artifacts/evaluation"))
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--test-size", type=float, default=0.25)
        command.add_argument("--cv-folds", type=int, default=5, help="Patient-grouped training CV; 0 disables it")
    args = parser.parse_args()
    try:
        if args.command == "extract":
            features = extract_table(pd.read_csv(args.waveforms, dtype={"patient": str}))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            features.to_csv(args.output, index=False)
            print(f"Wrote {len(features)} breaths to {args.output}")
            return
        if args.command == "demo":
            waveforms, labels = make_demo(args.seed, args.patients, args.breaths_per_patient)
            features = extract_table(waveforms)
        else:
            features = pd.read_csv(args.features, dtype={"patient": str})
            labels = pd.read_csv(args.labels, dtype={"patient": str})
        data, audit = prepare_dataset(features, labels)
        record = evaluate(data, audit, args.out, args.seed, args.test_size,
                          synthetic=args.command == "demo", cv_folds=args.cv_folds)
        if args.command == "demo":
            waveforms.to_csv(args.out / "waveforms.csv", index=False)
            labels.to_csv(args.out / "labels.csv", index=False)
            features.to_csv(args.out / "features.csv", index=False)
            write_demo_figures(record, waveforms, labels, args.out)
        print(record["data_kind"])
        print(json.dumps(record["split"], indent=2))
        for name, metrics in record["models"].items():
            print(f"{name:20s} accuracy={metrics['accuracy']:.3f}  macro F1={metrics['macro_f1']:.3f}  DTA recall={metrics['dta_recall']:.3f}")
        print(f"Results: {args.out.resolve()}")
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
