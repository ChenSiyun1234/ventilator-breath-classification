"""Headless figures for saved experiments and the synthetic demo."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import CLASS_NAMES

COLORS = ["#2563eb", "#0891b2", "#7c3aed", "#d97706"]


def plot_comparison(models, destination, title, subtitle):
    names = list(models)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    fig.suptitle(title + "\n" + subtitle, fontsize=14, fontweight="bold")
    for ax, metric, label in zip(axes, ["accuracy", "macro_f1", "dta_recall"], ["Accuracy", "Macro F1", "DTA recall"]):
        scores = [models[n][metric] for n in names]
        bars = ax.barh(np.arange(len(names)), scores, color=COLORS[:len(names)], height=0.58)
        ax.set_yticks(np.arange(len(names)), names)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.5, 1])
        ax.set_title(label, fontsize=12)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=0.16)
        ax.set_axisbelow(True)
        for bar, score in zip(bars, scores):
            ax.text(score + 0.02, bar.get_y() + bar.get_height()/2, f"{score:.2f}", va="center", fontsize=10)
    fig.savefig(destination, dpi=160, facecolor="white")
    plt.close(fig)


def plot_confusions(record, destination):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), constrained_layout=True)
    fig.suptitle("Synthetic demonstration | held-out synthetic patients", fontsize=14, fontweight="bold")
    for ax, (name, results) in zip(axes, record["models"].items()):
        matrix = np.asarray(results["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        for row in range(3):
            for col in range(3):
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="white" if matrix[row, col] > matrix.max()/2 else "#172554")
        ax.set_xticks(range(3), CLASS_NAMES, rotation=25)
        ax.set_yticks(range(3), CLASS_NAMES)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
    axes[0].set_ylabel("True class")
    fig.savefig(destination, dpi=160, facecolor="white")
    plt.close(fig)


def plot_waveforms(waveforms, labels, destination):
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.2), constrained_layout=True)
    fig.suptitle("Synthetic waveform examples | mathematical demo patterns", fontsize=14, fontweight="bold")
    for col, name in enumerate(CLASS_NAMES):
        label = labels[(labels.bsa + 2 * labels.dta) == col].iloc[0]
        samples = waveforms[(waveforms.patient == label.patient) & (waveforms.breath_id == label.breath_id)]
        for row, field, unit in [(0, "flow_l_min", "Flow (L/min)"), (1, "pressure_cm_h2o", "Pressure (cm H2O)")]:
            ax = axes[row, col]
            ax.plot(samples.time_s, samples[field], color=COLORS[col], linewidth=2)
            boundary = samples.loc[samples.inspiration.eq(0), "time_s"].iloc[0]
            ax.axvline(boundary, color="#94a3b8", linestyle="--", linewidth=1)
            ax.grid(alpha=0.15)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel(unit)
            if row == 0:
                ax.set_title(name)
            else:
                ax.set_xlabel("Time (s)")
    fig.savefig(destination, dpi=160, facecolor="white")
    plt.close(fig)


def write_demo_figures(record, waveforms, labels, out):
    out = Path(out)
    plot_waveforms(waveforms, labels, out / "waveforms.png")
    plot_comparison(record["models"], out / "model_comparison.png",
                    "Synthetic demonstration", "Generated patterns; scores describe this demo only")
    plot_confusions(record, out / "confusion_matrices.png")
