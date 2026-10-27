"""Figures and a markdown table from evaluation JSONs and training logs.

Examples:
    uv run python scripts/plot_results.py
    uv run python scripts/plot_results.py --run "Persistence=results/persistence_val.json" \
        --run "U-Net (MSE, last)=results/unet_mse_last_val.json"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_RUNS = [
    ("Persistence", "results/persistence_val.json"),
    ("Advection", "results/advection_val.json"),
    ("U-Net (MSE)", "results/unet_mse_best_val.json"),
    ("U-Net (weighted MSE)", "results/unet_weighted_mse_best_val.json"),
]
DEFAULT_TRAINING = [
    ("U-Net (MSE)", "results/training/unet_mse_metrics.csv"),
    ("U-Net (weighted MSE)", "results/training/unet_weighted_mse_metrics.csv"),
]


def parse_specs(specs: list[str] | None, default: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if not specs:
        return list(default)
    out = []
    for s in specs:
        label, _, path = s.partition("=")
        if not path:
            raise SystemExit(f"expected 'Label=path', got {s!r}")
        out.append((label, path))
    return out


def load_runs(specs: list[tuple[str, str]]) -> dict[str, dict]:
    runs = {}
    for label, path in specs:
        p = Path(path)
        if not p.exists():
            print(f"warning: {p} not found, skipping '{label}'")
            continue
        runs[label] = json.loads(p.read_text())
    return runs


def plot_csi_by_lead_time(runs: dict[str, dict], out: Path) -> None:
    first = next(iter(runs.values()))
    thresholds = first["thresholds"]
    minutes = 5 * (np.arange(len(first["mse"])) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for k, (ax, thr) in enumerate(zip(axes.ravel(), thresholds, strict=False)):
        for label, res in runs.items():
            ax.plot(minutes, np.array(res["csi"], dtype=float)[:, k], marker="o", ms=3, label=label)
        ax.set_title(f"Threshold {thr}")
        ax.set_xlabel("Lead time (min)")
        ax.set_ylabel("CSI")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_csi_by_threshold(runs: dict[str, dict], out: Path) -> None:
    thresholds = next(iter(runs.values()))["thresholds"]
    x = np.arange(len(thresholds))
    width = 0.8 / len(runs)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (label, res) in enumerate(runs.items()):
        ax.bar(x + (i - (len(runs) - 1) / 2) * width, np.array(res["csi_pooled"], dtype=float),
               width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds)
    ax.set_xlabel("VIL threshold")
    ax.set_ylabel("CSI (pooled over lead times)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def markdown_table(runs: dict[str, dict]) -> str:
    thresholds = next(iter(runs.values()))["thresholds"]
    labels = list(runs)
    csi = np.array([np.array(runs[lb]["csi_pooled"], dtype=float) for lb in labels])
    csi_m = np.nanmean(csi, axis=1)
    mse = np.array([float(np.mean(runs[lb]["mse"])) for lb in labels])
    values = np.column_stack([csi, csi_m, mse])  # (models, thresholds + CSI-M + MSE)
    best = np.concatenate([np.nanargmax(values[:, :-1], axis=0), [np.argmin(mse)]])

    header = "| Model | " + " | ".join(f"CSI@{t}" for t in thresholds) + " | CSI-M | MSE |"
    lines = [header, "|" + "---|" * (values.shape[1] + 1)]
    for i, label in enumerate(labels):
        cells = []
        for j, v in enumerate(values[i]):
            s = f"{v:.4f}" if j == values.shape[1] - 1 else f"{v:.3f}"
            cells.append(f"**{s}**" if best[j] == i else s)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_training(specs: list[tuple[str, str]], out: Path) -> bool:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    found = False
    for label, path in specs:
        if not Path(path).exists():
            print(f"warning: {path} not found, skipping '{label}'")
            continue
        df = pd.read_csv(path).dropna(subset=["val/csi_m"])
        axes[0].plot(df["epoch"], df["val/csi_m"], label=label)
        axes[1].plot(df["epoch"], df["val/mse"], label=label)
        found = True
    if not found:
        plt.close(fig)
        return False
    axes[0].set(xlabel="Epoch", ylabel="val CSI-M")
    axes[1].set(xlabel="Epoch", ylabel="val MSE")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", help="'Label=path/to/eval.json' (repeatable)")
    p.add_argument("--training", action="append", help="'Label=path/to/metrics.csv' (repeatable)")
    p.add_argument("--out-dir", type=Path, default=Path("assets"))
    p.add_argument("--table", type=Path, default=Path("results/results_table.md"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(parse_specs(args.run, DEFAULT_RUNS))
    if not runs:
        raise SystemExit("no evaluation JSON found")

    plot_csi_by_lead_time(runs, args.out_dir / "csi_by_lead_time.png")
    plot_csi_by_threshold(runs, args.out_dir / "csi_by_threshold.png")
    table = markdown_table(runs)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(table + "\n")
    if plot_training(parse_specs(args.training, DEFAULT_TRAINING), args.out_dir / "training_curves.png"):
        print("saved training_curves.png")
    print(f"saved figures in {args.out_dir}/ and table in {args.table}\n")
    print(table)


if __name__ == "__main__":
    main()
