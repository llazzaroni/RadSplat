#!/usr/bin/env python3
"""Plot GSplat eval statistics over training steps."""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


STEP_RE = re.compile(r"^(?P<stage>[a-zA-Z0-9_]+)_step(?P<step>\d+)\.json$")


def _load_from_stats_dir(run_dir: Path, stage: str) -> List[Tuple[int, Dict]]:
    stats_dir = run_dir / "stats"
    if not stats_dir.is_dir():
        return []

    rows: List[Tuple[int, Dict]] = []
    for p in sorted(stats_dir.glob("*.json")):
        m = STEP_RE.match(p.name)
        if not m:
            continue
        if m.group("stage") != stage:
            continue
        step = int(m.group("step"))
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rows.append((step, payload))
    rows.sort(key=lambda x: x[0])
    return rows


def _load_from_aggregate(run_dir: Path, step_interval: int) -> List[Tuple[int, Dict]]:
    agg = run_dir / "gsplat_stats.json"
    if not agg.is_file():
        return []
    with agg.open("r", encoding="utf-8") as f:
        arr = json.load(f)
    if not isinstance(arr, list):
        return []

    rows: List[Tuple[int, Dict]] = []
    for i, item in enumerate(arr):
        if not isinstance(item, dict):
            continue
        step = i * step_interval
        rows.append((step, item))
    return rows


def _collect_metric_series(rows: List[Tuple[int, Dict]]) -> Dict[str, Tuple[List[int], List[float]]]:
    series: Dict[str, Tuple[List[int], List[float]]] = {}
    for step, payload in rows:
        for k, v in payload.items():
            if not isinstance(v, (int, float)):
                continue
            xs, ys = series.setdefault(k, ([], []))
            xs.append(step)
            ys.append(float(v))
    return series


def _load_run(
    run_dir: Path, source: str, stage: str, step_interval: int
) -> Tuple[List[Tuple[int, Dict]], str]:
    rows: List[Tuple[int, Dict]] = []
    used_source = source
    if source in ("auto", "stats_dir"):
        rows = _load_from_stats_dir(run_dir, stage=stage)
        if rows:
            used_source = "stats_dir"
    if not rows and source in ("auto", "aggregate"):
        rows = _load_from_aggregate(run_dir, step_interval=step_interval)
        if rows:
            used_source = "aggregate"
    return rows, used_source


def plot_metrics(rows: List[Tuple[int, Dict]], out_dir: Path, title_prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    series = _collect_metric_series(rows)
    if not series:
        raise RuntimeError("No numeric metrics found to plot.")

    preferred_order = [
        "psnr",
        "ssim",
        "lpips",
        "cc_psnr",
        "cc_ssim",
        "cc_lpips",
        "num_GS",
        "ellipse_time",
        "mem",
    ]
    metric_names = [m for m in preferred_order if m in series] + [
        m for m in sorted(series.keys()) if m not in preferred_order
    ]

    # Individual plots.
    for metric in metric_names:
        xs, ys = series[metric]
        plt.figure(figsize=(7, 4))
        plt.plot(xs, ys, marker="o", linewidth=1.5)
        plt.xlabel("Step")
        plt.ylabel(metric)
        plt.title(f"{title_prefix} - {metric}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}.png", dpi=180)
        plt.close()

    # Combined plot for common metrics if available.
    combo = [m for m in ["psnr", "ssim", "lpips"] if m in series]
    if combo:
        fig, axes = plt.subplots(1, len(combo), figsize=(6 * len(combo), 4))
        if len(combo) == 1:
            axes = [axes]
        for ax, metric in zip(axes, combo):
            xs, ys = series[metric]
            ax.plot(xs, ys, marker="o", linewidth=1.5)
            ax.set_title(metric)
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"{title_prefix} - Main Metrics")
        fig.tight_layout()
        fig.savefig(out_dir / "main_metrics.png", dpi=180)
        plt.close(fig)


def plot_comparison(
    all_series: Dict[str, Dict[str, Tuple[List[int], List[float]]]], out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = sorted(
        {metric for run_metrics in all_series.values() for metric in run_metrics.keys()}
    )
    if not metrics:
        raise RuntimeError("No numeric metrics found to compare.")

    preferred_order = [
        "psnr",
        "ssim",
        "lpips",
        "cc_psnr",
        "cc_ssim",
        "cc_lpips",
        "num_GS",
        "ellipse_time",
        "mem",
    ]
    metric_names = [m for m in preferred_order if m in metrics] + [
        m for m in metrics if m not in preferred_order
    ]

    for metric in metric_names:
        plt.figure(figsize=(8, 4.5))
        plotted = False
        for label, run_metrics in all_series.items():
            if metric not in run_metrics:
                continue
            xs, ys = run_metrics[metric]
            if len(xs) == 0:
                continue
            plt.plot(xs, ys, marker="o", linewidth=1.5, label=label)
            plotted = True
        if not plotted:
            plt.close()
            continue
        plt.xlabel("Step")
        plt.ylabel(metric)
        plt.title(f"GSplat Comparison - {metric}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"compare_{metric}.png", dpi=180)
        plt.close()

    combo = [m for m in ["psnr", "ssim", "lpips"] if m in metrics]
    if combo:
        fig, axes = plt.subplots(1, len(combo), figsize=(6 * len(combo), 4))
        if len(combo) == 1:
            axes = [axes]
        for ax, metric in zip(axes, combo):
            for label, run_metrics in all_series.items():
                if metric not in run_metrics:
                    continue
                xs, ys = run_metrics[metric]
                ax.plot(xs, ys, marker="o", linewidth=1.5, label=label)
            ax.set_title(metric)
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.3)
        axes[0].legend()
        fig.suptitle("GSplat Comparison - Main Metrics")
        fig.tight_layout()
        fig.savefig(out_dir / "compare_main_metrics.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GSplat evaluation stats over training steps.")
    parser.add_argument(
        "--run-dir",
        required=True,
        action="append",
        type=Path,
        help="GSplat result directory (repeat --run-dir for comparison).",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Optional label for each --run-dir (same order). Repeat as needed.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "stats_dir", "aggregate"],
        default="auto",
        help="Where to read stats from.",
    )
    parser.add_argument(
        "--stage",
        default="val",
        help="Stage prefix for stats_dir files (default: val).",
    )
    parser.add_argument(
        "--step-interval",
        type=int,
        default=100,
        help="Step interval to infer steps for gsplat_stats.json entries.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output folder for plots (default: single=<run-dir>/plots, multi=./gsplat_compare_plots).",
    )
    args = parser.parse_args()

    run_dirs = [p.resolve() for p in args.run_dir]
    labels = args.label if args.label is not None else []
    if labels and len(labels) != len(run_dirs):
        raise ValueError(
            f"Number of --label ({len(labels)}) must match number of --run-dir ({len(run_dirs)})."
        )
    if not labels:
        labels = [p.name for p in run_dirs]

    if len(run_dirs) == 1:
        run_dir = run_dirs[0]
        out_dir = args.out_dir.resolve() if args.out_dir is not None else (run_dir / "plots")
        rows, used_source = _load_run(
            run_dir, source=args.source, stage=args.stage, step_interval=args.step_interval
        )
        if not rows:
            stats_glob = run_dir / "stats"
            agg_file = run_dir / "gsplat_stats.json"
            raise RuntimeError(
                "No stats found. Expected either "
                f"'{stats_glob}/*_stepXXXX.json' or '{agg_file}'."
            )
        plot_metrics(rows, out_dir, title_prefix=f"GSplat ({used_source})")
        print(f"Plotted {len(rows)} points from {used_source} into: {out_dir}")
        return

    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else (Path.cwd() / "gsplat_compare_plots")
    )
    all_series: Dict[str, Dict[str, Tuple[List[int], List[float]]]] = {}
    for run_dir, label in zip(run_dirs, labels):
        rows, used_source = _load_run(
            run_dir, source=args.source, stage=args.stage, step_interval=args.step_interval
        )
        if not rows:
            print(f"[WARN] skipping {label}: no stats found in {run_dir}")
            continue
        all_series[label] = _collect_metric_series(rows)
        print(f"[OK] {label}: {len(rows)} points from {used_source}")

    if not all_series:
        raise RuntimeError("No valid runs to compare.")

    plot_comparison(all_series, out_dir=out_dir)
    print(f"Comparison plots written to: {out_dir}")


if __name__ == "__main__":
    main()
