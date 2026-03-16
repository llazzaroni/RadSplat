#!/usr/bin/env python3
"""Rank GSplat runs by the PSNR/SSIM/LPIPS combined metric."""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple


STEP_RE = re.compile(r"^(?P<stage>[a-zA-Z0-9_]+)_step(?P<step>\d+)\.json$")


def _load_from_stats_dir(run_dir: Path, stage: str, max_step: int = -1) -> List[Tuple[int, Dict]]:
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
        if max_step >= 0 and step > max_step:
            continue
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rows.append((step, payload))
    rows.sort(key=lambda x: x[0])
    return rows


def _load_from_aggregate(run_dir: Path, step_interval: int, max_step: int = -1) -> List[Tuple[int, Dict]]:
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
        if max_step >= 0 and step > max_step:
            continue
        rows.append((step, item))
    return rows


def _load_run(
    run_dir: Path, source: str, stage: str, step_interval: int, max_step: int = -1
) -> Tuple[List[Tuple[int, Dict]], str]:
    rows: List[Tuple[int, Dict]] = []
    used_source = source
    if source in ("auto", "stats_dir"):
        rows = _load_from_stats_dir(run_dir, stage=stage, max_step=max_step)
        if rows:
            used_source = "stats_dir"
    if not rows and source in ("auto", "aggregate"):
        rows = _load_from_aggregate(run_dir, step_interval=step_interval, max_step=max_step)
        if rows:
            used_source = "aggregate"
    return rows, used_source


def _combo(psnr: float, ssim: float, lpips: float) -> float:
    return (
        (10.0 ** (-float(psnr) / 10.0))
        * math.sqrt(max(0.0, 1.0 - float(ssim)))
        * float(lpips)
    ) ** (1.0 / 3.0)


def _extract_combo_series(rows: List[Tuple[int, Dict]]) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    for step, payload in rows:
        psnr = payload.get("psnr")
        ssim = payload.get("ssim")
        lpips = payload.get("lpips")
        if not isinstance(psnr, (int, float)):
            continue
        if not isinstance(ssim, (int, float)):
            continue
        if not isinstance(lpips, (int, float)):
            continue
        out.append((step, _combo(float(psnr), float(ssim), float(lpips))))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank runs by the minimum psnr/ssim/lpips combined score (lower is better)."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        action="append",
        type=Path,
        help="Run directory (repeat --run-dir for multiple runs).",
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
        help="Step interval used to infer steps from gsplat_stats.json.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=-1,
        help="Optional max step to include. -1 means all.",
    )
    parser.add_argument(
        "--mode",
        choices=["best", "mean", "final"],
        default="best",
        help="How to rank each run: best=min(combo), mean=avg(combo), final=last(combo).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional output JSON file with ranking details.",
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

    records = []
    for run_dir, label in zip(run_dirs, labels):
        rows, used_source = _load_run(
            run_dir,
            source=args.source,
            stage=args.stage,
            step_interval=args.step_interval,
            max_step=args.max_step,
        )
        if not rows:
            print(f"[WARN] {label}: no stats found in {run_dir}")
            continue
        combo_series = _extract_combo_series(rows)
        if not combo_series:
            print(f"[WARN] {label}: no valid psnr/ssim/lpips records")
            continue

        values = [v for _, v in combo_series]
        best_step, best_value = min(combo_series, key=lambda x: x[1])
        final_step, final_value = combo_series[-1]
        mean_value = sum(values) / float(len(values))
        if args.mode == "best":
            rank_value = best_value
        elif args.mode == "mean":
            rank_value = mean_value
        else:
            rank_value = final_value

        records.append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "source": used_source,
                "num_records": len(combo_series),
                "rank_value": rank_value,
                "best_combo": best_value,
                "best_step": best_step,
                "mean_combo": mean_value,
                "final_combo": final_value,
                "final_step": final_step,
            }
        )

    if not records:
        raise RuntimeError("No valid runs to rank.")

    records.sort(key=lambda r: r["rank_value"])
    print(f"Ranking mode: {args.mode} (lower is better)")
    for i, r in enumerate(records, start=1):
        print(
            f"{i:2d}. {r['label']}: rank={r['rank_value']:.6f} | "
            f"best={r['best_combo']:.6f}@{r['best_step']} | "
            f"mean={r['mean_combo']:.6f} | final={r['final_combo']:.6f}@{r['final_step']} | "
            f"n={r['num_records']} | source={r['source']}"
        )

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"Wrote ranking JSON: {args.out_json}")


if __name__ == "__main__":
    main()
