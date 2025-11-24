import matplotlib.pyplot as plt
import json
import pandas as pd
from utils import create_sfm_df, create_complete_df
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path

def rescaled_x(n, end=5000):
    if n <= 1:
        return np.array([0.0])
    return np.arange(n, dtype=float) * (end / (n - 1))


def main(args):
    exp_df = create_complete_df("experiments_results")
    sfm_df = create_sfm_df("experiments_results")

    nerf_steps = exp_df["nerf-steps"].unique()
    rays_sampled = exp_df["rays-sampled"].unique()
    sampling_strategy = exp_df["sampling-strategy"].unique()
    percentage_random = exp_df["percentage-random"].unique()

    plt.figure(figsize=(8, 4.5))

    iterations = [int(c) for c in args.iterations]
    points = [int(c) for c in args.points]

    length_plot = int(args.len_plot)

    for nerf_steps_value in nerf_steps:
        if nerf_steps_value not in set(iterations):
            continue
        for rays_sampled_value in rays_sampled:
            if rays_sampled_value not in set(points):
                continue
            for sampling_strategy_value in sampling_strategy:
                if sampling_strategy_value == "random"  or sampling_strategy_value == "canny" or sampling_strategy_value == "sobel":
                    if sampling_strategy_value == "random" and args.random == False:
                        continue
                    if sampling_strategy_value == "canny" and args.canny == False:
                        continue
                    if sampling_strategy_value == "sobel" and args.sobel == False:
                        continue
                    df_loop = exp_df[(exp_df["nerf-steps"] == nerf_steps_value) & (exp_df["rays-sampled"] == rays_sampled_value) & (exp_df["sampling-strategy"] == sampling_strategy_value)]

                    df_loop = df_loop[df_loop["scene-name"].isin(set(args.scenes))]

                    df_loop = df_loop[df_loop["step"] < length_plot]

                    metric_cols = ["ssim", "psnr", "lpips"]

                    agg_scenes = (
                        df_loop
                        .groupby(["scene-name", "step"])[metric_cols]
                        .mean()
                    )

                    print(len(df_loop), len(agg_scenes), nerf_steps_value)
                
                    agg = agg_scenes.groupby("step")[metric_cols].mean()
                    
                    y = agg[args.metric].to_numpy()
                    if len(y) == 0:
                        continue
                    x = rescaled_x(len(y), len(y) * 100)
                    
                    plt.plot(x, y, marker="o", label="nerf steps: " + str(nerf_steps_value) + "; number of points: " + str(rays_sampled_value) + "; strategy: " + str(sampling_strategy_value))
                    #plt.plot(x, y, label="nerf steps: " + str(nerf_steps_value) + "; number of points: " + str(rays_sampled_value) + "; strategy: " + str(sampling_strategy_value))
                else:
                    if sampling_strategy_value == "mixed-canny" and args.canny_mixed == False:
                        continue
                    if sampling_strategy_value == "mixed-sobel" and args.sobel_mixed == False:
                        continue
                    for percentage_random_value in percentage_random:
                        df_loop = exp_df[(exp_df["nerf-steps"] == nerf_steps_value) & (exp_df["rays-sampled"] == rays_sampled_value) & (exp_df["sampling-strategy"] == sampling_strategy_value) & (exp_df["percentage-random"] == percentage_random_value)]
                    
                        df_loop = df_loop[df_loop["scene-name"].isin(set(args.scenes))]

                        df_loop = df_loop[df_loop["step"] <= length_plot]

                        metric_cols = ["ssim", "psnr", "lpips"]
                    
                        agg_scenes = (
                            df_loop
                            .groupby(["scene-name", "step"])[metric_cols]
                            .mean()
                        )

                        print(len(df_loop), len(agg_scenes), nerf_steps_value)
                    
                        agg = agg_scenes.groupby("step")[metric_cols].mean()
                        
                        y = agg[args.metric].to_numpy()
                        if len(y) == 0:
                            continue

                        x = rescaled_x(len(y), len(y) * 100)
                        plt.plot(x, y, marker="o", label="nerf steps: " + str(nerf_steps_value) + "; points: " + str(rays_sampled_value) + "; strategy: " + str(sampling_strategy_value) + "; percentage: " + str(percentage_random_value))
                        #plt.plot(x, y, label=str(nerf_steps_value) + " " + str(rays_sampled_value) + " " + str(sampling_strategy_value) + " " + str(percentage_random[0]))

                    
    metric_cols = ["ssim", "psnr", "lpips"]
    sfm_df = sfm_df[sfm_df["scene-name"].isin(set(args.scenes))]
    sfm_df = sfm_df[sfm_df["step"] < length_plot]
    agg_scenes = (
        sfm_df
        .groupby(["scene-name", "step"])[metric_cols]
        .mean()
    )
    print(len(sfm_df), len(agg_scenes))
    agg = agg_scenes.groupby("step")[metric_cols].mean()
    
    y = agg[args.metric].to_numpy()
    x = rescaled_x(len(y), len(y) * 100)
    plt.plot(x, y, marker="o", label="SFM")
    #plt.plot(x, y, label="SFM")


    plt.xlim(0, int(args.len_plot))
    plt.title(args.metric + " over records")
    plt.xlabel("Record index")
    plt.ylabel(args.metric)
    plt.grid(True, alpha=0.3)
    plt.legend(
        title="Source",
        fontsize="small",
        title_fontsize="small",
        markerscale=0.7,
        handlelength=1.0,
        handletextpad=0.3,
        borderpad=0.2,
        labelspacing=0.2
    )
    plt.tight_layout()
    plt.savefig(Path("/home/llazzaroni/ds-lab/RadSplat/plot_metrics/plots/img1"), dpi=150)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build cycling winners dataset.")
    p.add_argument("--metric", default="ssim")
    p.add_argument("--random", default=True)
    p.add_argument("--sobel", default=False)
    p.add_argument("--canny", default=False)
    p.add_argument("--sobel-mixed", default=False)
    p.add_argument("--canny-mixed", default=False)
    p.add_argument(
        "--scenes",
        nargs="+",
        default=["bicycle", "bonsai", "counter", "flowers", "garden",
                "kitchen", "room", "stump", "treehill"],
        help="List of scene names to process"
    )
    p.add_argument(
        "--points",
        nargs="+",
        default=[500000, 1000000, 1500000],
        help="List of scene names to process"
    )
    p.add_argument(
        "--iterations",
        nargs="+",
        default=[500, 1000, 2500, 5000, 10000],
        help="List of scene names to process"
    )
    p.add_argument("--len-plot", default=5000)
    args = p.parse_args()
    main(args)