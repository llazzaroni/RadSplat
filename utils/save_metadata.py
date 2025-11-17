import pandas as pd
import argparse
import json

def create_parser():
    parser = argparse.ArgumentParser(
        description="parser"
    )


    parser.add_argument(
        "--nerf-model",
        "-nm",
        type=str,
        required=True,
        help="nerf model used"
    )
    parser.add_argument(
        "--nerf-steps",
        "-ns",
        type=int,
        required=True,
        help="amount of steps to train nerf"
    )

    parser.add_argument(
        "--num-rays",
        "-nr",
        type=int,
        required=True,
        help="number of rays sampled"
    )

    parser.add_argument(
        "--sampling-strategy",
        "-s",
        type=str,
        required=True,
        help="sanpling stragegy used to dample rays from nerf"
    )

    parser.add_argument(
        "--experiment-name",
        "-e",
        type=str,
        required=True,
        help="experiment-name"
    )

    parser.add_argument(
        "--scene-name",
        "-n",
        type=str,
        required=True,
        help="scene-name"
    )

    return parser

if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()

    metadata = {
            "experiment-name" : args.experiment_name,
            "scene-name" : args.scene_name,
            "nerf-model" : args.nerf_model,
            "nerf-steps" : args.nerf_steps,
            "rays-sampled" : args.num_rays,
            "sampling-strategy" : args.sampling_strategy
        }


    with open("metadata.json", "w") as f:
        json.dump(metadata, f)


