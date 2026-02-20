# RadSplat — reproduction + fast-training extensions

This repository contains code to reproduce the core pipeline of **RadSplat: Radiance Field-Informed Gaussian Splatting**
and additional experiments focused on **fast-training regimes / low-iteration performance**.

> Paper: https://arxiv.org/abs/2403.13806  
> Project page (authors): https://m-niemeyer.github.io/radsplat/

## What’s in this repo (high level)
RadSplat bridges **NeRF supervision / priors** with **3D Gaussian Splatting (3DGS)** to obtain robust optimization and high-quality real-time rendering.

In this student project, we focused on:
- **Fast-training initialization** for 3DGS (improves early convergence)
- **Quality–speed tradeoffs** under limited optimization steps
- Evaluation on standard metrics (**PSNR / SSIM / LPIPS**) across scenes

## Quickstart

The code in this repository was developed and tested on the **ETH Zurich student compute cluster**

The full experimental pipeline (data preparation → training → evaluation → plotting) is orchestrated via:

```bash
bash submission_scripts/full_pipeline.sh