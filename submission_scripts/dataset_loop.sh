#!/usr/bin/env bash
set -u
set -o pipefail

BASE_DIR_DATA="/cluster/scratch/rbollati/dataset"   # where *_sparse datasets are
BASE_DIR_MODELS="/cluster/scratch/rbollati"          # where <scene>_models will be
TMP_ROOT="/cluster/home/rbollati/tmp"

{
for d in "$BASE_DIR_DATA"/*_sparse; do
  [ -d "$d" ] || continue
  scene="$(basename "$d")"
  [ "$scene" = "flowers_sparse" ] && continue

  prefix="${scene%%_sparse}"                  # e.g. kitchen_sparse -> kitchen
  models_dir="$BASE_DIR_MODELS/${prefix}_models"
  aug_dataset="$BASE_DIR_DATA/${prefix}_sparse_aug"

  mkdir -p "$models_dir"
  echo "===== Scene: $scene ====="

  # 1) Train 5 NeRF ensembles
  for i in 1 2 3 4 5; do
    run_dir="$models_dir/nerf_ensemble_${i}"
    mkdir -p "$run_dir"
    cd "$run_dir" || { echo "[WARN] cannot cd to $run_dir"; continue; }

    echo "[INFO] Training nerf_ensemble_${i} in $run_dir"
    if ! ns-train nerfacto \
      --vis tensorboard \
      --experiment-name ensemble_1 \
      --timestamp nerf \
      --steps-per-eval-image 30000 \
      --max-num-iterations 30000 \
      --save-only-latest-checkpoint True \
      --logging.steps-per-log 100 \
      --logging.profiler pytorch \
      colmap \
      --downscale-factor 1 \
      --colmap-path "$BASE_DIR_DATA/$scene/sparse/0" \
      --images-path "$BASE_DIR_DATA/$scene/images"
    then
      echo "[WARN] ns-train failed for $scene ensemble $i, continuing."
    fi
  done

  # 2) Run nerf_step (do not crash whole script on failure)
  nerf_cfg="$models_dir/nerf_ensemble_1/outputs/ensemble_1/nerfacto/nerf/config.yml"
  ray_out="$models_dir/nerf_ensemble_1/outputs/ensemble_1/nerfacto/nerf/ray_sample.pt"

  if ! python ~/RadSplat/RadSplat/nerf_step.py \
    --nerf-folder "$nerf_cfg" \
    --output-name "$ray_out" \
    --sampling-size 1000000 \
    --ray-sampling-strategy random
  then
    echo "[WARN] nerf_step failed for $scene, continuing."
  fi

  # 3) Run images.py with the 5 ensemble configs
  nerf_folders=()
  exp_dirs=()
  for i in 1 2 3 4 5; do
    nerf_folders+=("$models_dir/nerf_ensemble_${i}/outputs/ensemble_1/nerfacto/nerf/config.yml")
    exp_dirs+=("$models_dir/nerf_ensemble_${i}")
  done

  if ! python ~/RadSplat/RadSplat/images.py \
    --nerf-folders "${nerf_folders[@]}" \
    --exp-dirs "${exp_dirs[@]}" \
    --input-dataset "$BASE_DIR_DATA/$scene" \
    --output-dataset "$aug_dataset" \
    --tmp-root "$TMP_ROOT" \
    --tau 1.5 \
    --debug-plot-dir image_supervision \
    --num-final-samples 200 \
    --final-render-scale 0.125
  then
    echo "[WARN] images.py failed for $scene, continuing."
  fi
done

echo "All scenes processed."
} 2>&1 | tee -a ~/overnight_pipeline.log




python run_gsplat_dual.py default \
  --no-nerf-init \
  --pt-path /cluster/scratch/rbollati/bicycle_models/nerf_ensemble_1/outputs/ensemble_1/nerfacto/nerf/ray_sample.pt \
  --data-dir /cluster/scratch/rbollati/dataset/bicycle_sparse_aug \
  --result-dir /cluster/scratch/rbollati/bicycle_models/RadSplat_dual_full_nores_l2_1 \
  --data-factor 1 \
  --nerf-samples-data-factor 8 \
  --batch-size 1 \
  --nerf-batch-factor 20 \
  --dual-nerf-decay-steps-to-quarter 300 \
  --dual-nerf-disable-threshold 0.1 \
  --max-steps 20000 \
  --strategy.reset-every 100000000 \
  --dual-nerf-loss-weight 2 \
  --deterministic \
  --disable-viewer


python run_gsplat_dual.py default \
  --no-nerf-init \
  --pt-path /cluster/scratch/rbollati/bicycle_models/nerf_ensemble_1/outputs/ensemble_1/nerfacto/nerf/ray_sample.pt \
  --data-dir /cluster/scratch/rbollati/dataset/bicycle_sparse_aug \
  --result-dir /cluster/scratch/rbollati/bicycle_models/gsplat_sfm_l2_nores_2 \
  --data-factor 1 \
  --nerf-samples-data-factor 8 \
  --batch-size 1 \
  --nerf-batch-factor 20 \
  --dual-nerf-decay-steps-to-quarter 300 \
  --dual-nerf-disable-threshold 0.1 \
  --max-steps 20000 \
  --strategy.reset-every 100000000 \
  --dual-nerf-loss-weight 0 \
  --deterministic \
  --disable-viewer

conda deactivate
conda activate nerfstudio


python ~/RadSplat/RadSplat/images.py \
  --nerf-folders /cluster/scratch/rbollati/bonsai_models/nerf_ensemble_1/outputs/ensemble_1/nerfacto/nerf/config.yml /cluster/scratch/rbollati/bonsai_models/nerf_ensemble_2/outputs/ensemble_1/nerfacto/nerf/config.yml /cluster/scratch/rbollati/bonsai_models/nerf_ensemble_3/outputs/ensemble_1/nerfacto/nerf/config.yml /cluster/scratch/rbollati/bonsai_models/nerf_ensemble_4/outputs/ensemble_1/nerfacto/nerf/config.yml /cluster/scratch/rbollati/bonsai_models/nerf_ensemble_5/outputs/ensemble_1/nerfacto/nerf/config.yml \
  --exp-dirs /cluster/scratch/rbollati/bonsai_models/nerf_ensemble_1 /cluster/scratch/rbollati/bonsai_models/nerf_ensemble_2 /cluster/scratch/rbollati/bonsai_models/nerf_ensemble_3 /cluster/scratch/rbollati/bonsai_models/nerf_ensemble_4 /cluster/scratch/rbollati/bonsai_models/nerf_ensemble_5 \
  --input-dataset /cluster/scratch/rbollati/dataset/bonsai_sparse \
  --output-dataset /cluster/scratch/rbollati/dataset/bonsai_sparse_aug \
  --tmp-root /cluster/home/rbollati/tmp \
  --tau 1.5 \
  --debug-plot-dir image_supervision \
  --num-final-samples 200 \
  --final-render-scale 0.125

conda deactivate
conda activate gsplat

python run_gsplat_dual.py default \
  --no-nerf-init \
  --pt-path /cluster/scratch/rbollati/bonsai_models/nerf_ensemble_1/outputs/ensemble_1/nerfacto/nerf/ray_sample.pt \
  --data-dir /cluster/scratch/rbollati/dataset/bonsai_sparse_aug \
  --result-dir /cluster/scratch/rbollati/bonsai_models/RadSplat_dual_full_nores_l2_1 \
  --data-factor 1 \
  --nerf-samples-data-factor 8 \
  --batch-size 1 \
  --nerf-batch-factor 20 \
  --dual-nerf-decay-steps-to-quarter 500 \
  --dual-nerf-disable-threshold 0.1 \
  --max-steps 20000 \
  --strategy.reset-every 100000000 \
  --dual-nerf-loss-weight 2 \
  --deterministic \
  --disable-viewer


python run_gsplat_dual.py default \
  --no-nerf-init \
  --pt-path /cluster/scratch/rbollati/bonsai_models/nerf_ensemble_1/outputs/ensemble_1/nerfacto/nerf/ray_sample.pt \
  --data-dir /cluster/scratch/rbollati/dataset/bonsai_sparse_aug \
  --result-dir /cluster/scratch/rbollati/bonsai_models/gsplat_sfm_l2_nores_2 \
  --data-factor 1 \
  --nerf-samples-data-factor 8 \
  --batch-size 1 \
  --nerf-batch-factor 20 \
  --dual-nerf-decay-steps-to-quarter 300 \
  --dual-nerf-disable-threshold 0.1 \
  --max-steps 20000 \
  --strategy.reset-every 100000000 \
  --dual-nerf-loss-weight 0 \
  --deterministic \
  --disable-viewer

conda deactivate
conda activate nerfstudio

python ~/RadSplat/RadSplat/images.py \
  --nerf-folders /cluster/scratch/rbollati/counter_models/nerf_ensemble_1/outputs/ensemble_1/nerfacto/nerf/config.yml /cluster/scratch/rbollati/counter_models/nerf_ensemble_2/outputs/ensemble_1/nerfacto/nerf/config.yml /cluster/scratch/rbollati/counter_models/nerf_ensemble_3/outputs/ensemble_1/nerfacto/nerf/config.yml /cluster/scratch/rbollati/counter_models/nerf_ensemble_4/outputs/ensemble_1/nerfacto/nerf/config.yml /cluster/scratch/rbollati/counter_models/nerf_ensemble_5/outputs/ensemble_1/nerfacto/nerf/config.yml \
  --exp-dirs /cluster/scratch/rbollati/counter_models/nerf_ensemble_1 /cluster/scratch/rbollati/counter_models/nerf_ensemble_2 /cluster/scratch/rbollati/counter_models/nerf_ensemble_3 /cluster/scratch/rbollati/counter_models/nerf_ensemble_4 /cluster/scratch/rbollati/counter_models/nerf_ensemble_5 \
  --input-dataset /cluster/scratch/rbollati/dataset/counter_sparse \
  --output-dataset /cluster/scratch/rbollati/dataset/counter_sparse_aug \
  --tmp-root /cluster/home/rbollati/tmp \
  --tau 1.5 \
  --debug-plot-dir image_supervision \
  --num-final-samples 200 \
  --final-render-scale 0.125


conda deactivate
conda activate gsplat

python run_gsplat_dual.py default \
  --no-nerf-init \
  --pt-path /cluster/scratch/rbollati/counter_models/nerf_ensemble_1/outputs/ensemble_1/nerfacto/nerf/ray_sample.pt \
  --data-dir /cluster/scratch/rbollati/dataset/counter_sparse_aug \
  --result-dir /cluster/scratch/rbollati/counter_models/RadSplat_dual_full_nores_l2_1 \
  --data-factor 1 \
  --nerf-samples-data-factor 8 \
  --batch-size 1 \
  --nerf-batch-factor 20 \
  --dual-nerf-decay-steps-to-quarter 300 \
  --dual-nerf-disable-threshold 0.1 \
  --max-steps 20000 \
  --strategy.reset-every 100000000 \
  --dual-nerf-loss-weight 2 \
  --deterministic \
  --disable-viewer


python run_gsplat_dual.py default \
  --no-nerf-init \
  --pt-path /cluster/scratch/rbollati/counter_models/nerf_ensemble_1/outputs/ensemble_1/nerfacto/nerf/ray_sample.pt \
  --data-dir /cluster/scratch/rbollati/dataset/counter_sparse_aug \
  --result-dir /cluster/scratch/rbollati/counter_models/gsplat_sfm_l2_nores_2 \
  --data-factor 1 \
  --nerf-samples-data-factor 8 \
  --batch-size 1 \
  --nerf-batch-factor 20 \
  --dual-nerf-decay-steps-to-quarter 300 \
  --dual-nerf-disable-threshold 0.1 \
  --max-steps 20000 \
  --strategy.reset-every 100000000 \
  --dual-nerf-loss-weight 0 \
  --deterministic \
  --disable-viewer

conda deactivate
conda activate nerfstudio

python ~/RadSplat/RadSplat/images.py \
  --nerf-folders /cluster/scratch/rbollati/garden_models/nerf_ensemble_1/outputs/ensemble_1/nerfacto/nerf/config.yml /cluster/scratch/rbollati/garden_models/nerf_ensemble_2/outputs/ensemble_1/nerfacto/nerf/config.yml /cluster/scratch/rbollati/garden_models/nerf_ensemble_3/outputs/ensemble_1/nerfacto/nerf/config.yml /cluster/scratch/rbollati/garden_models/nerf_ensemble_4/outputs/ensemble_1/nerfacto/nerf/config.yml /cluster/scratch/rbollati/garden_models/nerf_ensemble_5/outputs/ensemble_1/nerfacto/nerf/config.yml \
  --exp-dirs /cluster/scratch/rbollati/garden_models/nerf_ensemble_1 /cluster/scratch/rbollati/garden_models/nerf_ensemble_2 /cluster/scratch/rbollati/garden_models/nerf_ensemble_3 /cluster/scratch/rbollati/garden_models/nerf_ensemble_4 /cluster/scratch/rbollati/garden_models/nerf_ensemble_5 \
  --input-dataset /cluster/scratch/rbollati/dataset/garden_sparse \
  --output-dataset /cluster/scratch/rbollati/dataset/garden_sparse_aug \
  --tmp-root /cluster/home/rbollati/tmp \
  --tau 1.5 \
  --debug-plot-dir image_supervision \
  --num-final-samples 200 \
  --final-render-scale 0.125

conda deactivate
conda activate gsplat

python run_gsplat_dual.py default \
  --no-nerf-init \
  --pt-path /cluster/scratch/rbollati/garden_models/nerf_ensemble_1/outputs/ensemble_1/nerfacto/nerf/ray_sample.pt \
  --data-dir /cluster/scratch/rbollati/dataset/garden_sparse_aug \
  --result-dir /cluster/scratch/rbollati/garden_models/RadSplat_dual_full_nores_l2_1 \
  --data-factor 1 \
  --nerf-samples-data-factor 8 \
  --batch-size 1 \
  --nerf-batch-factor 20 \
  --dual-nerf-decay-steps-to-quarter 300 \
  --dual-nerf-disable-threshold 0.1 \
  --max-steps 20000 \
  --strategy.reset-every 100000000 \
  --dual-nerf-loss-weight 2 \
  --deterministic \
  --disable-viewer


python run_gsplat_dual.py default \
  --no-nerf-init \
  --pt-path /cluster/scratch/rbollati/garden_models/nerf_ensemble_1/outputs/ensemble_1/nerfacto/nerf/ray_sample.pt \
  --data-dir /cluster/scratch/rbollati/dataset/garden_sparse_aug \
  --result-dir /cluster/scratch/rbollati/garden_models/gsplat_sfm_l2_nores_2 \
  --data-factor 1 \
  --nerf-samples-data-factor 8 \
  --batch-size 1 \
  --nerf-batch-factor 20 \
  --dual-nerf-decay-steps-to-quarter 300 \
  --dual-nerf-disable-threshold 0.1 \
  --max-steps 20000 \
  --strategy.reset-every 100000000 \
  --dual-nerf-loss-weight 0 \
  --deterministic \
  --disable-viewer

conda deactivate
conda activate nerfstudio

python ~/RadSplat/RadSplat/images.py \
  --nerf-folders /cluster/scratch/rbollati/kitchen_models/nerf_ensemble_1/outputs/ensemble_1/nerfacto/nerf/config.yml /cluster/scratch/rbollati/kitchen_models/nerf_ensemble_2/outputs/ensemble_1/nerfacto/nerf/config.yml /cluster/scratch/rbollati/kitchen_models/nerf_ensemble_3/outputs/ensemble_1/nerfacto/nerf/config.yml /cluster/scratch/rbollati/kitchen_models/nerf_ensemble_4/outputs/ensemble_1/nerfacto/nerf/config.yml /cluster/scratch/rbollati/kitchen_models/nerf_ensemble_5/outputs/ensemble_1/nerfacto/nerf/config.yml \
  --exp-dirs /cluster/scratch/rbollati/kitchen_models/nerf_ensemble_1 /cluster/scratch/rbollati/kitchen_models/nerf_ensemble_2 /cluster/scratch/rbollati/kitchen_models/nerf_ensemble_3 /cluster/scratch/rbollati/kitchen_models/nerf_ensemble_4 /cluster/scratch/rbollati/kitchen_models/nerf_ensemble_5 \
  --input-dataset /cluster/scratch/rbollati/dataset/kitchen_sparse \
  --output-dataset /cluster/scratch/rbollati/dataset/kitchen_sparse_aug \
  --tmp-root /cluster/home/rbollati/tmp \
  --tau 1.5 \
  --debug-plot-dir image_supervision \
  --num-final-samples 200 \
  --final-render-scale 0.125

conda deactivate
conda activate gsplat

python run_gsplat_dual.py default \
  --no-nerf-init \
  --pt-path /cluster/scratch/rbollati/kitchen_models/nerf_ensemble_1/outputs/ensemble_1/nerfacto/nerf/ray_sample.pt \
  --data-dir /cluster/scratch/rbollati/dataset/kitchen_sparse_aug \
  --result-dir /cluster/scratch/rbollati/kitchen_models/RadSplat_dual_full_nores_l2_1 \
  --data-factor 1 \
  --nerf-samples-data-factor 8 \
  --batch-size 1 \
  --nerf-batch-factor 20 \
  --dual-nerf-decay-steps-to-quarter 300 \
  --dual-nerf-disable-threshold 0.1 \
  --max-steps 20000 \
  --strategy.reset-every 100000000 \
  --dual-nerf-loss-weight 2 \
  --deterministic \
  --disable-viewer


python run_gsplat_dual.py default \
  --no-nerf-init \
  --pt-path /cluster/scratch/rbollati/kitchen_models/nerf_ensemble_1/outputs/ensemble_1/nerfacto/nerf/ray_sample.pt \
  --data-dir /cluster/scratch/rbollati/dataset/kitchen_sparse_aug \
  --result-dir /cluster/scratch/rbollati/kitchen_models/gsplat_sfm_l2_nores_2 \
  --data-factor 1 \
  --nerf-samples-data-factor 8 \
  --batch-size 1 \
  --nerf-batch-factor 20 \
  --dual-nerf-decay-steps-to-quarter 300 \
  --dual-nerf-disable-threshold 0.1 \
  --max-steps 20000 \
  --strategy.reset-every 100000000 \
  --dual-nerf-loss-weight 0 \
  --deterministic \
  --disable-viewer

conda deactivate
conda activate nerfstudio

python ~/RadSplat/RadSplat/images.py \
  --nerf-folders /cluster/scratch/rbollati/room_models/nerf_ensemble_1/outputs/ensemble_1/nerfacto/nerf/config.yml /cluster/scratch/rbollati/room_models/nerf_ensemble_2/outputs/ensemble_1/nerfacto/nerf/config.yml /cluster/scratch/rbollati/room_models/nerf_ensemble_3/outputs/ensemble_1/nerfacto/nerf/config.yml /cluster/scratch/rbollati/room_models/nerf_ensemble_4/outputs/ensemble_1/nerfacto/nerf/config.yml /cluster/scratch/rbollati/room_models/nerf_ensemble_5/outputs/ensemble_1/nerfacto/nerf/config.yml \
  --exp-dirs /cluster/scratch/rbollati/room_models/nerf_ensemble_1 /cluster/scratch/rbollati/room_models/nerf_ensemble_2 /cluster/scratch/rbollati/room_models/nerf_ensemble_3 /cluster/scratch/rbollati/room_models/nerf_ensemble_4 /cluster/scratch/rbollati/room_models/nerf_ensemble_5 \
  --input-dataset /cluster/scratch/rbollati/dataset/room_sparse \
  --output-dataset /cluster/scratch/rbollati/dataset/room_sparse_aug \
  --tmp-root /cluster/home/rbollati/tmp \
  --tau 1.5 \
  --debug-plot-dir image_supervision \
  --num-final-samples 200 \
  --final-render-scale 0.125

conda deactivate
conda activate gsplat

python run_gsplat_dual.py default \
  --no-nerf-init \
  --pt-path /cluster/scratch/rbollati/room_models/nerf_ensemble_1/outputs/ensemble_1/nerfacto/nerf/ray_sample.pt \
  --data-dir /cluster/scratch/rbollati/dataset/room_sparse_aug \
  --result-dir /cluster/scratch/rbollati/room_models/RadSplat_dual_full_nores_l2_1 \
  --data-factor 1 \
  --nerf-samples-data-factor 8 \
  --batch-size 1 \
  --nerf-batch-factor 20 \
  --dual-nerf-decay-steps-to-quarter 300 \
  --dual-nerf-disable-threshold 0.1 \
  --max-steps 20000 \
  --strategy.reset-every 100000000 \
  --dual-nerf-loss-weight 2 \
  --deterministic \
  --disable-viewer


python run_gsplat_dual.py default \
  --no-nerf-init \
  --pt-path /cluster/scratch/rbollati/room_models/nerf_ensemble_1/outputs/ensemble_1/nerfacto/nerf/ray_sample.pt \
  --data-dir /cluster/scratch/rbollati/dataset/room_sparse_aug \
  --result-dir /cluster/scratch/rbollati/room_models/gsplat_sfm_l2_nores_2 \
  --data-factor 1 \
  --nerf-samples-data-factor 8 \
  --batch-size 1 \
  --nerf-batch-factor 20 \
  --dual-nerf-decay-steps-to-quarter 300 \
  --dual-nerf-disable-threshold 0.1 \
  --max-steps 20000 \
  --strategy.reset-every 100000000 \
  --dual-nerf-loss-weight 0 \
  --deterministic \
  --disable-viewer

conda deactivate
conda activate nerfstudio

python ~/RadSplat/RadSplat/images.py \
  --nerf-folders /cluster/scratch/rbollati/stump_models/nerf_ensemble_1/outputs/ensemble_1/nerfacto/nerf/config.yml /cluster/scratch/rbollati/stump_models/nerf_ensemble_2/outputs/ensemble_1/nerfacto/nerf/config.yml /cluster/scratch/rbollati/stump_models/nerf_ensemble_3/outputs/ensemble_1/nerfacto/nerf/config.yml /cluster/scratch/rbollati/stump_models/nerf_ensemble_4/outputs/ensemble_1/nerfacto/nerf/config.yml /cluster/scratch/rbollati/stump_models/nerf_ensemble_5/outputs/ensemble_1/nerfacto/nerf/config.yml \
  --exp-dirs /cluster/scratch/rbollati/stump_models/nerf_ensemble_1 /cluster/scratch/rbollati/stump_models/nerf_ensemble_2 /cluster/scratch/rbollati/stump_models/nerf_ensemble_3 /cluster/scratch/rbollati/stump_models/nerf_ensemble_4 /cluster/scratch/rbollati/stump_models/nerf_ensemble_5 \
  --input-dataset /cluster/scratch/rbollati/dataset/stump_sparse \
  --output-dataset /cluster/scratch/rbollati/dataset/stump_sparse_aug \
  --tmp-root /cluster/home/rbollati/tmp \
  --tau 1.5 \
  --debug-plot-dir image_supervision \
  --num-final-samples 200 \
  --final-render-scale 0.125