BASE_DIR="/cluster/scratch/rbollati/dataset"
MODELS_ROOT="/cluster/scratch/rbollati"

for d in "$BASE_DIR"/*_sparse_vggt; do
  [ -d "$d" ] || continue
  scene="$(basename "$d")"
  echo "Scene: $scene"

  model_dir="$MODELS_ROOT/models_${scene}"
  mkdir -p "$model_dir"
  cd "$model_dir" || exit 1
  
  for i in 1 2 3 4 5; do
    run_dir="nerf_ensemble_${i}"
    mkdir -p "$run_dir"
    cd "$run_dir" || exit 1

    echo "  Ensemble: $i"

    ns-train depth-nerfacto \
        --vis tensorboard \
        --experiment-name ensemble_1 \
        --timestamp depth \
        --steps-per-eval-image 30000 \
        --max-num-iterations 30000 \
        --save-only-latest-checkpoint True \
        --logging.steps-per-log 100 \
        colmap \
        --downscale-factor 1 \
        --colmap-path "$BASE_DIR/$scene/sparse/0" \
        --images-path "$BASE_DIR/$scene/images" \
        --depths-path "$BASE_DIR/$scene/depths_vggt"

    cd ..
  done
done