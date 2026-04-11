BASE_DIR="/cluster/scratch/rbollati/dataset"
MODELS_ROOT="/cluster/scratch/rbollati"
TMP_ROOT="/cluster/home/rbollati/tmp"

for d in "$BASE_DIR"/*_sparse_vggt; do
  [ -d "$d" ] || continue
  scene="$(basename "$d")"
  prefix="${scene%%_sparse_vggt}"

  echo "Scene: $scene"
  echo "Prefix: $prefix"

  model_dir="$MODELS_ROOT/models_${prefix}"

  cfg1="$model_dir/nerf_ensemble_1/outputs/ensemble_1/depth-nerfacto/depth/config.yml"
  cfg2="$model_dir/nerf_ensemble_2/outputs/ensemble_1/depth-nerfacto/depth/config.yml"
  cfg3="$model_dir/nerf_ensemble_3/outputs/ensemble_1/depth-nerfacto/depth/config.yml"
  cfg4="$model_dir/nerf_ensemble_4/outputs/ensemble_1/depth-nerfacto/depth/config.yml"
  cfg5="$model_dir/nerf_ensemble_5/outputs/ensemble_1/depth-nerfacto/depth/config.yml"

  for cfg in "$cfg1" "$cfg2" "$cfg3" "$cfg4" "$cfg5"; do
    [ -f "$cfg" ] || { echo "Missing config: $cfg"; continue 2; }
  done

  python ~/RadSplat/RadSplat/images.py \
    --nerf-folders \
      "$cfg1" \
      "$cfg2" \
      "$cfg3" \
      "$cfg4" \
      "$cfg5" \
    --exp-dirs \
      "$model_dir/nerf_ensemble_1" \
      "$model_dir/nerf_ensemble_2" \
      "$model_dir/nerf_ensemble_3" \
      "$model_dir/nerf_ensemble_4" \
      "$model_dir/nerf_ensemble_5" \
    --input-dataset "$BASE_DIR/$scene" \
    --output-dataset "$BASE_DIR/${prefix}_sparse_aug" \
    --tmp-root "$TMP_ROOT" \
    --tau 1.5 \
    --debug-plot-dir image_supervision \
    --num-final-samples 200 \
    --final-render-scale 0.125 \
    --camera-id 1
done
