#!/usr/bin/env bash
# Swin UPerNet ISIC2017 training with Dice+CE Loss
ROOT_DIR="/home/lvyp/Femas/Swin-Transformer-Semantic-Segmentation"
cd "$ROOT_DIR"
set -euo pipefail

# --------- Configuration ---------
CONFIG="configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_isic2017.py"
GPUS=1
# Optional: resume training from checkpoint (leave empty to train from scratch)
# Example: RESUME_FROM="$ROOT_DIR/work_dirs/xxx/iter_96000.pth"
RESUME_FROM=""
PRETRAINED="$ROOT_DIR/pretrained/swin_tiny_patch4_window7_224.pth"
# work_dir save location
WORK_DIR="$ROOT_DIR/work_dirs/upernet_swin_tiny_patch4_window7_512x512_160k_isic2017"
# ---------------------------------

# training command
CMD=(
    tools/dist_train.sh 
    "$CONFIG" 
    "$GPUS" 
    --work-dir "$WORK_DIR"
    --options "model.pretrained=$PRETRAINED"
)
# Only add --resume-from if RESUME_FROM is set
if [[ -n "${RESUME_FROM}" ]]; then
  CMD=( "${CMD[@]:0:5}" --resume-from "$RESUME_FROM" "${CMD[@]:5}" )
fi

echo "=========================================="
echo "Training configuration:"
echo "  Project root: $ROOT_DIR"
echo "  Config file: $ROOT_DIR/$CONFIG"
echo "  Number of GPUs: $GPUS"
echo "  Work directory: $WORK_DIR"
echo "  Resume checkpoint: ${RESUME_FROM:-<train from scratch>}"
echo "  Pretrained model: $PRETRAINED"
echo "  Loss function: DiceCELoss (dice_weight=1.0, ce_weight=1.0)"
echo "  Class weights: [0.1, 0.9] (background, lesion)"
echo "=========================================="
echo ""

echo "Running: ${CMD[@]}"
"${CMD[@]}"

