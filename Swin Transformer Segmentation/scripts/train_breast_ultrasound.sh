#!/usr/bin/env bash
# Swin UPerNet Breast Ultrasound training script (224x224)
ROOT_DIR="/home/lvyp/Femas/Swin-Transformer-Semantic-Segmentation"
cd "$ROOT_DIR"
set -euo pipefail

# --------- Configuration ---------
CONFIG="configs/swin/upernet_swin_tiny_patch4_window7_224x224_160k_breast_ultrasound.py"
GPUS=1  # Single GPU training to avoid NCCL conflicts
RESUME_FROM=""
PRETRAINED="$ROOT_DIR/pretrained/swin_tiny_patch4_window7_224.pth"
WORK_DIR="$ROOT_DIR/work_dirs/upernet_swin_tiny_patch4_window7_224x224_160k_breast_ultrasound"
PORT=29502  # Use different port
# ---------------------------------

export PORT=$PORT

# training command
CMD=(
    tools/dist_train.sh 
    "$CONFIG" 
    "$GPUS" 
    --work-dir "$WORK_DIR"
    --options "model.pretrained=$PRETRAINED"
)

if [[ -n "${RESUME_FROM}" ]]; then
  CMD=( "${CMD[@]:0:5}" --resume-from "$RESUME_FROM" "${CMD[@]:5}" )
fi

echo "=========================================="
echo "Training configuration (Breast Ultrasound 224x224):"
echo "  Project root: $ROOT_DIR"
echo "  Config file: $CONFIG"
echo "  Number of GPUs: $GPUS"
echo "  Work directory: $WORK_DIR"
echo "  Pretrained model: $PRETRAINED"
echo ""
echo "Configuration details:"
echo "  Dataset: BreastUltrasoundDataset"
echo "  Image size: 224×224"
echo "  Training iterations: 160k"
echo "  Loss function: DiceCELoss (dice_weight=1.0, ce_weight=1.0)"
echo "  Class weights: [0.1, 0.9] (background, lesion)"
echo "=========================================="
echo ""

echo "Running: ${CMD[@]}"
"${CMD[@]}"

