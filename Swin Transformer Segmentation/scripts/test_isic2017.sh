#!/bin/bash

# ISIC2017 test script
# Evaluate test data using trained model

ROOT_DIR="/home/lvyp/Femas/Swin-Transformer-Semantic-Segmentation"
cd "$ROOT_DIR"

CONFIG_FILE="configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_isic2017.py"
CHECKPOINT_FILE="/home/lvyp/Femas/Swin-Transformer-Semantic-Segmentation/segmentation_pretrained_models/isic18.pth"
TEST_DATA_ROOT="/home/lvyp/ISIC2018"

echo "=========================================="
echo "Starting ISIC2017 dataset testing"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "Checkpoint file: $CHECKPOINT_FILE"
echo "Test data: $TEST_DATA_ROOT/test"
echo "=========================================="

# Run testing
# Note: ISIC2017Dataset's evaluate method will compute and display all 5 metrics:
# - Miou (Mean Intersection over Union)
# - Dsc (Dice Similarity Coefficient)
# - Acc (Accuracy)
# - Spe (Specificity)
# - Sen (Sensitivity)
# The --eval parameter value won't affect output, as ISIC2017Dataset ignores it and always computes all metrics
python tools/test.py \
    $CONFIG_FILE \
    $CHECKPOINT_FILE \
    --eval mIoU mDice \
    --options \
        data.test.data_root=$TEST_DATA_ROOT \
        data.test.img_dir=test/image \
        data.test.ann_dir=test/mask \
    --show-dir test2018_results

echo "=========================================="
echo "Testing completed! Results saved in test_results/ directory"
echo "=========================================="

