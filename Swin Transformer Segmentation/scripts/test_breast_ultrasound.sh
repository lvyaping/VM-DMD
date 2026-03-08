#!/bin/bash

# Breast Ultrasound test script
# Evaluate test data using trained model

ROOT_DIR="/home/lvyp/Femas/Swin-Transformer-Semantic-Segmentation"
cd "$ROOT_DIR"

CONFIG_FILE="configs/swin/upernet_swin_tiny_patch4_window7_224x224_160k_breast_ultrasound.py"
# Default to latest checkpoint (160k iterations), can be modified as needed
CHECKPOINT_FILE="/home/lvyp/Femas/Swin-Transformer-Semantic-Segmentation/segmentation_pretrained_models/BUSI.pth"
TEST_DATA_ROOT="/home/lvyp/Breast_ultrasound/"

echo "=========================================="
echo "Starting Breast Ultrasound dataset testing"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "Checkpoint file: $CHECKPOINT_FILE"
echo "Test data: $TEST_DATA_ROOT/test"
echo "=========================================="
echo ""
echo "Note: BreastUltrasoundDataset's evaluate method will compute and display all 6 metrics:"
echo "  - Miou (Mean Intersection over Union)"
echo "  - Dsc  (Dice Similarity Coefficient)"
echo "  - Acc  (Accuracy)"
echo "  - Spe  (Specificity)"
echo "  - Sen  (Sensitivity)"
echo "  - Pre  (Precision)"
echo "The --eval parameter value won't affect output, as BreastUltrasoundDataset ignores it and always computes all metrics"
echo "=========================================="
echo ""

# Check if checkpoint file exists
if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_FILE"
    echo "Please check the path or train the model"
    exit 1
fi

# Run testing
# Note: BreastUltrasoundDataset's evaluate method will compute and display all 6 metrics:
# - Miou (Mean Intersection over Union)
# - Dsc (Dice Similarity Coefficient)
# - Acc (Accuracy)
# - Spe (Specificity)
# - Sen (Sensitivity)
# - Pre (Precision)
# The --eval parameter value won't affect output, as BreastUltrasoundDataset ignores it and always computes all metrics
python tools/test.py \
    $CONFIG_FILE \
    $CHECKPOINT_FILE \
    --eval mIoU mDice \
    --options \
        data.test.data_root=$TEST_DATA_ROOT \
        data.test.img_dir=test/image \
        data.test.ann_dir=test/mask \
    --show-dir test_results_BUSI

echo ""
echo "=========================================="
echo "Testing completed! Results saved in test_results_breast_ultrasound/ directory"
echo "=========================================="
