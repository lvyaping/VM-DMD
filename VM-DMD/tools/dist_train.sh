#!/usr/bin/env bash

# 
# bash tools/dist_train.sh configs/config_setting.py 4

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

RESUME_FROM=${3:-""}
WORK_DIR=${4:-""}

export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

CMD="python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT train.py"

if [ ! -z "$WORK_DIR" ]; then
    CMD="$CMD --work-dir $WORK_DIR"
fi

if [ ! -z "$RESUME_FROM" ]; then
    CMD="$CMD --resume-from $RESUME_FROM"
fi

echo "Running: $CMD"
echo "GPUS: $GPUS, PORT: $PORT"

$CMD

