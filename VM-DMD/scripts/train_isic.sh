#!/usr/bin/env bash
# bash scripts/train_multi_gpu.sh 4

GPUS=${1:-1}  

find_free_port() {
    local port=$1
    while lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; do
        port=$((port + 1))
    done
    echo $port
}

PORT=${PORT:-$(find_free_port 29500)}

echo "=========================================="
echo "Multi-GPU distributed training."
echo "GPUS: $GPUS"
echo "PORT: $PORT"
echo "=========================================="

export PYTHONPATH="CODE_PATH/VM-DMD:$PYTHONPATH"

cd VM-DMD

python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    train.py

echo "Finished！"

