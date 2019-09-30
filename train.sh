#!/usr/bin/env bash

GPUS=$1
PORT=$2
CONFIG=$3

# shellcheck disable=SC2068
python -m torch.distributed.launch --nproc_per_node="$GPUS" --master_port="$PORT" train.py "$CONFIG" --launcher pytorch ${@:4}
