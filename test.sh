#!/usr/bin/env bash

GPUS=$1
PORT=$2
CONFIG=$3
CHECKPOINT=$4

# shellcheck disable=SC2068
python -m torch.distributed.launch --nproc_per_node="$GPUS" --master_port="$PORT" test.py "$CONFIG" "$CHECKPOINT" --launcher pytorch ${@:5}
