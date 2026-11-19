#!/bin/bash
set -euo pipefail
RUN_DIR="$(realpath "${1:?Usage: bash launch_policy.sh /absolute/run_dir [rank] [master_addr]}")"
MACHINE_RANK="${2:-0}"
MASTER_ADDR="${3:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
source "${CONDA_SH:-/datastore/truongtq/miniconda3/etc/profile.d/conda.sh}"
conda activate "${TRAIN_ENV:-/datastore/truongtq/miniconda3/envs/omnigen2-rl}"
export RUN_DIR PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RL_DIR="$(cat "$RUN_DIR/repo_rl_path.txt")"
export PYTHONPATH="$RL_DIR${PYTHONPATH:+:$PYTHONPATH}"
mapfile -t TOPOLOGY < <(python -c 'import json,os; from pathlib import Path; c=json.loads((Path(os.environ["RUN_DIR"])/"launch.json").read_text()); print(c["policy_processes"]); print(c["machines"])')
python -c 'import torch; assert torch.cuda.device_count()==8, "Expose exactly eight allocated policy GPUs per machine"'
cd "$RUN_DIR"
accelerate launch \
    --machine_rank="$MACHINE_RANK" --main_process_ip="$MASTER_ADDR" \
    --main_process_port="$MASTER_PORT" --num_machines="${TOPOLOGY[1]}" \
    --num_processes="${TOPOLOGY[0]}" --use_fsdp \
    --fsdp_offload_params false --fsdp_sharding_strategy HYBRID_SHARD \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap OmniGen2TransformerBlock \
    --fsdp_state_dict_type SHARDED_STATE_DICT --fsdp_forward_prefetch false \
    --fsdp_use_orig_params True --fsdp_cpu_ram_efficient_loading false \
    --fsdp_sync_module_states True \
    "$RUN_DIR/train_distributed.py" --config "$RUN_DIR/train.yml"
