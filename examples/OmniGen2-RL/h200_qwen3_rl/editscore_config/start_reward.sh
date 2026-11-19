#!/bin/bash
set -euo pipefail
RUN_DIR="$(realpath "${1:?Usage: bash start_reward.sh /absolute/run_dir}")"
source "${CONDA_SH:-/datastore/truongtq/miniconda3/etc/profile.d/conda.sh}"
conda activate "${REWARD_ENV:-/datastore/truongtq/miniconda3/envs/qwen_vl}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -c 'import torch; assert torch.cuda.device_count()==1, "Expose exactly one allocated reward GPU"'
exec python -B "$RUN_DIR/reward_server.py" --settings "$RUN_DIR/settings.json"
