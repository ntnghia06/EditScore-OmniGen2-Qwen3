#!/bin/bash
#SBATCH --job-name=omnigen2_qwen3
#SBATCH --output=logs/omnigen2_qwen3_%j.out
#SBATCH --error=logs/omnigen2_qwen3_%j.err
#SBATCH --account=uit
#SBATCH --partition=full
#SBATCH --qos=mig7-141
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:7g.141gb:1

set -euo pipefail
# Submit from this folder: mkdir -p logs; sbatch run_train.slurm
BUNDLE_DIR="${SLURM_SUBMIT_DIR:?Submit from h200_qwen3_rl}"
test -f "$BUNDLE_DIR/settings.json" || { echo 'ERROR: cd to h200_qwen3_rl before sbatch' >&2; exit 1; }
CONDA_SH=/datastore/truongtq/miniconda3/etc/profile.d/conda.sh
REWARD_ENV=/datastore/truongtq/miniconda3/envs/qwen_vl
# Select an existing OmniGen2 training environment automatically; override here if needed.
TRAIN_ENV=""
for candidate in \
    /datastore/anhndt/Truongtq/miniconda3/envs/omnigen2-rl \
    /datastore/anhndt/miniconda3/envs/omnigen2-rl \
    /datastore/truongtq/miniconda3/envs/omnigen2-rl; do
    if [ -x "$candidate/bin/python" ]; then TRAIN_ENV="$candidate"; break; fi
done
test -n "$TRAIN_ENV" || { echo 'ERROR: set TRAIN_ENV in run_train.slurm to your OmniGen2 environment' >&2; exit 1; }
source "$CONDA_SH"
conda activate "$TRAIN_ENV"
TRAIN_PY="$TRAIN_ENV/bin/python"
REWARD_PY="$REWARD_ENV/bin/python"
test -x "$REWARD_PY"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HOME="$BUNDLE_DIR/outputs/hf_cache"
export HF_DATASETS_CACHE="$BUNDLE_DIR/outputs/datasets_cache"
export MPLCONFIGDIR="$BUNDLE_DIR/outputs/matplotlib"
export WANDB_MODE=disabled
unset RANK WORLD_SIZE LOCAL_RANK ACCELERATE_USE_FSDP ACCELERATE_USE_DEEPSPEED
mkdir -p "$BUNDLE_DIR/outputs" "$HF_DATASETS_CACHE" "$MPLCONFIGDIR" "$BUNDLE_DIR/logs"
RUN_DIR="$BUNDLE_DIR/outputs/job_${SLURM_JOB_ID}"
nvidia-smi
"$TRAIN_PY" -B -c 'from transformers import Qwen2_5_VLModel; import accelerate, peft, diffusers, yaml, timm; print("Policy dependencies OK", flush=True)'
"$REWARD_PY" -B -c 'from transformers import Qwen3VLForConditionalGeneration; import peft, swift, qwen_vl_utils; print("Reward dependencies OK", flush=True)'
"$TRAIN_PY" -B "$BUNDLE_DIR/prepare_run.py" --bundle "$BUNDLE_DIR" --run-dir "$RUN_DIR"

REWARD_PID=""
cleanup() {
    if [ -n "$REWARD_PID" ]; then kill "$REWARD_PID" 2>/dev/null || true; wait "$REWARD_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT
"$REWARD_PY" -B "$RUN_DIR/reward_server.py" --settings "$RUN_DIR/settings.json" \
    > "$BUNDLE_DIR/logs/reward_${SLURM_JOB_ID}.log" 2>&1 &
REWARD_PID=$!
"$TRAIN_PY" -B - "$RUN_DIR/settings.json" "$REWARD_PID" <<'PY'
import json, os, sys, time, urllib.request
from pathlib import Path
settings = json.load(open(sys.argv[1]))
pid = int(sys.argv[2])
url = f'http://127.0.0.1:{settings["port"]}/ping'
for attempt in range(600):
    os.kill(pid, 0)
    state = Path(f'/proc/{pid}/stat')
    if state.exists() and state.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
        raise RuntimeError('Reward worker exited; inspect logs/reward_JOBID.log')
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            assert json.load(response)['ready']
        print('Reward server ready', flush=True)
        break
    except (OSError, ValueError):
        if attempt % 12 == 0: print('Waiting for frozen reward model to load...', flush=True)
        time.sleep(5)
else:
    raise TimeoutError('Reward server did not become ready; inspect logs/reward_JOBID.log')
PY
cd "$RUN_DIR"
echo "Training config: $RUN_DIR/train.yml"
echo "Checkpoints: $RUN_DIR/checkpoints"
"$TRAIN_PY" -B "$RUN_DIR/train_h200.py" --config "$RUN_DIR/train.yml"
