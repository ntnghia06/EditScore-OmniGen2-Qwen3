#!/bin/bash
#SBATCH --job-name=omnigen2_gedit
#SBATCH --output=logs/omnigen2_gedit_%j.out
#SBATCH --error=logs/omnigen2_gedit_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -euo pipefail
BUNDLE_DIR="${SLURM_SUBMIT_DIR:?Submit from h200_qwen3_rl}"
test -f "$BUNDLE_DIR/settings.json" || { echo 'ERROR: cd to h200_qwen3_rl before sbatch' >&2; exit 1; }
cd "$BUNDLE_DIR"
source /datastore/truongtq/miniconda3/etc/profile.d/conda.sh
TRAIN_ENV=""
for candidate in \
    /datastore/anhndt/Truongtq/miniconda3/envs/omnigen2-rl \
    /datastore/anhndt/miniconda3/envs/omnigen2-rl \
    /datastore/truongtq/miniconda3/envs/omnigen2-rl; do
    if [ -x "$candidate/bin/python" ]; then TRAIN_ENV="$candidate"; break; fi
done
test -n "$TRAIN_ENV" || { echo 'ERROR: set TRAIN_ENV to your OmniGen2 environment' >&2; exit 1; }
conda activate "$TRAIN_ENV"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$BUNDLE_DIR/outputs/hf_cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# A fresh cache lookup avoids stale .pyc from the earlier damaged source checkout.
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/omnigen2_gedit_pycache_${SLURM_JOB_ID}"
BENCH=/datastore/anhndt/Truongtq/OtherEditingBenchmark
nvidia-smi
EXTRA_ARGS=()
# Offload is enabled on MIG 35GB; set CPU_OFFLOAD=0 to keep all components on GPU.
if [ "${CPU_OFFLOAD:-1}" = 1 ]; then EXTRA_ARGS+=(--cpu-offload); fi
python -B generate_omnigen2_gedit.py \
    --settings "$BUNDLE_DIR/settings.json" \
    --metadata "$BENCH/GEdit-Bench-EN/metadata.jsonl" \
    --data-root "$BENCH/GEdit-Bench-EN" \
    --output-root "$BENCH/OmniGen2-Base/GEdit-Bench-EN" \
    --seed 20 --steps 50 --text-guidance 5.0 --image-guidance 1.5 \
    --time-shift-base-res 168 --max-long-edge 1024 \
    "${EXTRA_ARGS[@]}" "$@"
