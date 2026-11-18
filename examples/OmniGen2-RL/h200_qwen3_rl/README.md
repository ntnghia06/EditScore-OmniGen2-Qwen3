# OmniGen2 Flow-GRPO + Qwen3-VL reward, one H200 141GB

Upload this entire `h200_qwen3_rl` folder into **your existing EditScore repository**:

```text
EditScore-original/                         # or EditScore; the repo name is irrelevant
  examples/OmniGen2-RL/
    omnigen2/                              # original repo package, required
    h200_qwen3_rl/                          # upload this folder here
      settings.json
      train.yml
      prepare_run.py
      train_h200.py
      reward_runtime.py
      reward_server.py
      run_train.slurm
      test_integration.py
      README.md
```

All added code is contained in this folder. Original repo files are unchanged.
This uses EditScore's trainer, **not Flow-Factory**. It does not require vLLM or
the original EditScore reward server/proxy.

## Server preparation

The submitted user must be able to read the checkpoint, backbone, architecture
and training images, and write into this folder. The requested dataset is:

```text
/datastore/anhndt/Truongtq/TrainingEditingModel/RunFLUX/unique_metadata.jsonl
```

Relative source paths resolve against:

```text
/datastore/shared/datasets/EditReward-Data/extracted_data
```

Required row fields: `instruction` (or `prompt`) and `source_img` (or
`source_image`/`src_image`). Existing edited images are ignored. Preparation
validates all source images and deduplicates by source path + instruction.
It creates the `input_images`/`meta_data` format expected by OmniGen2.

Use the existing OmniGen2 environment. `run_train.slurm` searches these paths:

```text
/datastore/anhndt/Truongtq/miniconda3/envs/omnigen2-rl
/datastore/anhndt/miniconda3/envs/omnigen2-rl
/datastore/truongtq/miniconda3/envs/omnigen2-rl
```

If your environment is elsewhere, set `TRAIN_ENV` to its absolute path and
remove the search loop. The reward process uses the existing environment:

```text
/datastore/truongtq/miniconda3/envs/qwen_vl
```

It needs the same dependencies as the previous Stage2 scorer, including
Qwen3-VL in Transformers, PEFT, ms-swift and qwen-vl-utils. Policy and reward
use **separate environments**, share the allocated GPU and run forwards
sequentially. Both sets of model weights remain resident in GPU memory.

## Set local pretrained paths

Open `settings.json`. These reward paths are already configured:

```text
reward_checkpoint:
/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/Qwen3-VL-Embedding/output_training/Stage2_R0RelKL10kDecay98CosineOnceBatch4Eval5

reward_pretrained:
/datastore/truongtq/TrainingRewardModel/Models/Qwen3-VL-Reranker-8B
```

For a training directory the loader chooses `best.pt` if it exists. Otherwise
it accepts a PEFT export or an unambiguous training checkpoint. If multiple
files remain, specify the exact `.pt` path; it refuses to guess. It restores
and validates all LoRA tensors and all six-metric heads. Architecture and
prompt definitions are read from the configured Qwen3 project.

Set these three fields to the **existing local model paths** when known:

```json
{
  "omnigen_transformer_bin": "/actual/path/OmniGen2/transformer/pytorch_model.bin",
  "text_encoder": "/actual/path/Qwen2.5-VL-3B-Instruct",
  "vae_model": "/actual/path/FLUX.1-dev"
}
```

The VAE directory must contain `vae/config.json`; it is the FLUX.1-dev VAE
used by the original OmniGen2 implementation, not a trainable FLUX policy.
With empty fields, the transformer defaults to the repo's existing
`pretrained_models/OmniGen2/transformer/pytorch_model.bin`; text encoder and VAE
are searched offline in `hf_cache_roots`. It does **not** download weights.
Add the actual HF hub directory to `hf_cache_roots` if needed. A model ID alone
is not a local path in these explicit fields.

## Submit

For example, if the repo was uploaded under
`/datastore/anhndt/Truongtq/TrainingEditingModel/EditScore-original`:

```bash
cd /datastore/anhndt/Truongtq/TrainingEditingModel/EditScore-original/examples/OmniGen2-RL/h200_qwen3_rl
mkdir -p logs

# Normalize uploaded line endings. No sed and no external conda activation needed.
python3 - <<'PY'
from pathlib import Path
p = Path('run_train.slurm')
p.write_bytes(p.read_bytes().replace(b'\r\n', b'\n').removeprefix(b'\xef\xbb\xbf'))
PY

sbatch run_train.slurm
```

Always submit **from this folder**, because the Slurm script uses
`SLURM_SUBMIT_DIR`. The Slurm job activates both environments itself.
Requests: partition `full`, QoS `mig7-141`, one `gpu:7g.141gb`, 16 CPUs,
128GB system RAM, 72 hours. This allocation still needs to be available.

## Training and reward

`train.yml` retains the original EditScore PPO clipping, KL formula and
timestep loss reweighting. It keeps rank 32 / alpha 64, learning rate 4e-4,
20 SDE sampling steps, sigma coefficient 0.7, text/image guidance 4/2,
CFG range 0..0.6 and 12 sampled training timesteps out of 20. BF16 frozen
transformer/text encoder/VAE, FP32 trainable LoRA, gradient checkpointing.
The local copy also stops the outer data loop at `max_train_steps` and saves
the final checkpoint even when no periodic checkpoint has been written yet.

The one-GPU batch is **one source + instruction, 12 independently sampled
images**. Forward batch is 1, gradient accumulation config is 1 (the trainer
internally accumulates across the 12 selected timesteps), giving **12 optimizer
updates per rollout group**. This is a deliberate reduction from the original
8-GPU batch and does not reproduce its effective batch or update schedule.

The reward server scores 12 images in **6 pair forwards**. For a group, it
averages the six relevance vectors, selects metrics with relevance >= 0.5,
then computes the arithmetic mean of selected raw metric scores for each image.
No division by 10, clipping to [0,10], tie probability, or pairwise win probability
is applied. GRPO normalizes these per-image rewards within the group. Scores
can be negative or exceed 10. Reward parameters are frozen and in eval mode.

By default, if no metric is selected, the job **fails with an explicit error**.
To choose a defined fallback instead, set `empty_metric_policy` to
`highest_relevance`; that metric is then used and the server prints a warning.
`reward_max_pixels: 262144` controls Qwen3 visual preprocessing, separate from
policy image preprocessing. Change it to `null` to use the processor default.

The dataset preprocessing limits image area to 512^2 and preserves aspect
ratio, with dimensions aligned to the OmniGen2 codec. It does not force every
source to a 512x512 square.

**1000 is optimizer steps, not an epoch.** At 12 updates/group, it reaches
1000 steps after scoring 84 groups, with a partial final update group. It does
not traverse all ~8k conditions. For one complete traversal with N unique
conditions, use `max_train_steps: 12*N` (e.g. 97584 for N=8132). That is much more
training than the default run and is not promised to fit the 72-hour allocation.
The retained 4e-4 learning rate is the repo value, not a tuned value for Qwen3.

## Logs, checkpoints and reruns

```text
h200_qwen3_rl/
  logs/
    omnigen2_qwen3_JOBID.out
    omnigen2_qwen3_JOBID.err
    reward_JOBID.log
  outputs/
    job_JOBID/
      train.yml, settings.json, data.yml, training_pairs.jsonl
      train_h200.py, reward_server.py, reward_runtime.py
      checkpoints/
        logs/
        tokenizer/
        checkpoint-120/
          model.safetensors, optimizer.bin, ...
          lora_adapter/
            adapter_config.json
            adapter_model.safetensors
        ...
```

Accelerate filenames can vary with the installed version. Full optimizer/RNG
checkpoints are saved every 120 optimizer steps and at the end; at most two
checkpoint directories are retained by the original retention logic. The
additional `lora_adapter` export is for inference. Frozen text encoder weights
are not copied into outputs. Full checkpoints still include the frozen
transformer, so reserve tens of GB and check your **account quota**, not only
filesystem free space.

Each submit creates a new `outputs/job_JOBID` and starts fresh. The config and
three executable Python files are frozen when the job starts; edits while a
job is queued still apply when it starts. This integration does not implement
an exact mid-rollout resume protocol. Do not assume that changing
`resume_from_checkpoint` provides bit-for-bit continuation.

The server log reports groups/images, mean/min/max reward and elapsed seconds
after every scoring request. The trainer prints optimizer progress and losses.
The server is stopped when the training job exits. Any reward failure stops
training rather than substituting zero rewards.

## Validation scope

Local tests exercise metadata conversion, unique-pair deduplication, offline
path resolution, HTTP reward transport/order, metric selection and the
single-process adaptations. They do not load either model or establish peak
VRAM, convergence, wall time or compatibility of an unknown server environment.
On the server, first set `max_train_steps: 12` and `checkpointing_steps: 12`
to validate a complete rollout/reward/backward/save cycle, then restore the
desired values and submit a fresh job.
