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
      prepare_pretrained.slurm
      train_h200.py
      compact_checkpoint.py
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

If the transformer was downloaded in Diffusers format to
`outputs/pretrained/OmniGen2/transformer`, convert it on a GPU compute node:

```bash
mkdir -p logs
sbatch prepare_pretrained.slurm
```

Wait for this job to succeed before submitting training. It reads only local
weights and creates `outputs/pretrained/OmniGen2/transformer/pytorch_model.bin`
in BF16. Set `omnigen_transformer_bin` to that absolute path. It validates an
existing `.bin` and skips conversion if one is already present. The converter
uses a normal-partition MIG 35GB allocation and the same environment search as
the training script. A disk quota error must be resolved before restoring
source, converting weights or saving checkpoints; cloning again cannot bypass
the storage limit.

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
      train_h200.py, compact_checkpoint.py, reward_server.py, reward_runtime.py
      checkpoints/
        logs/
        tokenizer/
        checkpoint-120/
          optimizer.bin, random_states_0.pkl, lora_scheduler.pt, ...
          omnigen2_checkpoint.json, checkpoint_complete.json
          lora_adapter/
            adapter_config.json
            adapter_model.safetensors
        ...
```

Accelerate filenames can vary with the installed version. By default,
`train.checkpoint_format` is `lora` (the key can be omitted). Save/load hooks
replace full model serialization with the portable `lora_adapter` export.
Optimizer/scaler/RNG state is still saved by Accelerate, and the scheduler is
also saved explicitly. Frozen transformer, VAE and text encoder weights are
not copied into each checkpoint. For rank 32 / ~20.43M FP32 LoRA parameters,
LoRA weights plus two FP32 Adam moments require roughly 245MB before minor
metadata overhead, instead of a full transformer copy of several GB. These
are estimates; actual artifacts determine the size printed at save time.

Checkpoints are saved at the configured interval and at the end (the template
uses 120 steps and retains two). The same `lora_adapter` directory works with
the Fireward inference script. Frozen pretrained weights must remain available
when restoring training. The hooks validate LoRA configuration and pretrained
paths; they can also load a legacy full-model checkpoint via Accelerate's
normal loader. This is model/optimizer restoration, not a new exact mid-rollout
resume implementation. To explicitly retain the former full-model behavior,
set `train.checkpoint_format: full`; this again requires several GB per save.

Checkpoint files are first written to `.checkpoint-N.incomplete`, fsynced, and
then renamed to `checkpoint-N` only on success. A quota failure leaves the
hidden incomplete folder for inspection; it is not a completed training state.
The Slurm script also performs a real 512MiB write/fsync/delete probe before
loading either model. This can fail early on quota problems, but does not
reserve space or guarantee that later writes fit. Keep enough quota for all
retained checkpoints and other jobs. `df` alone does not show user quota.

Each submit creates a new `outputs/job_JOBID` and starts fresh. The config and
four executable Python files are frozen when the job starts; edits while a
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

## Pretrained OmniGen2 inference on GEdit

`generate_omnigen2_gedit.py` loads the original transformer, Qwen2.5-VL text
encoder and VAE from explicit command-line paths. Set `TRANSFORMER_PATH`,
`TEXT_ENCODER_PATH` and `VAE_MODEL_PATH` at the top of
`run_generate_omnigen2_gedit.slurm`. It loads no RL checkpoint or LoRA. A
converted `.bin` is supported; if absent, an available local Diffusers
`transformer/` directory is supported instead. No weights are downloaded.
Transformer architecture is read from its adjacent pretrained `config.json`,
not from the training YAML. Neither `settings.json` nor `train.yml` is required
for inference. `--settings` remains an optional fallback for older invocations.

Defaults follow the EditScore GEdit evaluation launcher: Euler (ODE), 50 steps,
text guidance 5.0, image guidance 1.5, CFG over the entire trajectory, original
empty negative-prompt template, time-shift base resolution 168. Seed 20 matches
the earlier benchmark generators. These inference defaults differ from the
20-step SDE training rollout settings.

Source images preserve aspect ratio, with long edge at most 1024 and no
upscaling. Edge padding aligns the model canvas to multiples of 16, and is
cropped off before saving. Output contains only edited PNG images:

```text
/datastore/anhndt/Truongtq/OtherEditingBenchmark/OmniGen2-Base/GEdit-Bench-EN/
  background_change/<source_filename>.png
  ...
```

Run from this folder after editing the pretrained paths in the Slurm file:

```bash
mkdir -p logs
sbatch run_generate_omnigen2_gedit.slurm
```

Set `LIMIT=0` in the Slurm file to process all samples (the default), or
`LIMIT=100` for the first 100. The Python flags are `--limit 100` and optional
`--start-index 100`; omitting `--limit` or setting it to 0 processes all remaining
samples. The limit applies to metadata rows after duplicate-ID filtering, not
to the number of newly generated images. Valid existing outputs are skipped.

Direct invocation example (can also add `--dry-run` to validate on CPU):

```bash
python -B generate_omnigen2_gedit.py \
  --transformer-path /local/pretrained/OmniGen2 \
  --text-encoder /local/pretrained/Qwen2.5-VL-3B-Instruct \
  --vae-model /local/pretrained/FLUX.1-dev \
  --limit 100 --cpu-offload
```

Alternatively pass `--pretrained-root /local/pretrained` when the three model
directories have the names shown above. Explicit paths override the common
root and optional settings fallback. The VAE root must contain `vae/config.json`;
the transformer directory must contain its own `config.json` beside the weights.

The Slurm allocation is normal / mig2-35, 8 CPUs, 64GB host RAM and 72 hours.
Model CPU offload is enabled. It uses the existing OmniGen2 environment and
stores Slurm logs in `logs/omnigen2_gedit_JOBID.{out,err}`. Timing totals are
printed in the log, not saved into the edited-image directory.

Resume skips existing PNGs only after format, dimensions and decoding checks.
PNG writes use a temporary file, fsync and atomic replacement. Configuration
identity/locks live under `outputs/inference_gedit`, outside the edited-image
folder. A different seed/model/config needs a different `--output-root`; the
script rejects mixing settings in an existing registered output folder. Do not
run two jobs against the same output root. `--limit 5` can be appended to the
sbatch command for an initial subset. `--overwrite` regenerates the selected
images with the same registered configuration.

CPU/mocked tests are in `test_omnigen2_gedit.py`; actual GPU inference and peak
memory have not been validated in this workspace.

## Fireward checkpoint inference on GEdit

The standalone `generate_omnigen2_gedit_fireward.py` and
`run_generate_omnigen2_gedit_fireward.slurm` add the requested `_fireward`
suffix to the code filenames. They load the original pretrained components
and then the trained checkpoint. PNG filenames retain the source filename so
the benchmark evaluator can find them. Output is separate from the base model:

```text
/datastore/anhndt/Truongtq/OtherEditingBenchmark/OmniGen2-Fireward/GEdit-Bench-EN/<task_type>/<source_filename>.png
```

Edit `PRETRAINED_ROOT` (or the three individual paths), `CHECKPOINT_PATH` and
`LIMIT` in the Slurm file. The initial checkpoint example is:

```text
<h200_qwen3_rl>/outputs/job_53794/checkpoints/checkpoint-120
```

Use an actual completed checkpoint directory. The parent `checkpoints/` is
not a checkpoint: the script does not silently choose the latest step. It
prefers `checkpoint-120/lora_adapter/` and also accepts that adapter directory
directly. Legacy direct-Diffusers LoRA prefixes are normalized in memory; the
checkpoint files are not changed. If only a full `model.safetensors` exists,
the script reconstructs the adapter from the original job's `train.yml`, then
strictly restores the full model state. An explicit `--lora-config` YAML/PEFT
JSON can be passed when the original config is elsewhere. Rank and alpha are
never guessed from the current training recipe.

The loader verifies all restored LoRA keys/tensors and refuses a checkpoint
with missing/invalid adapter tensors. It never falls back to base-only
inference on a checkpoint error. Checkpoint identity is included in resume
metadata outside the edited-image folder; a different checkpoint needs a
different output root. All model parameters are frozen for inference.
After validating a PEFT export, the loader passes its underlying OmniGen2
module to the pipeline, with the injected LoRA layers still active and unmerged.
This preserves the explicit `ref_image_hidden_states` forward parameter that
the original pipeline inspects before forwarding source latents. Leaving the
generic PeftModel wrapper in place causes the pipeline to omit that argument.

```bash
mkdir -p logs
sbatch run_generate_omnigen2_gedit_fireward.slurm
# Optional subset, overriding LIMIT in the Slurm file:
sbatch run_generate_omnigen2_gedit_fireward.slurm --limit 10
```

Resource allocation, seed 20, Euler 50-step sampling, guidance 5/1.5, resizing
and image resume are the same as the pretrained GEdit generator. `LIMIT=0`
means all samples. Logs are `logs/omnigen2_gedit_fireward_JOBID.{out,err}`.
Tests in `test_omnigen2_gedit_fireward.py` use real CPU tensors and mocked PEFT
plumbing; loading the real OmniGen2/PEFT versions on the server remains untested.
