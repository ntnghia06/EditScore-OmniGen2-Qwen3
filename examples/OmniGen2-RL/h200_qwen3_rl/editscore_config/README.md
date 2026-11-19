# OmniGen2 Flow-GRPO: setting EditScore, reward Qwen3 và data EditReward

Bộ chạy này giữ hyperparameter policy của EditScore gốc, sử dụng reward
Qwen3 custom và dữ liệu trong `../settings.json` của bạn.

Nguồn recipe: VectorSpaceLab/EditScore, commit
`4609c5d2ebb62fdebf665d3c924686d896ef1f74`.

## File và setting

- `train.yml`: recipe single-machine, 8 GPU policy.
- `train_4machine.yml`: recipe 4 máy × 8 GPU policy.
- `settings.json`: kế thừa `../settings.json`, bổ sung địa chỉ/port reward.
- `prepare_profile.py`: kiểm tra recipe, chuẩn bị data và đóng băng run.
- `start_reward.sh`: chạy Qwen3 trên một GPU reward riêng.
- `launch_policy.sh`: launch trainer gốc bằng Accelerate FSDP.
- `upstream_reference/`: bản lưu 11 file gốc và manifest SHA256; không dùng
  reward/data config trong đó để chạy bộ này.

| Tham số | train.yml | train_4machine.yml |
|---|---:|---:|
| GPU policy | 8 | 32 |
| global_batch_size | 144 | 576 |
| batch_size mỗi GPU | 18 | 18 |
| num_unique_prompts_per_sampling | 12 | 48 |
| num_images_per_prompt | 12 | 12 |
| batch_size_per_forward mỗi GPU | 9 | 9 |
| gradient_accumulation_steps trong YAML | 1 | 1 |
| num_update_steps_per_sampling | 2 | 2 |
| Ảnh đóng góp mỗi optimizer update, toàn bộ GPU | 72 | 288 |
| max_train_steps | 1000 | 1000 |
| learning_rate | 4e-4 | 4e-4 |
| warmup_t | 0 | 0 |

Giữ nguyên phần train, transport, val, logger, seed, kiến trúc và xử lý ảnh/text
của upstream tương ứng. Khác tên run và đường dẫn data/reward; đường dẫn
pretrained được resolve theo settings thực tế. Recipe 32 GPU lấy từ cấu hình
policy Avg@4 gốc; Avg@4 là cách chấm EditScore, không áp dụng cho Qwen3.

Setting chung:

- LoRA rank 32, alpha 64, dropout 0; BF16, gradient checkpointing, EMA 0.
- AdamW betas (0.9, 0.95), weight decay 0.01, eps 1e-8; max grad norm 1.
- Scheduler timm_constant_with_warmup, không scale LR.
- Sampling 20 bước, sigma_coef 0.7, text guidance 4, image guidance 2,
  CFG range [0.0, 0.6], train_timesteps_fraction 0.6, reuse 1.
- PPO clip [1e-4, 5e-4], advantage clip [-5, 5], KL weight 0.04;
  policy_loss_reweighting bật. Không sửa công thức loss.
- Max input/output area 262144 pixel, max side 2048; maximum_text_tokens 888,
  RL max_sequence_length 1024. Giới hạn diện tích không ép mọi ảnh thành vuông.
- Logger WandB + TensorBoard; checkpoint mỗi 50 update, không giới hạn số
  checkpoint theo YAML gốc. Cần cài backend và cấu hình WandB.
- Checkpoint Accelerate/FSDP gốc, không dùng compact checkpoint một GPU.
  Trainer gốc còn lưu text encoder/tokenizer khi bắt đầu run.

1000 update với 2 update/vòng tương ứng 500 vòng sampling: recipe 8 GPU có
6000 lượt điều kiện và 72000 rollout; recipe 32 GPU có 24000 lượt điều kiện
và 288000 rollout. Số điều kiện duy nhất phụ thuộc dataset/sampler.

## Reward và data

Kế thừa đường dẫn đang có trong `../settings.json`:

```text
metadata:
/datastore/anhndt/Truongtq/TrainingEditingModel/RunFLUX/unique_metadata.jsonl
data_root:
/datastore/shared/datasets/EditReward-Data/extracted_data
reward_pretrained:
/datastore/truongtq/TrainingRewardModel/Models/Qwen3-VL-Reranker-8B
reward_checkpoint:
/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/Qwen3-VL-Embedding/output_training/Stage2_R0RelKL10kDecay98CosineOnceBatch4Eval5
```

Qwen3 đóng băng, infer theo cặp A/B. Chọn chung các metric có relevance trung
bình nhóm >= 0.5; reward mỗi ảnh là trung bình cộng raw score của các metric
được chọn. Không chia 10, không lấy trung bình nhân, không dùng margin A/B.
Không metric nào được chọn thì theo empty_metric_policy, mặc định error.

Rank 0 gửi toàn bộ global batch tới một server Qwen3: 72 pair-forward/vòng
cho recipe 8 GPU hoặc 288 cho recipe 32 GPU. Giới hạn timeout của reward
client gốc vẫn áp dụng. GPU reward là tài nguyên bổ sung ngoài GPU policy.

## Chuẩn bị run

Thư mục server:
`/datastore/anhndt/Truongtq/TrainingEditingModel/EditScore-Qwen3-new/examples/OmniGen2-RL/h200_qwen3_rl/editscore_config`.

Trước khi prepare, kiểm tra pretrained trong `../settings.json`. Nếu reward
khác máy rank 0, đặt `reward_host` trong settings.json của thư mục này thành
IP/hostname máy reward; reward_bind_host giữ 0.0.0.0, chọn port còn trống.
Nếu cùng máy, có thể giữ reward_host 127.0.0.1.

```bash
source /datastore/truongtq/miniconda3/etc/profile.d/conda.sh
conda activate /datastore/truongtq/miniconda3/envs/omnigen2-rl
cd /datastore/anhndt/Truongtq/TrainingEditingModel/EditScore-Qwen3-new/examples/OmniGen2-RL/h200_qwen3_rl/editscore_config

RUN="$PWD/outputs/editscore_qwen3_$(date +%Y%m%d_%H%M%S)"
python prepare_profile.py --config train.yml --run-dir "$RUN"
echo "$RUN"
```

Để dùng 32 GPU, thay bằng `--config train_4machine.yml`. Mỗi run dùng thư mục
mới. Giữ lại đường dẫn RUN, dùng cùng run trên filesystem chung giữa mọi máy.
Config, data và trainer/reward code được đóng băng trong run.

## Launch trong tài nguyên GPU đã được cấp

Không chạy training trên login node. Các shell không tự xin GPU hoặc đặt QoS.
Không dùng chung GPU policy và reward trong bộ chạy này.

Trên máy/GPU reward, giữ server chạy và chờ log Reward ready:

```bash
RUN=/absolute/path/to/prepared/run
CUDA_VISIBLE_DEVICES=0 bash start_reward.sh "$RUN" > "$RUN/reward.log" 2>&1
```

Script reward dùng môi trường `/datastore/truongtq/miniconda3/envs/qwen_vl`.
CUDA_VISIBLE_DEVICES phải theo phân bổ thực tế của scheduler.

Recipe 8 GPU, trên máy policy được cấp đúng 8 GPU:

```bash
RUN=/absolute/path/to/prepared/run
bash launch_policy.sh "$RUN" 0 127.0.0.1
```

Recipe 32 GPU, chạy trên cả 4 máy với cùng RUN; MASTER_ADDR là IP máy policy
rank 0 mà các máy khác truy cập được:

```bash
bash launch_policy.sh "$RUN" 0 MASTER_ADDR  # máy 0
bash launch_policy.sh "$RUN" 1 MASTER_ADDR  # máy 1
bash launch_policy.sh "$RUN" 2 MASTER_ADDR  # máy 2
bash launch_policy.sh "$RUN" 3 MASTER_ADDR  # máy 3
```

Launcher dùng HYBRID_SHARD, TRANSFORMER_BASED_WRAP, OmniGen2TransformerBlock,
SHARDED_STATE_DICT, use_orig_params True, sync_module_states True như upstream.
Không đếm MIG bằng nvidia-smi để quyết định world size. Trainer lấy từ train.py
gốc và chỉ thêm guard dừng theo max_train_steps ở đầu vòng dataloader.

Output trainer: `RUN/experiments/<name trong train.yml>/checkpoint-N/`.
Đây là checkpoint FSDP sharded, không giả định có lora_adapter như helper một GPU.

Đã kiểm tra recipe khớp upstream, chuẩn bị run bằng fixture CPU và cú pháp
shell. Chưa chạy thực nghiệm 8/32 GPU, chưa đo VRAM/thời gian trên cluster.
