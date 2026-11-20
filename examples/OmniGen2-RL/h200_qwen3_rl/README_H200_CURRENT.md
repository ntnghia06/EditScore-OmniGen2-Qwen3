# Setting training OmniGen2 + Qwen3 reward trên một H200

Cập nhật: 2026-11-20.

Tài liệu mô tả YAML 200 step người dùng đã cung cấp và cấu hình thực tế được
ghi trong log `omnigen2_qwen3_64127.err`. Đây là bộ chạy một GPU tại
`h200_qwen3_rl/`, không phải bộ distributed trong `editscore_config/`.

`train.yml` trong bản repo local hiện vẫn là template cũ: 1000 step, LR 4e-4,
accumulation 1 và 12 update/sampling. Vì vậy, để tái lập run hiện tại cần dùng
cấu hình 200 step dưới đây; không suy setting thực nghiệm từ template cũ.
Tài liệu này không tự thay đổi YAML hoặc job đang chạy trên server.

## Tài nguyên và môi trường

| Thành phần | Setting |
|---|---|
| Policy process | 1 |
| GPU | 1 H200 full, GRES gpu:7g.141gb:1 |
| GPU policy/reward | Cùng một GPU; hai process dùng hai môi trường riêng |
| Slurm account | uit |
| Partition / QoS | full / mig7-141 |
| Node / task | 1 / 1 |
| CPU / RAM | 16 CPU / 128 GB RAM hệ thống |
| Time limit | 72 giờ; đây là giới hạn job, không phải thời gian train dự kiến |
| Policy precision | BF16 |
| Policy environment | /datastore/truongtq/miniconda3/envs/omnigen2-rl |
| Reward environment | /datastore/truongtq/miniconda3/envs/qwen_vl |
| Conda initialization | /datastore/truongtq/miniconda3/etc/profile.d/conda.sh |

Thông số Slurm lấy từ run_train.slurm; cấu hình một process và batch được
xác nhận trong log job 64127. Log không đo đầy đủ peak VRAM hoặc RAM.

Thư mục chạy trên server:

```text
/datastore/anhndt/Truongtq/TrainingEditingModel/EditScore-Qwen3-new/examples/OmniGen2-RL/h200_qwen3_rl
```

## Policy và phần được cập nhật

- Policy: OmniGen2 transformer.
- Text encoder: Qwen/Qwen2.5-VL-3B-Instruct, đóng băng toàn bộ, gồm vision encoder.
- VAE: component VAE của black-forest-labs/FLUX.1-dev, đóng băng.
- Chỉ train LoRA transformer, rank 32, alpha 64, dropout 0.
- Target modules: to_q, to_k, to_v, to_out.0.
- Tham số LoRA trainable: 20.428.800 theo kiến trúc/target modules hiện tại.
- Backbone transformer dùng BF16; tham số LoRA trainable dùng FP32 trong trainer.
- Bật gradient checkpointing; EMA 0; allow_tf32 false.

Kiến trúc transformer: patch_size 2, in_channels 16, hidden_size 2520,
num_layers 32, num_refiner_layers 2, num_attention_heads 21, num_kv_heads 7,
multiple_of 256, norm_eps 1e-5, axes_dim_rope [40,40,40],
axes_lens [10000,10000,10000], text_feat_dim 2048, timestep_scale 1000.

Pretrained được resolve từ settings.json rồi ghi đường dẫn tuyệt đối vào
outputs/job_JOBID/train.yml. Log 64127 xác nhận text encoder và VAE đọc tại:

```text
h200_qwen3_rl/outputs/pretrained/Qwen2.5-VL-3B-Instruct
h200_qwen3_rl/outputs/pretrained/FLUX.1-dev
```

Transformer phải trỏ tới file OmniGen2/transformer/pytorch_model.bin đã extract.
Không suy đường dẫn transformer chỉ từ hai đường dẫn đã thấy trong log.

## Dữ liệu

```text
metadata:
/datastore/anhndt/Truongtq/TrainingEditingModel/RunFLUX/unique_metadata.jsonl

data_root:
/datastore/shared/datasets/EditReward-Data/extracted_data
```

Log xác nhận dataset sau chuẩn bị có 8132 điều kiện source + instruction.
prepare_run.py kiểm tra ảnh, loại trùng theo source path + instruction và tạo
training_pairs.jsonl dùng cho OmniGen2. Không cần nhãn chosen/rejected hoặc
ảnh edited có sẵn; policy sinh rollout mới trong quá trình RL.

| Setting xử lý dữ liệu | Giá trị |
|---|---:|
| maximum_text_tokens | 888 |
| use_chat_template | true |
| prompt_dropout_prob | 0 |
| ref_img_dropout_prob | 0 |
| max_output_pixels | 262144 |
| max_input_pixels | [262144,262144,262144,262144] |
| max_side_length | 2048 |
| dataloader_num_workers | 4 |

262144 là giới hạn diện tích bằng 512² pixel, không có nghĩa ép mọi ảnh thành
512×512. Dataset giữ tỉ lệ theo logic xử lý ảnh OmniGen2.

## Batch và số update

```yaml
train:
  global_batch_size: 12
  batch_size: 12
  gradient_accumulation_steps: 12
  max_train_steps: 200
  rl:
    num_unique_prompts_per_sampling: 1
    num_images_per_prompt: 12
    batch_size_per_forward: 1
    num_update_steps_per_sampling: 1
    reuse_samples_nums: 1
```

Một vòng sampling lấy 1 source/instruction, sinh 12 rollout với nhiễu ngẫu
nhiên khác nhau, chấm reward rồi tích lũy gradient để cập nhật policy một lần.

- Điều kiện batch: 12 / (1 × 12) = 1 optimizer update/sampling.
- Mỗi ảnh train trên ceil(20 × 0.6) = 12 transition.
- Accelerator accumulation nội bộ: 12 × 12 = 144 lần backward transition
  cho mỗi optimizer update, tương ứng 12 ảnh, không phải 144 ảnh độc lập.
- 200 update tương ứng 200 lượt điều kiện và 2400 rollout, chưa phải một epoch.
- 200 / 8132 tương đương khoảng 2,46% số lượt điều kiện của một lượt dataset.
  Số điều kiện duy nhất cần kiểm tra sampler/ID nếu cần thống kê chính xác.

Dòng log Total train batch size = 144 được tính bằng công thức log chung của
trainer. Không dùng con số đó để kết luận effective batch là 144 ảnh; trong
cấu hình này một optimizer update tích lũy 12 ảnh.

## Optimizer và sampling

| Setting | Giá trị |
|---|---|
| learning_rate | 5e-5 |
| scale_lr | false |
| lr_scheduler | timm_constant_with_warmup |
| warmup_t / warmup_lr_init | 20 update / 1e-7 |
| warmup_prefix / t_in_epochs | true / false |
| AdamW betas | (0.9, 0.95) |
| AdamW weight_decay / epsilon | 0.01 / 1e-8 |
| use_8bit_adam | false |
| max_grad_norm | 1 |
| set_grads_to_none | true |
| seed | 2233 |
| device_specific_seed / workder_specific_seed | true / true |
| SDE sampling steps | 20 |
| sigma_coef | 0.7 |
| text_guidance_scale / image_guidance_scale | 4 / 2 |
| cfg_range | [0.0, 0.6] |
| max_sequence_length | 1024 |
| negative_prompt | chuỗi rỗng; use_ori_neg_prompt_template true |
| train_timesteps_fraction | 0.6 |
| time_shift_base_res | 168 |
| transport | snr_type lognorm, do_shift true, dynamic_time_shift true |

## Qwen3 reward

```text
reward_project:
/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/Qwen3-VL-Embedding

reward_pretrained:
/datastore/truongtq/TrainingRewardModel/Models/Qwen3-VL-Reranker-8B

reward_checkpoint:
/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/Qwen3-VL-Embedding/output_training/Stage2_R0RelKL10kDecay98CosineOnceBatch4Eval5
```

Reward model ở eval mode, đóng băng backbone, adapter và metric heads.
Một nhóm 12 rollout chia thành 6 cặp A/B. Mỗi lần forward xử lý ba conversation:
source + instruction + A, source + instruction + B, source + instruction.

Các head cung cấp scores_A, scores_B và relevance_probs, mỗi tensor [1,6].
Sáu metric: aesthetic, fidelity, realism, background_consistency,
foreground_consistency, structure_consistency; thứ tự lấy từ module checkpoint.

Trong mỗi nhóm, tính relevance trung bình qua 6 cặp, chọn chung các metric
có relevance >= 0.5. Reward mỗi ảnh là trung bình cộng raw score trên bộ metric
được chọn. Điểm không giới hạn về 0–1/0–10, không chia 10, không dùng trung
bình nhân hoặc lấy margin A/B. Tie/three-way probability không làm reward RL.
Không có metric đạt ngưỡng thì báo lỗi theo empty_metric_policy: error.
Reward max_pixels 262144, min_pixels không override.

## Loss Flow-GRPO

Reward được chuẩn hóa riêng trong nhóm 12 rollout cùng điều kiện:

```text
A_i = (reward_i - group_mean) / (group_std + 1e-8)
```

Sau đó clip advantage về [-5,5]. Policy loss là PPO clipped surrogate với
ratio current/rollout policy; clip_range [1e-4,5e-4]. Áp dụng timestep
reweighting khi policy_loss_reweighting true. Loss tổng cộng thêm
0.04 × KL loss với reference pretrained, được lấy bằng cách tắt LoRA.
apply_cfg_in_training true. Reward model không nhận gradient.

Code: train_h200.py và ../omnigen2/grpo/utils.py tính policy loss, advantage,
transition log-probability và KL. KL reference khác thống kê approx_kl giữa
current policy và rollout policy.

## Launch, log và checkpoint

- train_visualization_interval 20; num_train_visualization_samples 1.
- logger.log_with []; theo dõi bằng Slurm stdout/stderr và log reward.
- checkpointing_steps 20, checkpoints_total_limit 3.
- checkpoint_format lora, resume_from_checkpoint null.
- LoRA, optimizer, scheduler và RNG được lưu; không sao chép frozen transformer.
- Checkpoint ghi vào .checkpoint-N.incomplete, chỉ publish checkpoint-N khi
  hoàn tất. Có checkpoint_complete.json và lora_adapter/ để infer.
- Lưu trạng thái không bổ sung cơ chế resume mid-rollout chính xác từng bit.

Các file chính trong bundle: settings.json, train.yml, prepare_run.py,
train_h200.py, compact_checkpoint.py, reward_server.py, reward_runtime.py,
run_train.slurm. data.yml và reward_server.yml được tự sinh trong từng run.

Trước khi submit, đảm bảo train.yml trên server đúng setting 200 step ở tài
liệu này, đặc biệt accumulation 12, num_update_steps_per_sampling 1 và LR 5e-5.
Các đường dẫn data/pretrained/reward đặt trong settings.json. Sau đó:

```bash
cd /datastore/anhndt/Truongtq/TrainingEditingModel/EditScore-Qwen3-new/examples/OmniGen2-RL/h200_qwen3_rl
mkdir -p logs
sbatch run_train.slurm
```

Slurm tự bật reward server trên cùng GPU, chờ ready, chạy policy và dọn server
khi job kết thúc. Mỗi job có outputs/job_JOBID/ chứa config và script chụp lúc
job bắt đầu. Sửa file khi job còn queued vẫn ảnh hưởng snapshot lúc job khởi
động; sửa sau khi snapshot đã được tạo không đổi các file đã chụp.

```text
logs/omnigen2_qwen3_JOBID.out
logs/omnigen2_qwen3_JOBID.err
logs/reward_JOBID.log
outputs/job_JOBID/train.yml
outputs/job_JOBID/data.yml
outputs/job_JOBID/reward_server.yml
outputs/job_JOBID/training_pairs.jsonl
outputs/job_JOBID/checkpoints/checkpoint-N/lora_adapter/
```

Job 64127 xác nhận checkpoint LoRA lưu thành công đến step 160 trong bản log
đã cung cấp; bản log dừng ở step 176/200, chưa xác nhận hoàn tất 200 step.
Thời gian quan sát khoảng 129 giây/update, xấp xỉ 7,2 giờ cho 200 update nếu
tốc độ giữ nguyên. Không coi đây là kết quả benchmark chất lượng.

Các trường advantages_min/max và advantages_std trong logger hiện có hạn
chế do microbatch 1: min/max bị lấy trung bình, std một phần tử ra NaN.
Không nhầm với NaN loss/gradient. Đánh giá chất lượng bằng tập ảnh cố định,
cùng seed và evaluator độc lập; reward giữa các instruction không cùng độ khó.
