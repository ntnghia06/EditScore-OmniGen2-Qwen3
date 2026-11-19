# OmniGen2 Flow-GRPO: setting EditScore, reward Qwen3 và data EditReward

## Chạy từ một lệnh

Chỉnh đường dẫn và tài nguyên trong **settings.json của thư mục này**, rồi:

```bash
cd /datastore/anhndt/Truongtq/TrainingEditingModel/EditScore-Qwen3-new/examples/OmniGen2-RL/h200_qwen3_rl/editscore_config
python3 submit_train.py
```

Lệnh trên tự gọi sbatch. Không cần bật reward server, đặt IP, export biến môi
trường hoặc tự launch từng máy. Lệnh submit chỉ dùng Python standard library;
policy và reward dùng hai môi trường Conda đã cài ở đường dẫn cấu hình.
Không tự cài dependency hoặc tải pretrained.

Sau khi allocation được cấp, job tự thực hiện:

1. Lấy hostname các máy policy và reward từ Slurm.
2. Chọn port trên máy reward và tạo config runtime.
3. Kiểm tra ghi 512 MiB, đọc/kiểm tra data và resolve pretrained.
4. Chạy Qwen3 trên GPU reward riêng, chờ ping có đúng run_id của job.
5. Launch toàn bộ máy policy bằng Accelerate FSDP.
6. Theo dõi policy/reward; dừng policy nếu reward chết; tắt reward khi train xong.

## File cần chỉnh: settings.json

| Trường | Nội dung |
|---|---|
| metadata | unique_metadata.jsonl của bạn |
| data_root | thư mục gốc chứa ảnh source |
| omnigen_transformer_bin | file OmniGen2 transformer/pytorch_model.bin đã extract |
| text_encoder | thư mục local Qwen2.5-VL-3B-Instruct |
| vae_model | thư mục local FLUX.1-dev, có vae/config.json |
| reward_project | project chứa src/models/custom_qwen3_vl_reranker.py |
| reward_pretrained | thư mục Qwen3-VL-Reranker-8B |
| reward_checkpoint | checkpoint Stage2 của bạn |
| conda_sh | file khởi tạo Conda |
| train_env | đường dẫn môi trường OmniGen2 |
| reward_env | đường dẫn môi trường Qwen3 |
| recipe | train.yml (8 GPU) hoặc train_4machine.yml (32 GPU) |
| slurm | account, partition, QoS, GRES, CPU, RAM và thời gian |

Các đường dẫn pretrained mẫu đã điền theo thư mục tải trước đây; sửa lại
nếu pretrained thực tế nằm nơi khác. Bộ tự động không đọc settings.json ở
thư mục cha. reward_host, reward_bind_host và port được chọn lại lúc job chạy.
Nếu cần đăng nhập WandB trực tuyến, chỉnh wandb_mode; mặc định offline vẫn
ghi các metric tại máy chạy.

## Tài nguyên Slurm

Bộ chạy dùng heterogeneous job: group 0 cho policy, group 1 cho reward.
Tài liệu: https://slurm.schedmd.com/heterogeneous_jobs.html

| Recipe | Policy | Reward bổ sung | Tổng GPU |
|---|---|---|---:|
| train.yml | 1 máy × 8 GPU | 1 GPU | 9 |
| train_4machine.yml | 4 máy × 8 GPU | 1 GPU | 33 |

Mỗi máy policy phải được cấp đúng 8 GPU phù hợp bộ nhớ/compute của recipe.
Không tự đổi 8 GPU thành một MIG hoặc giảm batch. GRES mặc định gpu:8 cho
policy và gpu:1 cho reward là yêu cầu generic; đặt type GRES và QoS đúng với
cluster để nhận loại GPU mong muốn. Cluster phải cho phép heterogeneous job,
đủ tài nguyên và mạng giữa các compute node. Nếu Slurm từ chối tài nguyên,
cần điều chỉnh các trường slurm theo phân bổ được cấp hoặc dùng cluster có
đủ GPU; bộ chạy không thể vượt giới hạn tài khoản.

Có thể chỉ tạo jobscript để xem trước:

```bash
python3 submit_train.py --prepare-only
```

Lệnh in đường dẫn file run_train.slurm đã sinh. Có thể tự sbatch file đó.
File được sinh bằng UTF-8/LF để tránh lỗi DOS line breaks.

## Setting train theo EditScore gốc

Nguồn: VectorSpaceLab/EditScore commit
`4609c5d2ebb62fdebf665d3c924686d896ef1f74`.

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
| Ảnh mỗi optimizer update, toàn bộ GPU | 72 | 288 |
| max_train_steps | 1000 | 1000 |
| learning_rate | 4e-4 | 4e-4 |
| warmup_t | 0 | 0 |

Giữ nguyên phần train, transport, val, logger, seed, kiến trúc và xử lý ảnh/text
của upstream tương ứng. Khác tên run, data và reward; pretrained dùng local path.
Recipe 32 GPU lấy phần policy của config Avg@4 gốc; không áp dụng Avg@4 cho Qwen3.

- LoRA rank 32, alpha 64, dropout 0; BF16; gradient checkpointing; EMA 0.
- AdamW betas (0.9, 0.95), weight decay 0.01, eps 1e-8, gradient norm tối đa 1.
- Sampling 20 bước, sigma_coef 0.7, text guidance 4, image guidance 2,
  CFG range [0.0, 0.6]; train 12 transition mỗi ảnh; reuse 1.
- PPO clip [1e-4, 5e-4], advantage clip [-5, 5], KL weight 0.04;
  policy loss reweighting bật.
- Max output area 262144 pixel; max input area [262144] × 4; max side 2048.
- maximum_text_tokens 888; RL max_sequence_length 1024.
- Checkpoint mỗi 50 optimizer update, không giới hạn retention như YAML gốc.
  Checkpoint dùng Accelerate/FSDP gốc, không dùng compact checkpoint một GPU.
  Trainer gốc còn lưu text encoder và tokenizer lúc khởi động.

Qwen3 đóng băng, chọn metric có relevance trung bình nhóm >= 0.5 rồi lấy trung
bình raw score làm reward. Không chia 10 hoặc thay bằng reward EditScore.
Rank 0 gửi global batch tới server: 72 pair-forward/vòng cho recipe 8 GPU,
288 cho 32 GPU. Timeout reward client gốc vẫn áp dụng.

## Log, output và snapshot

Mỗi lần submit tạo `outputs/submission_<timestamp>_<id>/`:

```text
run_train.slurm
snapshot/                      # settings/config/helper frozen khi submit
upstream_train.py              # trainer nguồn frozen khi submit
logs/job_<jobid>.out
logs/job_<jobid>.err
logs/reward.log
logs/policy.log
run/train.yml
run/data.yml
run/reward_server.yml
run/training_pairs.jsonl
run/experiments/<name>/checkpoint-N/
```

Sửa settings hoặc seed sau khi submit không đổi snapshot của job đã nộp.
Đây là checkpoint FSDP sharded; không giả định có lora_adapter/ như helper
một GPU. Model module vẫn được import từ repo, nên giữ source trong lúc chạy.
Trainer chỉ thêm guard dừng max_train_steps ở đầu vòng dataloader.

launch_policy.sh và start_reward.sh được auto_run.py gọi, nên cần giữ.
upstream_reference/ chỉ giữ hai YAML policy gốc và manifest SHA256; bước
prepare dùng chúng để kiểm tra hyperparameter. Đã bỏ config reward/data,
script launch upstream và biến thể Avg@8 không được bộ chạy này sử dụng.
Hai file test phục vụ kiểm chứng bộ chạy và không được đưa vào snapshot job.

Đã kiểm tra config, chuẩn bị dữ liệu bằng fixture CPU, topology 8/32 GPU,
snapshot, readiness và cleanup qua process mock. Chưa thử allocation Slurm
hoặc training nhiều GPU thực tế.
