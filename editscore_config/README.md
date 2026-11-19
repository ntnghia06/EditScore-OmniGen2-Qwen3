# Cấu hình OmniGen2 Flow-GRPO gốc của EditScore

Thư mục này mô tả thiết lập training OmniGen2 Flow-GRPO của EditScore.
Các file YAML và shell được sao chép nguyên byte từ:

- Repo: https://github.com/VectorSpaceLab/EditScore
- Commit: `4609c5d2ebb62fdebf665d3c924686d896ef1f74`
- Thư mục nguồn: `examples/OmniGen2-RL/`

`manifest.json` ghi đường dẫn nguồn, kích thước và SHA256 của từng file.
Đây là thiết lập lấy từ source tại commit trên; không phải khẳng định tất cả
biến thể đều là cấu hình dùng cho mọi kết quả trong bài báo.

## Các file

```text
editscore_config/
  README.md
  manifest.json
  options/
    omnigen2_edit_rl_single_machine_editscore7b.yml
    omnigen2_edit_rl_4machine_editscore7b_avg4.yml
    omnigen2_edit_rl_4machine_editscore7b_avg8.yml
  reward_server/server_configs/
    editscore_7B.yml
    editscore_7B_avg4.yml
    editscore_7B_avg8.yml
  data_configs/train/example/
    train.yml
    edit/all.yml
  scripts/train/
    omnigen2_edit_rl_single_machine_editscore7b.sh
    omnigen2_edit_rl_4machine_editscore7b_avg4.sh
    omnigen2_edit_rl_4machine_editscore7b_avg8.sh
```

## Thiết lập policy và sampling chung

| Thành phần | Giá trị trong cấu hình gốc |
|---|---|
| Policy | OmniGen2 transformer |
| Text encoder | Qwen/Qwen2.5-VL-3B-Instruct |
| VAE | black-forest-labs/FLUX.1-dev, component VAE |
| Trọng số transformer | pretrained_models/OmniGen2/transformer/pytorch_model.bin |
| Finetune | LoRA rank 32, alpha 64, dropout 0 |
| Precision | BF16; gradient checkpointing bật |
| Learning rate | 4e-4, không scale LR |
| Scheduler | timm_constant_with_warmup; warmup_t = 0 |
| AdamW | betas = (0.9, 0.95), weight decay = 0.01, eps = 1e-8 |
| Gradient norm tối đa | 1 |
| EMA | 0 |
| Giới hạn diện tích source/output | 262144 pixel, tương đương diện tích 512² |
| Cạnh tối đa | 2048 |
| Rollout mỗi prompt | 12 |
| Sampling | 20 bước, sigma_coef = 0.7 |
| Guidance | text = 4, image = 2; CFG range = [0.0, 0.6] |
| Transition dùng train mỗi ảnh | ceil(20 × 0.6) = 12 |
| Reuse | 1 |
| PPO clipping | [1e-4, 5e-4], bất đối xứng quanh ratio = 1 |
| Advantage clipping | [-5, 5] |
| KL coefficient | 0.04 |
| Policy loss reweighting | bật |
| Max optimizer steps | 1000; không phải 1000 epoch |

Giới hạn diện tích không đồng nghĩa ép mọi ảnh thành 512×512. Xử lý kích thước
được thực hiện bởi dataset/pipeline của OmniGen2.

## Ba biến thể thực nghiệm

| Tham số | Single machine | 4 machine Avg@4 | 4 machine Avg@8 |
|---|---:|---:|---:|
| GPU policy theo batch cấu hình | 8 | 32 | 32 |
| Global rollout batch | 144 | 576 | 576 |
| Batch mỗi GPU | 18 | 18 | 18 |
| Prompt mỗi vòng sampling, toàn bộ GPU | 12 | 48 | 48 |
| Rollout mỗi prompt | 12 | 12 | 12 |
| Forward batch mỗi GPU | 9 | 9 | 9 |
| Gradient accumulation trong YAML | 1 | 1 | 1 |
| Optimizer update mỗi vòng sampling | 2 | 2 | 2 |
| Ảnh mỗi optimizer update, toàn bộ GPU | 72 | 288 | 288 |
| Reward num_pass | 1 | 4 | 8 |

Số GPU ở bảng là số GPU policy suy từ global_batch_size / batch_size, cũng
phù hợp cách launch 1 hoặc 4 máy, mỗi máy 8 GPU trong hướng dẫn gốc.
Tài nguyên cho reward server là riêng, không được cộng trong cột GPU policy.
Shell gốc tự đếm thiết bị bằng `nvidia-smi -L`; cần kiểm tra lại việc đếm này
trên node có MIG, không dùng nguyên shell để suy ra phân bổ Slurm.

Với 1000 optimizer update và 2 update/vòng sampling:

- Single machine: 500 vòng, 6000 lượt điều kiện, 72000 ảnh rollout.
- 4 machine: 500 vòng, 24000 lượt điều kiện, 288000 ảnh rollout.

Đây là số lượt theo batch recipe, không phải số sample duy nhất được bảo đảm.
Độ lớn dataset, sampler và việc lặp dữ liệu quyết định số điều kiện khác nhau.

## Reward EditScore gốc

- Backbone `Qwen/Qwen2.5-VL-7B-Instruct`, adapter `EditScore/EditScore-7B`.
- Inference reward dùng backend `qwen25vl_vllm`.
- `score_range: 25`, `temperature: 0.7`, `seed: 42`.
- Tensor parallel size 1, max_model_len 1536, max_num_seqs 64,
  max_num_batched_tokens 98304.
- Chấm source, edited image và instruction thành SC/PQ bằng sinh văn bản.
- Trong batch_evaluate, lấy min của danh sách điểm từng tiêu chí và quy về
  thang 0–10 bằng chia cho score_range / 10.
- Overall: `O = sqrt(SC * PQ)`; reward đưa vào RL: `r = O / 10`.
- Avg@4/Avg@8 là số lượt chấm reward, không phải số rollout mỗi prompt.
- Với nhiều pass, mặc định `average_last` lấy trung bình O của từng pass.
  Tùy chọn `average_first` tính O từ SC/PQ đã lấy trung bình qua các pass;
  ba config reward ở đây không truyền tùy chọn này nên dùng mặc định.
- Reward model chỉ infer, không được tối ưu bởi policy training.

GRPO chuẩn hóa reward theo nhóm các ảnh cùng prompt để tính advantage.
Policy dùng PPO clipped surrogate loss, có timestep reweighting, cộng KL với
reference policy. Với LoRA, reference được lấy khi tắt adapter. Code nằm ở
`examples/OmniGen2-RL/omnigen2/grpo/utils.py` và
`examples/OmniGen2-RL/train.py` trong repo đầy đủ.

## Dữ liệu và môi trường cần chuẩn bị

Hướng dẫn repo gốc dùng `EditScore/EditScore-RL-Data`, chuyển đường dẫn ảnh
thành tuyệt đối và dùng `scripts/data/extract_9_tasks.py` để bỏ text change và
portrait beautification. Điền JSONL đã xử lý vào
`examples/OmniGen2-RL/data_configs/train/example/edit/all.yml`.
File trong bản chụp giữ đường dẫn ví dụ của upstream. Đặt đường dẫn JSONL
thực tế trong data_configs/train/example/edit/all.yml trước khi chạy.

Cần môi trường policy, dependency của repo và môi trường/backend vLLM cho
reward server; pretrained transformer OmniGen2 dạng bin, Qwen2.5-VL-3B,
FLUX.1-dev VAE, Qwen2.5-VL-7B và adapter EditScore-7B. Không có trọng số nào
được lưu trong thư mục này. Cần đặt IP/port reward server trong config theo
máy thực tế; các địa chỉ 127.0.0.x là ví dụ upstream.

## Cách dùng các file gốc

Đây là bản lưu cấu hình, không phải một repo training độc lập. Các đường dẫn
trong YAML/shell được giữ nguyên và dựa trên working directory
`examples/OmniGen2-RL`. Để chạy recipe gốc, dùng các file tương ứng đã có sẵn
trong repo đầy đủ; không chạy shell ngay trong thư mục bản chụp này vì nó sẽ
tìm train.py theo vị trí tương đối của script.

Ví dụ khi đã được cấp đủ GPU, chuẩn bị dữ liệu/pretrained và reward server:

```bash
cd /path/to/EditScore/examples/OmniGen2-RL

# Recipe gốc: 1 máy, 8 GPU policy.
bash scripts/train/omnigen2_edit_rl_single_machine_editscore7b.sh \
    --world_size=1 --rank=0 --master_addr=127.0.0.1

# Recipe 4 máy: chạy trên từng máy với rank tương ứng 0,1,2,3.
# MASTER_ADDR phải là địa chỉ máy rank 0 có thể truy cập từ các máy khác.
bash scripts/train/omnigen2_edit_rl_4machine_editscore7b_avg4.sh \
    --world_size=4 --rank=0 --master_addr=MASTER_ADDR
```

Các shell này không phải Slurm batch script. Lệnh trên phải chạy trong tài
nguyên đã được scheduler cấp, không chạy training trên login node. FSDP gốc
dùng HYBRID_SHARD và SHARDED_STATE_DICT. Bản chụp không bổ sung Slurm hoặc
thay đổi mô hình phân tán gốc.
