# Chuẩn bị training trên GPU CUDA

Bản sửa giữ SFNO-SC2-L6-E128, 26 channels, 361×720 và bước dự báo 6 giờ.
Cấu hình khởi đầu dưới đây dành cho **một GPU**, batch 1 × accumulation 8.
Đây là lựa chọn thận trọng để đo trước; chưa phải cam kết về VRAM hoặc throughput.
GPU cụ thể được chọn sau khi đo VRAM và throughput bằng preflight; cấu hình không gắn với một model GPU.

## 1. Environment riêng trên máy GPU

Dùng Python 3.11 và môi trường sạch, không kế thừa torchvision/OpenCV từ môi trường khác.
Cài đúng `requirements-lock.txt`. Chọn CUDA wheel tương thích driver theo
[hướng dẫn chính thức PyTorch](https://docs.pytorch.org/get-started/locally/);
không tự đổi phiên bản torch-harmonics 0.7.4 để giải quyết cài đặt.
Nếu wheel Torch được khóa chưa có cho máy đó, dừng và kiểm chứng một environment
mới bằng toàn bộ tests trước khi thay lock.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m pip check
nvidia-smi
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 python -m pytest -q
python scripts/check_backend.py --device cuda
```

Lưu `pip freeze`, driver, GPU name và kết quả tests cùng experiment. Checkpoint
hiện lưu dependency versions, source/operator hashes, precision/determinism flags,
CUDA driver và hardware. `--deterministic` là tùy chọn cho regression, có thể chậm
hoặc gặp operation chưa hỗ trợ; seed không tự bảo đảm bitwise equivalence giữa máy.
Đặt `CUBLAS_WORKSPACE_CONFIG=:4096:8` **trước** khi khởi chạy nếu dùng deterministic CUDA.

## 2. Dữ liệu và statistics

Protocol mới: train **1995–2018**, validation **2019**, test **2020**.
Store mặc định là `data/dataset/era5_1995_2020_0p5deg_26ch.zarr`.
Store cũ kết thúc tháng 2/2019 thiếu phần held-out cần thiết. Không đổi tên store cũ
để vượt kiểm tra; giữ nguyên và tạo corpus theo contract mới. Downloader từ chối
resume với contract khác. Chưa có download corpus lớn nào được thực hiện trong phiên sửa code này.

Tensor thô 37,988 states x 27,031,680 bytes khoảng **1.027 TB**; dung lượng nén
phải đo trên dữ liệu đại diện. Dành thêm chỗ cho download tạm, statistics và nhiều
checkpoint. Không dùng RAM/GPU memory để thay cho thiếu persistent disk.

```bash
# Sample đã được tạo trong workspace này. Chỉ chạy download nếu chuyển sang máy mới chưa có sample.
python data/test_download_sample.py --output data/dataset/era5_sample_audited.zarr
python data/download_data.py --confirm-full-download
# Thư mục stats phải chưa tồn tại. Bảo quản bundle cũ dưới tên riêng trước bước này.
python scripts/compute_stats.py --time-step 1
python scripts/validate_data.py --full-scan --output runs/data_validation.json
```

Statistics v4 stream joint-channel moments bằng float64 Chan, RAM phụ thuộc một
số states và stride, không phụ thuộc số năm. Mỗi bundle lưu SHA256 từng array,
channel/unit/grid/time contract và checksum decoded training states float32.
`--full-scan` đọc toàn bộ từng split, kiểm finite và đối chiếu checksum train.
Đây là một lượt I/O có chủ đích trước khi thuê GPU lâu dài; metadata-only check
không thay thế nó. Giữ corpus bất biến sau xác minh.

Bundle được publish bằng rename sau khi hoàn tất; không ghi đè bundle đã có.
Statistics cũ hoặc checkpoint chưa có forecast contract phải được tái xác minh;
không có migration tự đoán metadata. Đổi statistics/architecture là một experiment
mới, không phải exact resume.

## 3. Preflight thực trên GPU

```bash
OMP_NUM_THREADS=1 python scripts/preflight_gpu.py \
  --device cuda --batch-size 1 --gradient-accumulation 8 \
  --updates 3 --rollout-steps 1 \
  --output runs/gpu_preflight_k1_b1.json
```

Lệnh dùng dữ liệu thật, GraphCast-style channel weighting và standardized-tendency loss,
BF16 autocast, clipping 1.0 và AdamW như train. Nó đo peak allocated/reserved VRAM
của cả updates và validation, output difference FP32/BF16, gradient norms,
zero/constant probes và RMSE từng lead. Không lưu weights này thành research checkpoint.

Kiểm tra report trước training: không non-finite, đủ VRAM trống cho runtime/I/O,
precision difference phù hợp với field scale và không có amplification bất thường
ở constant/near-constant diagnostics. `status=passed` chỉ nghĩa là smoke test chạy
được; không chứng minh skill hay physical stability.

M06 đã được tái hiện ở **full grid E128/L6 trên CPU**: constant input cho output
spatial std 2.2109. Bản sửa tách thành phần hằng trước quadrature:
`SHT(x) = SHT(x-c) + c sqrt(4π) e00`, với anchored centering để residual của trường
hằng bằng zero chính xác. Sau sửa, spatial std bằng 0; input ngẫu nhiên có output
RMS difference khoảng 2.9e−6 so với upstream trên cùng weights. Không thêm parameters
hoặc đổi epsilon. Có thêm centering/reduction và một tensor spatial tạm; cần đo peak
CUDA theo preflight. Checkpoint/config ghi `stabilize_sht_constants=True`; tắt bằng
`--no-stabilize-sht-constants` chỉ cho ablation có chủ đích, không exact resume.

Preflight chặn khi stabilized model không giữ trường hằng trong tolerance trên
backend thực. **Near-constant sensitivity vẫn còn**: IN có thể khuếch đại perturbation
1e−5 lên output O(1), ngay cả khi SHT đã giữ hằng chính xác. Phải xem activation
variance, noise sensitivity và lead errors trên real-data pilot trước long training;
không coi bản sửa SHT là bằng chứng toàn dynamics đã stable. Norm/epsilon changes
vẫn cần ablation riêng.

Sau khi B1 pass, có thể đo B2×4 hoặc B4×2 vào **report khác**. Chọn batch từ phép đo,
không tự suy một GPU nhiều VRAM phải dùng model lớn hơn. Preflight `--device cpu` dành cho
integration tests nhỏ, không mô phỏng VRAM CUDA. Với K2 cần preflight riêng có
`--rollout-steps 2 --gradient-checkpointing`.

## 4. Baseline đầu tiên với ngân sách hữu hạn

```bash
OMP_NUM_THREADS=1 python scripts/train.py \
  --device cuda --amp-dtype bfloat16 \
  --run-dir runs/sfno_k1 --stage-name sfno_1step \
  --epochs 25 --scheduler-epochs 25 --warmup-epochs 3 \
  --batch-size 1 --gradient-accumulation 8 \
  --rollout-steps 1 --patience 5 \
  --validation-batch-size 1 \
  --validation-num-workers 0 \
  --num-workers 4 --prefetch-factor 2 --no-persistent-workers \
  --no-gradient-checkpointing

# K=2: fine-tune từ best K=1, LR thấp hơn
OMP_NUM_THREADS=1 python scripts/train.py \
  --device cuda --amp-dtype bfloat16 \
  --run-dir runs/sfno_k2 --stage-name sfno_2step \
  --init-checkpoint runs/sfno_k1/checkpoints/best.pt \
  --epochs 10 --scheduler-epochs 10 --warmup-epochs 1 \
  --learning-rate 1e-4 --rollout-steps 2 \
  --batch-size 1 --gradient-accumulation 8 --gradient-checkpointing \
  --validation-batch-size 1 --validation-num-workers 0 \
  --num-workers 4 --prefetch-factor 2 --no-persistent-workers

# K=4: tiếp tục fine-tune
OMP_NUM_THREADS=1 python scripts/train.py \
  --device cuda --amp-dtype bfloat16 \
  --run-dir runs/sfno_k4 --stage-name sfno_4step \
  --init-checkpoint runs/sfno_k2/checkpoints/best.pt \
  --epochs 7 --scheduler-epochs 7 --warmup-epochs 0 \
  --learning-rate 5e-5 --rollout-steps 4 \
  --batch-size 1 --gradient-accumulation 8 --gradient-checkpointing \
  --validation-batch-size 1 --validation-num-workers 0 \
  --num-workers 4 --prefetch-factor 2 --no-persistent-workers

# K=8: fine-tune cuối cho cửa sổ 48 giờ
OMP_NUM_THREADS=1 python scripts/train.py \
  --device cuda --amp-dtype bfloat16 \
  --run-dir runs/sfno_k8 --stage-name sfno_8step \
  --init-checkpoint runs/sfno_k4/checkpoints/best.pt \
  --epochs 5 --scheduler-epochs 5 --warmup-epochs 0 \
  --learning-rate 2.5e-5 --rollout-steps 8 \
  --batch-size 1 --gradient-accumulation 8 --gradient-checkpointing \
  --validation-batch-size 1 --validation-num-workers 0 \
  --num-workers 4 --prefetch-factor 2 --no-persistent-workers
```

Mặc định validation quét tuần tự mọi sliding window hợp lệ. Ở stage K=1, train và validation đều tính đúng một lead; các stage K=2, K=4 và K=8 cũng tuân theo cùng quy tắc. Checkpoint được chọn bằng đúng objective của stage hiện tại. Validation/evaluation chỉ đọc một target lead tại một thời điểm:
CPU/GPU truth buffers không tăng tuyến tính với K. I/O hiện đọc tuần tự ở main process;
profile trước khi thêm asynchronous prefetch. Training BPTT vẫn tăng chi phí theo K,
không bị detach để tiết kiệm bộ nhớ.

Trước baseline 25 epochs, nên chạy pilot cùng cấu hình trên ít training samples
vào run directory riêng để xem actual loss/gradient scales. Pilot không phải held-out
skill evidence và không được resume thành full-data run: `--max-samples` là một phần
của exact-resume signature. Dùng `--init-checkpoint` nếu chủ đích chuyển stage.

FP16+GradScaler training bị từ chối trong profile SFNO complex này. Phép thử local
cho thấy unscale complex gradients không được hỗ trợ trên CPU; chưa có CUDA FP16
verification. Dùng BF16 hoặc `--no-amp` thay vì bỏ guards để thử một long run.

Nếu non-finite loss/gradient xuất hiện, training dừng trước optimizer update;
checkpoint có non-finite parameters hoặc Adam state bị từ chối. Khôi phục từ
last-good checkpoint sau khi tìm nguyên nhân. **Không tắt finite guards.**

Exact resume: lặp lại nguyên lệnh và thêm
`--resume runs/sfno_k1/checkpoints/last.pt`. Không đổi batch, sample panel,
warmup, dữ liệu, runtime hoặc scheduler. Ledger phía sau checkpoint cũ được archive
trước khi dựng lại history để tránh trộn trajectories.

Train objective tại stage K là:

`L_K = sum(k=1..K, gamma^(k-1) L_k) / sum(k=1..K, gamma^(k-1))`,

với mặc định `gamma=1`. Mỗi `L_k` là MSE của sai số trạng thái tại lead `k`,
chuẩn hóa bằng `time_diff_std`, có spherical area weights và variable/pressure
weights. Tại `K=1`, nó đúng bằng MSE của standardized tendency dự báo một bước.
Validation và test dùng nguyên `L_K`, cùng K và gamma của checkpoint.

## 5. Evaluation và các việc nghiên cứu còn mở

```bash
OMP_NUM_THREADS=1 python scripts/evaluate.py \
  --checkpoint runs/sfno_k8/checkpoints/best.pt \
  --device cuda --split test --num-workers 0 \
  --output-dir runs/sfno_k8/test_sliding_windows
```

Evaluation khóa cadence/grid/units, từ chối overlap train/validation của checkpoint,
và ghi checkpoint hash, data identity, initializations, statistics, precision và overrides.
So RMSE với persistence và training climatology trên cùng starts; xem physical bias,
variance ratio, ACC valid counts từng channel. ACC của climatology là undefined vì
forecast anomaly bằng zero, không phải bug.

Các sliding windows chồng lấn nên không độc lập thống kê; final research cần
block-bootstrap uncertainty, seasonal baseline, region/extreme/spherical
spectra diagnostics và nhiều seed. Không tinh chỉnh hyperparameters theo test year.
Không dùng RMSE thấp làm bằng chứng mass/energy conservation. Không thêm conservation
penalty khi thiếu đúng flux/forcing/vertical variables của budget.

Các ablation có giá trị sau baseline: forcing/time/static fields, K2 curriculum,
scalar/vector harmonics, stochastic closure và cross-resolution transfer.
Đây là thay đổi mô hình cần experiment riêng; không nằm trong bản sửa correctness.
