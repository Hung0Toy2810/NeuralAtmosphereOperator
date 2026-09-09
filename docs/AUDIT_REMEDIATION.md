# Theo dõi xử lý audit trước GPU training

Báo cáo gốc trong `audit/2026-09-06/` mô tả commit trước sửa và được giữ nguyên.
Bảng này theo dõi cả 22 mục theo mức độ. “Đã sửa” nghĩa là lỗi implementation
đã có biện pháp và regression tương ứng; không có nghĩa đã chứng minh forecast skill.

> **Trạng thái cấu hình hiện hành (2026-09-09):** project đã chuyển sang
> contract cố định 71 kênh và SFNO-SC2-L6-E192. Mọi phép đo E128/26 kênh bên
> dưới là bằng chứng lịch sử của đợt remediation trước, không phải benchmark
> capacity, VRAM hay stability của baseline hiện tại.

## Critical và High

| ID | Trạng thái | Thay đổi / điều kiện còn lại |
|---|---|---|
| C01 | Đã sửa contract | Train lưu forecast contract; eval/rollout kiểm ordered channels/units, actual latitude/longitude, cadence và physical step. Không crop ngầm. Valid/test từ chối overlap train; test từ chối overlap validation của checkpoint. Override statistics không bỏ các guard này. |
| H01 | Đã sửa protocol mặc định; cần dữ liệu mới | Validation 2019, test 2020, train vẫn 1995–2018. Panel validation 32 starts trải đều khoảng hợp lệ; lưu sample identity. Cần tải đủ năm và final evaluation rộng hơn; chưa có production score. |
| H02 | Đã sửa | Streaming float64 Chan moments, joint-channel reads, rolling delta buffer và online climatology. RAM không tăng theo số năm. Kiểm với direct weighted reductions, stride 3 và scheduler cache. |
| H03 | Đã sửa | Fail fast finite loss, `clip_grad_norm_(error_if_nonfinite=True)` trước optimizer; không advance scheduler khi exception. Save checkpoint từ chối non-finite model/Adam state; giữ last-good. Có injection tests qua CLI. Chặn FP16+GradScaler training profile chưa hỗ trợ complex gradients; BF16/FP32 giữ nguyên. |
| H04 | Đã sửa claim; giữ baseline | Notebook không còn suy isotropy từ complex shared-degree kernels. Không đổi kernel hoặc norm để tạo một architecture mới mà chưa có controlled experiment. Vector/SO(3) research vẫn mở. |
| H05 | Đã sửa | Resume signature có warmup factor, actual sample-index hash/count, validation panel/workers, deterministic flag, grid/unit và payload inventory. Same-config CPU resume exact; thay warmup/count bị chặn. |
| H06 | Bổ sung công cụ; efficacy còn mở | Evaluation có persistence/climatology, physical bias, variance ratio, ACC valid counts; preflight có precision/constant/gradient/lead diagnostics. Chưa có trained production/held-out ERA5 results, seasonal climatology, uncertainty, full physical budgets hoặc ensemble skill. |
| H07 | Đã sửa | `forecast_states` không cần truth; `rollout --forecast-only` forecast từ initialization sát cuối store ra ngoài observed interval. Target Zarr chỉ có ở verification mode. Valid timestamps được tạo theo trained physical step. |
| H08 | Đã sửa bundle | Stats v4 bắt buộc; hash từng artifact, units/grid/cadence/train interval và decoded training payload checksum. Atomic publication vào thư mục mới, không trộn bundle khi interrupted. Full scan đối chiếu payload; không tin metadata-only là proof of all data bytes. |

## Medium và Low

| ID | Trạng thái | Thay đổi / điều kiện còn lại |
|---|---|---|
| M01 | Đã sửa | Weighted vector norm cho zero subgradient hữu hạn ở perfect prediction và zero target; không thay objective bằng smoothing tùy ý. |
| M02 | Đã sửa | State xác minh exact set dimensions và transpose theo tên trước khi đọc. Loader hiện hành chỉ nhận flattened state đúng contract 71 kênh và từ chối representation surface/level cũ. Test store đổi thứ tự axes so đúng giá trị, không chỉ shape. |
| M03 | Đã tăng cường; cross-hardware còn mở | Runtime lưu packages, operator source hash, flags, driver/hardware/thread count; wheel layout hash đúng package. Data inventory phát hiện chunk edits dù metadata không đổi. Full scan là content check; exact CUDA continuation vẫn phải chứng minh trên target machine. |
| M04 | Đã chặn cấu hình sai | `use_complex_kernels=False` bị từ chối với backend 0.7.4 thay vì cho chạy một ablation giả. Real kernels cần implementation và retraining riêng. |
| M05 | Đã chặn tổ hợp không an toàn | SC1 khác input/internal grid, L1+LayerNorm và zero retained modes fail ngay ở config. Default SC2/L6 giữ nguyên. |
| M06 | Đã sửa constant leakage; E192/CUDA còn mở | Phép đo lịch sử full E128/L6 CPU: constant output spatial std 2.2109→0 nhờ anchored centering và analytical degree-zero SHT. Formula/gradcheck pass; random-field output RMS difference 2.9e−6. Không đổi epsilon/parameters. Baseline E192/71 kênh vẫn cần preflight và real-data pilot trên GPU đích. |
| M07 | Đã sửa các bottleneck được xác định | Lazy targets đọc từng lead ở validation/evaluation/export; CPU/device truth buffers không tăng theo K. Cache area weights, validate channel weights một lần, metric reductions giữ trên device. Throughput và target prefetch cần CUDA profiler trước bước tối ưu tiếp. |
| M08 | Đã sửa | Reject max_samples<=0 và empty metrics; reject non-finite fields/reductions. Metadata có hashes/precision/actual initializations/overrides. Không ghi đè evaluation đã hoàn tất; không giữ milestones từ run mới không yêu cầu. |
| M09 | Đã đồng nhất measure | In-memory normalizer helper dùng spherical area như CLI và nhận latitude rõ ràng; doc nói helper không tạo provenance bundle để train. |
| M10 | Đã sửa | Sample benchmark đòi để lại ít nhất một transition ngoài train; batch3 với T4 bị từ chối ở CLI và API. Sample vẫn chỉ là timing/overfit diagnostic. |
| M11 | Đã sửa | Downloader từ chối stride không chia hết duration của batch, tránh phase reset. Default 6h/7day giữ nguyên. |
| L01 | Sample lịch sử đã xác minh; cần sample 71 kênh | Sample ERA5 26 kênh/4 states (~73MiB) tại `era5_sample_audited.zarr` là artifact của audit cũ và không tương thích với pipeline hiện hành. Downloader/readiness mặc định nay dùng tên mới 71 kênh; cần tạo sample này trước preflight. |
| L02 | Đã sửa phần ảnh hưởng vận hành | Plot phân biệt train objective và validation selection; archive ledger/stale validation khi resume checkpoint cũ. Public model rollout dùng chung generator, discount range thống nhất, bỏ dead handle. Parameter count vẫn là allocated components, không claim identifiable degrees of freedom. |

## Kiểm chứng và giới hạn

Regression nằm tại `test/test_audit_regressions.py`, ngoài suite gốc. Bao gồm:

- CLI statistics → train → checkpoint → exact resume → evaluation/baselines → forecast-only.
- Guard sai cadence, shifted/regional grid, train-as-test, validation-as-test, empty eval và mixed statistics.
- Changed warmup/max_samples rejected; non-finite loss/gradient không update; checkpoint xấu không ghi đè file tốt.
- Weighted statistics so direct float64, bounded scheduler cache, lazy/eager target equality và payload-edit identity.
- Unsupported configurations, relative loss zero subgradient và CPU execution của preflight.

Các kết quả local mới được lưu dưới `audit/remediation/`. CUDA không có trong
workspace này. Không chạy full corpus hoặc long GPU training; không báo VRAM ước tính
như VRAM đã đo. [Runbook GPU](GPU_TRAINING.md) nêu lệnh cần chạy trên GPU trước
khi dùng baseline dài hạn.

Trong đợt remediation gốc, dataset channels và số SFNO parameters chưa thay đổi.
Sau đó project đã chuyển riêng sang contract 71 kênh, objective
standardized-tendency và E192–L6. Numerical SHT path giữ trường hằng vẫn được
duy trì; đây không phải exact reproduction của arithmetic upstream. Statistics
phải tính lại cho đúng state 71 kênh; checkpoint cũ thiếu forecast contract hoặc
dùng E128/26 kênh không được tự nâng cấp bằng phỏng đoán.

## Kết quả của đợt audit gốc (lịch sử)

- **67 tests pass**; Pyright/Pylance kiểm 42 file với **0 errors, 0 warnings**;
  Ruff F checks và `git diff --check` pass.
- MPS forward/backward pass; chỉ có cảnh báo padding performance của dependency.
- Phép đo lịch sử full E128/L6, 361×720 CPU: constant spatial std **2.2109 → 0**; zero input vẫn zero. Near-constant perturbation 1e−5 vẫn cho output spatial std khoảng **2.3198**; không suy rộng con số này cho E192/71 kênh.
- Pilot ERA5 thật dùng E8/L2, full grid, train 00→06 UTC và validation 12→18 UTC cùng 2018-01-01. Statistics chỉ từ hai training states. Sau **12 updates**, train loss **16.7129 → 11.6838**, validation loss **16.9253 → 12.0285**, không non-finite gradients. Hai periods cùng ngày tương quan mạnh: không phải forecast-skill evidence.
- Chưa tải corpus 71 kênh 1995–2020; chưa có measurements E192 trên GPU đích hoặc long-run checkpoint hiện hành.

Bằng chứng: [pytest](../audit/remediation/pytest.txt),
[MPS](../audit/remediation/backend_mps.txt),
[full-grid trước sửa](../audit/remediation/fullgrid_conditioning_before_cpu.json),
[full-grid sau sửa](../audit/remediation/fullgrid_conditioning_after_cpu.json),
[pilot dữ liệu thật](../audit/remediation/real_sample_probe.json).
Các harness trong cùng thư mục cho phép lặp lại conditioning/pilot.

## Kiểm chứng baseline hiện hành

- Runtime defaults: contract cố định 71 kênh, SC2–L6–E192, microbatch 1,
  accumulation 8 và K=1.
- **77 tests pass**; Pyright/Pylance kiểm cả `audit/remediation` với
  **0 errors, 0 warnings**; Ruff F checks trên các file vừa sửa và
  `git diff --check` đều pass.
- Benchmark sample mặc định chỉ chạy E192–L6. Các width khác vẫn có thể truyền
  tường minh để làm thí nghiệm, nhưng không còn là preset mặc định.
- Hai remediation harness đang chạy theo channel count/path từ config hiện hành.
  Snapshot và output trong `audit/2026-09-06/` cùng các JSON cũ vẫn được giữ
  nguyên để bảo toàn bằng chứng lịch sử.
