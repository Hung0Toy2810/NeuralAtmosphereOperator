# ERA5/WB2: 71 kênh trạng thái cho SFNO

Kiểm tra trực tiếp ngày **2026-09-08** trên metadata hợp nhất và coordinate `level`
của [nguồn Zarr đang cấu hình](https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr/.zmetadata).

Nguồn có **62 data variables**: 34 biến động đơn tầng, 15 biến động trên 13 tầng
áp suất và 13 trường tĩnh. Trải các tầng ra tạo 229 bản đồ động và 13 bản đồ tĩnh.
Đây không phải một tensor gồm 73 kênh. [WeatherBench2 Data Guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html)
mô tả cùng nguồn, cadence 6 giờ và 13 tầng áp suất.

Đối chiếu [bộ 73 kênh của NVIDIA Makani](https://github.com/NVIDIA/makani/blob/v0.1.0/config/sfnonet.yaml),
archive thiếu `100m_u_component_of_wind` và `100m_v_component_of_wind`.
Toàn bộ phần còn lại có sẵn: **6 + 5 × 13 = 71 kênh**. Không thay hai biến thiếu
bằng biến khác chỉ để đủ số 73.

## Thứ tự kênh được lưu và dự báo

| Chỉ số, bắt đầu từ 1 | Biến | Tầng áp suất | Số kênh |
|---|---|---|---:|
| 1–6 | u10, v10, T2m, surface pressure, mean sea-level pressure, TCWV | đơn tầng | 6 |
| 7–19 | `u_component_of_wind` | tất cả 13 tầng | 13 |
| 20–32 | `v_component_of_wind` | tất cả 13 tầng | 13 |
| 33–45 | `geopotential` | tất cả 13 tầng | 13 |
| 46–58 | `temperature` | tất cả 13 tầng | 13 |
| 59–71 | `specific_humidity` | tất cả 13 tầng | 13 |

Thứ tự tầng trong mỗi biến: **50, 100, 150, 200, 250, 300, 400, 500, 600, 700,
850, 925, 1000 hPa**, đã đọc từ coordinate của nguồn. Tên đầy đủ, units và tầng
được lưu cùng tensor; downloader và loader dùng chung cấu hình kênh.

Danh sách này là contract bất biến trong mã nguồn. `WeatherBenchDownloadConfig`
không còn nhận danh sách biến hoặc tầng áp suất; downloader luôn đọc đủ 71 kênh,
giữ nguyên cadence 6 giờ, rồi chỉ biến đổi trường bằng conservative regridding
từ 0,25° xuống 0,5°. Việc flatten thành trục `channel` chỉ đổi layout lưu trữ.

Relative humidity ở 500 hPa từng thuộc bộ compact 26 kênh. Đây là biến dẫn xuất,
không nằm trong bộ 73 kênh SFNO đối chiếu; cấu hình mới dùng specific humidity
trên đủ 13 tầng. Các trường tĩnh, precipitation/flux và biến chẩn đoán khác vẫn
cần thiết kế input/target riêng trước khi đưa vào mô hình.

## Chuyển từ bộ 26 kênh

- Model mặc định: **SC2–E192–L6**, input/output 71 kênh, lưới 361×720.
- Store mới: `data/dataset/era5_1995_2020_0p5deg_71ch.zarr`.
- Sample mới: `data/dataset/era5_sample_0p5_71ch.zarr`.
- Tạo lại statistics từ train 1995–2018 trong thư mục stats trống. Bảo quản bundle
  cũ dưới tên riêng; không dùng statistics 26 kênh với state 71 kênh.
- Dùng run directory mới. Checkpoint E128/26 kênh không tương thích trực tiếp với
  E192/71 kênh qua `--resume` hoặc `--init-checkpoint`.
- K=1 vẫn là dự báo 6 giờ; train/validation/test dùng cùng objective một bước.

Tensor 1995–2020 có 37,988 states; mỗi state float32 chiếm 73,817,280 bytes.
Tổng dung lượng logic **2,804,170,832,640 bytes ≈ 2.804 TB**. Đây là kích thước
chưa nén; cần đo compression trên sample trước khi quyết định dung lượng volume.

Có thể kiểm tra lại nguồn mà không tải payload khí tượng bằng:

```bash
python data/explore_data.py --remote
```
