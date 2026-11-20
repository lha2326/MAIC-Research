# Dữ liệu thực nghiệm, checkpoint và tình trạng lưu trữ

## 1. Dữ liệu có trong repository thông thường

| Thành phần | Đường dẫn | Trạng thái |
|:--|:--|:--|
| Bảng số liệu theo seed | [`results/study16/per_seed_values.csv`](../../results/study16/per_seed_values.csv) | 90 dòng (3 môi trường × 6 thuật toán × 5 seed) |
| Thống kê trung bình và độ lệch chuẩn | [`results/study16/summary.csv`](../../results/study16/summary.csv) | 18 cấu hình |
| Hình và PDF vector | [`assets/study16/figures`](../../assets/study16/figures) | Bao gồm hình cuối kỳ và hình từ lịch sử huấn luyện nếu đủ JSON |
| Video và metadata | [`assets/study16/demos`](../../assets/study16/demos) | Năm bộ GIF, MP4 và JSON |
| Script thực nghiệm | [`study16_pipeline.py`](../../study16_pipeline.py) | Điều phối chuẩn bị, huấn luyện, đánh giá và tổng hợp |
| Tài liệu nghiên cứu | [`docs/study16`](./) | Phương pháp, phân tích, tái lập |

## 2. Dữ liệu lớn không được commit vào Git

Toàn bộ thư mục `runs_study16/` có thể chứa 90 file `model_final.pth`, các checkpoint trung gian `model_ep*.pth`, `metrics_latest.json`, cấu hình, log và JSON đánh giá. Dữ liệu nặng không nên đưa trực tiếp vào lịch sử Git vì làm repository khó tái sử dụng, đồng thời có giới hạn kích thước đối với từng file.

**Tình trạng GitHub Release:** chỉ công bố là *đã lưu trữ đầy đủ* khi người vận hành đã kiểm tra GitHub Release thực tế có đầy đủ các phần archive và mã SHA-256 tương ứng. Chỉ riêng các hình và video có trên `main` không đồng nghĩa với việc checkpoint đã có trên GitHub Release.

Script đóng gói và đẩy lên Release được cung cấp riêng cùng gói cập nhật dự án, sử dụng dữ liệu thực tại VAST. Cần giữ bản sao ngoài VAST trước khi hủy instance.

## 3. Tiêu chí kiểm tra bộ dữ liệu đầy đủ

1. 90 file `DONE.json` báo trạng thái hoàn thành.
2. 90 checkpoint `model_final.pth` có kích thước khác 0.
3. 180 file `eval_model_final_greedy.json` / `eval_model_final_stochastic.json` đọc được và có `success_rate`, `reward_mean`.
4. Toàn bộ checkpoint trung gian được kiểm đếm riêng, không thay thế checkpoint cuối.
5. File archive đi kèm manifest danh sách tệp, mã SHA-256, hướng dẫn nối/giải nén.
6. Sau upload, kiểm tra tên và kích thước các asset trên Release, đồng thời thử phục hồi ở một thư mục độc lập.

## 4. Lưu ý về tái lập

Checkpoint `.pth` hiện tại không bao gồm toàn bộ replay buffer và optimizer state nên **không cho phép resume chính xác phiên huấn luyện đã dừng**. Để tiếp tục một run trong tình trạng nhất quán, cần có cơ chế snapshot toàn bộ trạng thái học hoặc huấn luyện lại từ khởi tạo tương ứng.
