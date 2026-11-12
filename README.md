# MAIC-Research

## Tăng cường khả năng khám phá trong học tăng cường đa tác tử dựa trên Information Gain

### 1. Tổng quan

Dự án nghiên cứu khả năng khám phá trong học tăng cường đa tác tử hợp tác
(Cooperative Multi-Agent Reinforcement Learning – MARL), tập trung vào phương
pháp MAIC sử dụng Information Gain để định hướng quá trình khám phá môi trường.

Nghiên cứu bao gồm tái hiện kết quả MAIC, phân tích tính ổn định của
quá trình tối ưu chính sách, thử nghiệm Exact KL và đánh giá khả năng
mở rộng trên các môi trường có kích thước khác nhau.

### 2. Mục tiêu

- Tái hiện thực nghiệm MAIC và các thuật toán đối chứng.
- Phân tích vai trò của Information Gain trong khám phá.
- Kiểm tra ảnh hưởng của Exact KL đến quá trình học.
- Đánh giá tính ổn định của chính sách qua nhiều random seed.
- Kiểm chứng hiệu quả trên môi trường 16×16, 24×24 và 32×32.

### 3. Mã nguồn kế thừa

Dự án được phát triển trên cơ sở mã nguồn MAIC của Hoàng Văn Phi.

Repository gốc:
https://github.com/hoangvanphi2004/MAIC

Phiên bản mã gốc được ghi nhận trong:
docs/phi_source_commit.txt

Các mã nguồn kế thừa và những thành phần triển khai bổ sung được phân biệt
trong tài liệu nghiên cứu.

### 4. Thuật toán

| Thuật toán | Vai trò |
|---|---|
| MAIC | Phương pháp sử dụng Information Gain |
| MASAC | Đối chứng không sử dụng Information Gain |
| MADDPG | Thuật toán đối chứng |
| COMA-configured MAIC | Cấu hình IG=0 và entropy=0 theo mã Phi |
| MAIC + Exact KL | Biến thể nghiên cứu cơ chế KL |
| MASAC + Exact KL | Đối chứng cho biến thể Exact KL |

Lưu ý: COMA-configured MAIC không được đồng nhất với triển khai
COMA on-policy nguyên bản.

### 5. Môi trường thực nghiệm

- Multi-Target: khám phá và thu thập mục tiêu.
- PushBox: đẩy hộp trong môi trường hợp tác.
- PassSparse: phối hợp kích hoạt công tắc và di chuyển qua cửa.

Các kích thước nghiên cứu:

| Kích thước | Horizon |
|---|---:|
| 16×16 | 40 |
| 24×24 | 60 |
| 32×32 | 80 |

### 6. Thực nghiệm

#### 6.1. Tái hiện trên môi trường 16×16

Các thuật toán MAIC, MASAC và MADDPG được huấn luyện trên
10 random seed từ 42 đến 51.

Cấu hình COMA theo mã nguồn Phi cũng đã hoàn thành
30 run, mỗi run 5.000 episode.

Kết quả đánh giá COMA-configured MAIC:

| Môi trường | Greedy success | Stochastic success |
|---|---:|---:|
| Multi-Target | 0% | 0% |
| PushBox | 0% | 0% |
| PassSparse | 0% | 0% |

Các giá trị trên được tính từ 10 seed, mỗi seed được đánh giá
100 episode cho mỗi chế độ.

#### 6.2. Phân tích Exact KL

Thực nghiệm MAIC + Exact KL và MASAC + Exact KL
trên PushBox16 với 10 random seed.

Kết quả huấn luyện cho thấy 6/10 seed MAIC + Exact KL
có reward trung bình dương trong 500 episode cuối,
so với 1/10 seed của MASAC + Exact KL.

Đây là kết quả huấn luyện, không đồng nghĩa với tỷ lệ thành công
trên tập đánh giá độc lập.

#### 6.3. Mở rộng môi trường

Các môi trường 24×24 và 32×32 đã được kiểm tra
về khả năng khởi tạo và chạy thử.

Thực nghiệm huấn luyện mở rộng đang tiếp tục được cập nhật.

### 7. Cấu trúc mã nguồn

- third_party/MAIC/: mã nguồn MAIC kế thừa.
- baselines/: thuật toán đối chứng bổ sung.
- tools/: công cụ huấn luyện, đánh giá và thống kê.
- configs/: cấu hình thực nghiệm.
- patches/: những điều chỉnh mã nguồn.
- reports/: kết quả thực nghiệm.
- docs/: tài liệu nghiên cứu.

### 8. Hạn chế

Kết quả có thể nhạy với random seed, mất ổn định trong quá trình
huấn luyện và chưa chứng minh khả năng khái quát hóa trên mọi
cấu hình môi trường.

Việc tăng kích thước lưới không nhất thiết làm tăng độ phức tạp
phối hợp giữa các tác tử.

### 9. Ghi nhận

Dự án ghi nhận đóng góp của Hoàng Văn Phi đối với mã nguồn
MAIC ban đầu và các tác giả của những thuật toán liên quan.

Việc phân phối mã nguồn kế thừa phải tuân thủ quyền sử dụng
của tác giả gốc.
