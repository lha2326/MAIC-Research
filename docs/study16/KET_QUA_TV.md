# Kết quả thực nghiệm Study16

Bộ thực nghiệm chính của bài báo gồm **3 môi trường × 6 cấu hình thuật toán × 5 seed (42–46) = 90 run**. Mỗi run huấn luyện 10.000 episode, horizon 40 bước. Đánh giá hai chế độ greedy và stochastic trên 100 episode/seed/chế độ; tổng số 180 bản ghi đánh giá cuối.

## Tỷ lệ hoàn thành nhiệm vụ ở checkpoint cuối

| Method | MTE16 greedy | MTE16 stochastic | PushBox16 stochastic | PassSparse16 stochastic |
|:--|--:|--:|--:|--:|
| MAIC | 9.8% | 41.4% | 0.0% | 0.0% |
| MAIC + Exact KL | 25.6% | 37.8% | 60.0% | 0.0% |
| MASAC | 7.2% | 17.4% | 0.0% | 0.0% |
| MASAC + Exact KL | 37.2% | 44.2% | 0.0% | 0.0% |
| MADDPG | 4.6% | 13.4% | 60.0% | 0.0% |
| COMA-configured MAIC | 0.0% | 0.0% | 0.0% | 0.0% |

*Các tỷ lệ trong bảng là trung bình qua 5 seed, không phải tỷ lệ thành công của một chính sách duy nhất.*

## Diễn giải

**MTE16.** MAIC + Exact KL tăng greedy success từ 9,8% lên 25,6%, nhưng stochastic success giảm từ 41,4% xuống 37,8%. MASAC + Exact KL đạt trung bình cao hơn MASAC ở cả hai chế độ. Kết quả phân hóa giữa các seed nên không thể khẳng định Exact KL luôn cải thiện MAIC.

**PushBox16.** MAIC gốc không thành công ở bất kỳ seed nào trong 5 seed. MAIC + Exact KL hoàn thành ở 3/5 seed, tương ứng giá trị trung bình 60%; MADDPG cũng đạt 3/5 seed thành công nhưng không trùng hoàn toàn các seed. Kiểm định hoán đổi dấu chính xác theo cặp, hai phía, cho so sánh MAIC–Exact KL có p = 0,25. Vì vậy không được diễn giải là ưu thế đã được chứng minh ở mức ý nghĩa 5%.

**PassSparse16.** Tất cả cấu hình có tỷ lệ thành công bằng 0%. MAIC + Exact KL ghi nhận số trạng thái duy nhất trung bình cao hơn MAIC, nhưng cần kiểm chứng định nghĩa chỉ số trong mã trước khi diễn giải. Tăng số trạng thái được ghi nhận không đồng nghĩa với phối hợp thành công.

## Giới hạn và yêu cầu tái lập

- So sánh bằng 5 seed huấn luyện độc lập; không xem 500 episode đánh giá là 500 seed.
- Không trộn với thực nghiệm 10 seed × 5.000 episode trước đây.
- `COMA_Phi` là cấu hình MAIC với IG = 0 và entropy = 0, không phải COMA on-policy nguyên bản.
- Checkpoint trung gian và video chỉ là công cụ kiểm tra, không thay thế tỷ lệ thành công tính trên nhiều episode.
- Giữ riêng dữ liệu gốc, bản tóm tắt, video chính sách và tài liệu dẫn xuất.

Xem [Results (English)](RESULTS_EN.md), [cách tái lập](REPRODUCIBILITY.md) và [bộ hình/video](FIGURES_AND_DEMOS.md).
