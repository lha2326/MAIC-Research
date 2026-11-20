# Lịch sử thực nghiệm và phạm vi so sánh

Các giai đoạn nghiên cứu được giữ tách biệt để không dùng nhầm kết quả thăm dò cho bộ thực nghiệm chính.

| Giai đoạn | Quy mô được ghi nhận | Chức năng |
|:--|:--|:--|
| Tái hiện MAIC/MASAC/MADDPG | 10 seed, tối đa 5.000 episode/run | Khảo sát khả năng học ban đầu của các thuật toán trên môi trường 16×16 |
| Cấu hình `COMA_Phi` | 10 seed, 5.000 episode/run | Đối chứng tắt IG và entropy trong mã MAIC gốc |
| Exact KL pilot | Nhiều seed, checkpoint đến 5.000 episode | Phát hiện tín hiệu cải thiện reward và trường hợp chính sách suy giảm |
| Study16, thực nghiệm chính | 5 seed × 6 thuật toán × 3 môi trường, 10.000 episode/run | Đánh giá có đối chứng ở checkpoint cuối với 180 JSON |
| Môi trường 24×24 và 32×32 | Thử khả năng khởi tạo/chạy trước đây | Chưa gộp vào kết quả chính của Study16 |

Các báo cáo và hình gốc của thí nghiệm trước được lưu tại [reports](../../reports), [reports_stage_b](../../reports_stage_b), [reports_mte16_10seeds](../../reports_mte16_10seeds) và [assets/demos](../../assets/demos). Xem [README lịch sử ngày 12/11/2026](README_20261112.md) để tìm công thức và phân tích từng bước trước khi chuẩn hóa Study16.

Không quy đổi số episode, số seed hoặc checkpoint của pilot thành kết quả của Study16. Cụ thể, một GIF thành công ở checkpoint 2.000 episode thuộc pilot không chứng minh chính sách ở checkpoint 10.000 episode thành công. Mọi kết luận trong bài báo được neo vào [bảng 90 run](../../results/study16/per_seed_values.csv), còn các thí nghiệm thăm dò chỉ dùng làm bối cảnh thiết kế.
