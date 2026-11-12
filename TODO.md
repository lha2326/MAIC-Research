# TODO — Tái hiện MAIC (theo buổi trao đổi với thầy, 31/10/2026)

## Thầy chốt
- [x] Lấy luôn code MAIC của Phi (github.com/hoangvanphi2004/MAIC), tái hiện đúng như trong repo, không chỉnh sửa.
- [x] Bỏ HASAC khỏi đợt này. HASAC chỉ để làm ablation sau, hoặc để thành open question / hướng bài riêng nếu không chạy được.
- [x] Toy problem dạng grid world có tính exploration, lấy từ multigrid của repo: chọn 3 bài (MultiTargetEmpty-16x16, PushBox-16x16, PassSparse-16x16).
- [x] Các phương pháp chạy: MASAC + IG (= MAIC), MASAC thường, MADDPG, COMA.
- [x] Đóng gói bằng Docker.
- [ ] Push lên GitHub (repo của mình) và share cho thầy.
- [ ] Phần của Phi: chỉ cite, không cần làm thêm.

## Chạy
- [ ] Smoke test trên Vast: `bash scripts/smoke_test.sh 4` (khoảng 15 phút).
- [ ] Chạy đủ 36 run (3 bài × 4 phương pháp × 3 seed): `bash scripts/run_all.sh <số job song song>`.
- [ ] Chép `runs/` và `reports/` về máy (có thể bỏ các file video `.mp4` để nhẹ).

## Đọc kết quả
- [ ] So sánh theo success rate (đánh giá greedy và stochastic) và `unique_states_seen`, không chỉ reward trung bình.
- [ ] Đối chiếu với checkpoint Phi để sẵn trong repo (`model_ep5400.pth`, MultiTargetEmpty): greedy success 67%, stochastic 87% (100 episode, eval seed 20260).
- [ ] Nếu bản train lại theo `train_config.json` không đạt gần mức đó: hỏi Phi config/số episode đã dùng để tạo `model_ep5400.pth`.

## Để sau (không thuộc đợt này)
- [ ] HASAC: ablation hoặc open question.
- [ ] COMA gốc on-policy (Foerster et al., 2018), nếu reviewer đòi baseline COMA đúng nghĩa.
- [ ] Đề xuất framework và lấp gap lý thuyết.
