# Thiết lập và tái lập thực nghiệm Study16

## 1. Phạm vi

| Thuộc tính | Thiết lập |
|:--|:--|
| Môi trường | `MultiGrid-MultiTargetEmpty-16x16-v0`, `MultiGrid-PushBox-16x16-v0`, `MultiGrid-PassSparse-16x16-v0` |
| Số tác tử | 2 |
| Kích thước | 16 × 16 |
| Horizon | 40 bước/episode |
| Thuật toán | MAIC, MAIC + Exact KL, MASAC, MASAC + Exact KL, MADDPG, COMA_Phi |
| Seed huấn luyện | 42–46 |
| Episode huấn luyện | 10.000/run |
| Số run | 90 |
| Checkpoint trung gian | Mỗi 500 episode |
| Đánh giá cuối | 100 episode mỗi seed và mỗi chế độ greedy/stochastic |
| Eval seed gốc | 20260 |
| Số JSON đánh giá cuối | 180 |

## 2. Siêu tham số MAIC và các biến thể

Theo cấu hình từ revision MAIC gốc `ded6164f99742db4769a7f05b45f6e6007ff8b1a` và phần điều chỉnh trong pipeline Study16:

| Tham số | Giá trị |
|:--|--:|
| `replay_size` | 100000 |
| `batch_size` | 1024 |
| `start_steps` | 100 |
| `hidden_dim` | 128 |
| `lr` | 0.0003 |
| `gamma` | 0.99 |
| `tau` | 0.02 |
| `alpha1` khởi tạo | 0.02 |
| `alpha2` khởi tạo | 0.02 |
| `alpha_kl` | 1.0 |
| `policy_update_steps` | 3 |
| `auto_entropy_tuning` | True |
| `target_entropy_scale` | 0.2 |
| `obs_state_mode` | simple |
| `steps_per_update` | 1 |
| `updates_num` | 1 |

**Giới hạn:** Không suy diễn tất cả siêu tham số này sang MADDPG; cấu hình thực tế của mỗi run cần được đối chiếu với `runs_study16/<env>/<algo>/seed<seed>/config.json` và `run_meta.json`. Không lấy siêu tham số khác trong luận văn thay cho cấu hình đã chạy.

## 3. Chuẩn bị môi trường

Trên một máy mới có Python, các thư viện phù hợp và GPU được hỗ trợ, từ thư mục gốc của repository:

```bash
git clone https://github.com/hoangvanphi2004/MAIC source_maic
git -C source_maic checkout ded6164f99742db4769a7f05b45f6e6007ff8b1a
python3 -m pip install -r requirements.txt
python3 -m pip install -e source_maic/refactor/multigrid
python3 study16_pipeline.py prepare
python3 study16_pipeline.py status
```

Bản mã Exact KL được lưu riêng tại `variants/exactkl/`; mã gốc nằm trong `source_maic/`. Không thay mã gốc bằng biến thể khi chạy đối chứng.

## 4. Huấn luyện và đánh giá

```bash
python3 study16_pipeline.py train --jobs 2
python3 study16_pipeline.py status
python3 study16_pipeline.py eval --jobs 2
python3 study16_pipeline.py summarize
```

Chỉ huấn luyện lại khi có chủ đích. Lệnh `train --retry-failed` di chuyển run chưa hoàn thành vào thư mục backup rồi bắt đầu lại từ episode 0; nó **không resume chính xác** checkpoint dở.

## 5. Tái tạo hình

```bash
python3 -m pip install pandas numpy matplotlib
python3 scripts/study16/plot_study16_color.py \
  --csv results/study16/per_seed_values.csv \
  --runs runs_study16 \
  --out assets/study16/figures
```

Hình cuối kỳ có thể dựng từ CSV 90 run. Đường học và đồ thị độ ổn định checkpoint phải lấy từ bản JSON huấn luyện/đánh giá tương ứng; nếu thiếu JSON thì không tự suy diễn số liệu.

## 6. Minh họa và số liệu kiểm chứng

Các video trong [`assets/study16/demos`](../../assets/study16/demos) mô tả từng episode. Chúng không thay thế đánh giá qua nhiều episode và không được sử dụng làm bằng chứng thống kê độc lập. File `source_maic/refactor/inference.mp4` là video mã nguồn gốc, không phải video Study16.

Thống kê theo seed và hướng kiểm chứng thêm được trình bày tại [phân tích nghiên cứu](PHAN_TICH_THEO_SEED.md). Checkpoint gốc và log cần lưu trữ riêng theo [hướng dẫn lưu trữ](LUU_TRU_DU_LIEU.md).
