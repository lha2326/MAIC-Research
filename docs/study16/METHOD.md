# Phương pháp nghiên cứu điều chuẩn phân kỳ KL trong MAIC

## 1. Đối tượng và phạm vi

Nghiên cứu kiểm tra tác động của cách đưa regularization theo phân kỳ Kullback–Leibler (KL) vào gradient của Actor trong MAIC. **Không thay đổi** bộ ước lượng information gain, kiến trúc Actor–Critic hoặc định nghĩa counterfactual advantage. MAIC gốc được ghim tại commit `ded6164f99742db4769a7f05b45f6e6007ff8b1a` của [mã nguồn Hoàng Văn Phi](https://github.com/hoangvanphi2004/MAIC).

Các mô hình MAIC và MASAC không đồng nghĩa với tất cả biến thể của các thuật toán này trong tài liệu. `COMA_Phi` là cấu hình MAIC với hệ số IG và entropy bằng 0, không phải triển khai COMA on-policy độc lập.

## 2. Chính sách và mục tiêu tối ưu

Chính sách chung của $n$ tác tử được phân tích thành

$$
\pi_{\theta}(\mathbf a\mid\mathbf o)
=\prod_{i=1}^{n}\pi_{\theta_i}(a_i\mid o_i).
$$

Với hành động $\mathbf a$ và quan sát $\mathbf o$, đặt $\ell_{\theta}=\sum_j\log\pi_{\theta_j}(a_j\mid o_j)$; định nghĩa $\ell_{\mathrm{old}}$ tương ứng cho chính sách tham chiếu không nhận gradient.

Trong mã nguồn MAIC gốc, hệ số cập nhật Actor có dạng

$$
\begin{aligned}
g_i^{\mathrm{sur}}={}&(\alpha_1c_H+\lambda)(\ell_{\theta}+1)\\
&-\lambda\ell_{\mathrm{old}}-\widehat A_i-\alpha_2c_I\widehat I.
\end{aligned}
$$

Biểu thức surrogate sử dụng phép dừng gradient ở hệ số nhân:

$$
\mathcal L_{\mathrm{sur}}=\sum_i\mathbb E_{\mathcal B}
\left[\mathrm{sg}(g_i^{\mathrm{sur}})\log\pi_{\theta_i}(a_i\mid o_i)\right].
$$

Trong đó $\mathrm{sg}$ là *stop-gradient*, $\widehat A_i$ là counterfactual advantage, $\widehat I$ là ước lượng information gain; $c_H,c_I$ là hệ số kích hoạt; $\lambda$ là hệ số điều chuẩn KL.

## 3. Điều chuẩn KL trực tiếp

Với không gian hành động rời rạc $\mathcal A_i$, phân kỳ KL được tính trên toàn bộ hành động:

$$
D_i(o_i;\theta_i)=\sum_{a_i\in\mathcal A_i}
\pi_{\theta_i}(a_i\mid o_i)
\log\frac{\pi_{\theta_i}(a_i\mid o_i)}{\pi_{\mathrm{old},i}(a_i\mid o_i)}.
$$

Sau khi loại bỏ thành phần surrogate KL khỏi hệ số Actor, đặt

$$
g_i^{\mathrm{base}}=\alpha_1c_H(\ell_{\theta}+1)
-\widehat A_i-\alpha_2c_I\widehat I.
$$

Hàm mất mát của biến thể được viết

$$
\mathcal L_{\mathrm{Actor}}^{\mathrm{ExactKL}}
=\mathcal L_{\mathrm{base}}+\lambda\mathcal L_{\mathrm{KL}},
$$

$$
\mathcal L_{\mathrm{base}}=\sum_i\mathbb E_{\mathcal B}
\left[\mathrm{sg}(g_i^{\mathrm{base}})\log\pi_{\theta_i}(a_i\mid o_i)\right],
$$

$$
\mathcal L_{\mathrm{KL}}=\sum_i\mathbb E_{\mathcal B}
\left[D_i(o_i;\theta_i)\right].
$$

Hạng KL được lấy gradient trực tiếp theo chính sách hiện tại. Phân phối chính sách cũ được cố định. Đây là thay đổi về biểu thức tối ưu Actor, **không phải thuật toán information gain mới**. Kết quả trên replay buffer không hàm ý bảo đảm tối ưu đơn điệu.

## 4. Thiết kế đối chứng

| Cấu hình | Hệ số IG | Hệ số entropy | Điều chuẩn trong Actor |
|:--|--:|--:|:--|
| MAIC | 1 | 1 | Hạng KL surrogate |
| MAIC + Exact KL | 1 | 1 | Hạng KL trực tiếp |
| MASAC | 0 | 1 | Hạng KL surrogate |
| MASAC + Exact KL | 0 | 1 | Hạng KL trực tiếp |
| COMA_Phi | 0 | 0 | Cấu hình MAIC gốc |
| MADDPG | — | — | Baseline độc lập |

Hệ số của các cấu hình MAIC là `scaled_information_gain_coef` và `scaled_entropy_coef` trong mã huấn luyện. Giá trị của những siêu tham số khác cần được truy xuất từ `config.json` thực tế của từng run; không áp các tham số nội bộ của MAIC cho MADDPG nếu baseline dùng cách triển khai khác.

## 5. Các giả thuyết thực nghiệm

1. Khác biệt giữa KL surrogate và KL trực tiếp có thể thay đổi khả năng học và duy trì chính sách ở nhiệm vụ thưa phần thưởng.
2. Tác động của KL có thể khác nhau khi bật hoặc tắt information gain.
3. Mức độ khám phá trạng thái không nhất thiết tương ứng với xác suất hoàn thành chuỗi phối hợp.

Đây là các giả thuyết được kiểm tra bằng thiết kế thực nghiệm, không phải kết luận lý thuyết. Xem [phân tích kết quả theo seed](PHAN_TICH_THEO_SEED.md).
