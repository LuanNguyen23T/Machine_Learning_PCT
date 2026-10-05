# TỔNG ÔN TRỌNG TÂM: PCA VÀ LDA (LÝ THUYẾT - TÍNH TAY - CODE)
> **Học phần:** Học máy và ứng dụng (Machine Learning & Applications)  
> **Chương 2:** Trích chọn đặc tính (Feature Extraction)  
> **Tài liệu tham khảo chính:** Slide bài giảng PGS. TS. Phạm Công Thắng - ĐH Bách Khoa Đà Nẵng  
> **Notebook thực hành liên quan:**  
> - [C1_PCA.ipynb](file:///Users/nguyenhuuhoangluan/Git_Code/Machine_Learning_PCT/Midterm/Review/C1_PCA.ipynb)  
> - [C1_LDA.ipynb](file:///Users/nguyenhuuhoangluan/Git_Code/Machine_Learning_PCT/Midterm/Review/C1_LDA.ipynb)

---

## MỤC LỤC
1. [Bối cảnh: Giảm chiều dữ liệu & Trích chọn đặc tính](#1-bối-cảnh-giảm-chiều-dữ-liệu--trích-chọn-đặc-tính)
2. [Bảng so sánh tổng quan PCA vs LDA](#2-bảng-so-sánh-tổng-quan-pca-vs-lda)
3. [PCA (Principal Component Analysis) - Lý thuyết & Công thức](#3-pca-principal-component-analysis---lý-thuyết--công-thức)
   - 3.1. Ý nghĩa & Bài toán tối ưu
   - 3.2. Phương pháp 1: Ma trận hiệp phương sai (Covariance Matrix)
   - 3.3. Phương pháp 2: Phân tích kỳ dị (SVD)
   - 3.4. Tỷ lệ phương sai giải thích (Explained Variance Ratio - EVR)
4. [LDA (Linear Discriminant Analysis) - Lý thuyết & Công thức](#4-lda-linear-discriminant-analysis---lý-thuyết--công-thức)
   - 4.1. Ý nghĩa & Tiêu chuẩn Fisher
   - 4.2. Ma trận tán xạ trong lớp ($S_W$) và giữa các lớp ($S_B$)
   - 4.3. Bài toán trị riêng suy rộng & Giới hạn số chiều ($k \le C - 1$)
5. [Hướng dẫn giải tính tay chi tiết từng bước (Step-by-step Hand Calculation)](#5-hướng-dẫn-giải-tính-tay-chi-tiết-từng-bước)
   - 5.1. Bài toán mẫu tính tay PCA (từ $\mathbb{R}^2$ xuống $\mathbb{R}^1$)
   - 5.2. Bài toán mẫu tính tay LDA (2 lớp từ $\mathbb{R}^2$ xuống $\mathbb{R}^1$)
6. [Hướng dẫn cấu hình và áp dụng code Notebook khi đổi Dataset](#6-hướng-dẫn-cấu-hình-và-áp-dụng-code-notebook-khi-đổi-dataset)
7. [Checklist ghi nhớ nhanh trước khi vào phòng thi](#7-checklist-ghi-nhớ-nhanh-trước-khi-vào-phòng-thi)

---

# 1. BỐI CẢNH: GIẢM CHIỀU DỮ LIỆU & TRÍCH CHỌN ĐẶC TÍNH

### 1.1. Lời nguyền số chiều (Curse of Dimensionality - Slide 5, 10)
- Khi dữ liệu có số chiều $D$ lớn:
  - Không gian đặc trưng trở nên thưa thớt (sparse).
  - Khoảng cách Euclidean giữa các điểm mất dần ý nghĩa phân biệt.
  - Tốn tài nguyên tính toán và dễ gây hiện tượng quá khớp (Overfitting).
  - Khó trực quan hóa để phân tích (con người chỉ nhìn được tối đa 2D hoặc 3D).

### 1.2. Phân biệt Feature Selection vs Feature Extraction (Slide 11, 12, 13)
- **Feature Selection (Lựa chọn đặc trưng):** Giữ lại một tập con các biến gốc $x_{i_1}, x_{i_2}, \dots$ và loại bỏ các biến dư thừa.
- **Feature Extraction (Trích chọn đặc trưng):** Tạo ra một tập đặc trưng mới $y$ là hàm biến đổi / tổ hợp tuyến tính của các biến cũ:
  $$y = A(x_1, x_2, \dots, x_D)$$
  Trong đó, PCA và LDA là hai kỹ thuật trích chọn đặc tính tuyến tính quan trọng nhất.

---

# 2. BẢNG SO SÁNH TỔNG QUAN PCA VS LDA

| Đặc điểm / Tiêu chí | PCA (Principal Component Analysis) | LDA (Linear Discriminant Analysis) |
| :--- | :--- | :--- |
| **Loại thuật toán** | **Học không giám sát (Unsupervised)** | **Học có giám sát (Supervised)** |
| **Nhãn lớp ($y$)** | **Không dùng** (chỉ dùng ma trận $X$) | **Bắt buộc có** (dùng cả $X$ và nhãn $y$) |
| **Mục tiêu tối ưu** | Tối đa hóa **phương sai** của dữ liệu (bảo toàn năng lượng / thông tin biến thiên) | Tối đa hóa **khoảng cách giữa các lớp**, đồng thời thu hẹp **độ phân tán nội bộ từng lớp** |
| **Độ đo chính** | Ma trận hiệp phương sai mẫu $S$ | Ma trận trong lớp $S_W$ và giữa các lớp $S_B$ |
| **Tiêu chuẩn toán** | Cực đại hóa $u^T S u$ với $\|u\| = 1$ | Cực đại hóa tỷ số Fisher $J(w) = \frac{w^T S_B w}{w^T S_W w}$ |
| **Phương trình giải** | $S v_i = \lambda_i v_i$ | $S_W^{-1} S_B w_i = \lambda_i w_i$ |
| **Số chiều tối đa $k$** | Tối đa $D$ chiều ($k \le D$) | **Tối đa $C - 1$ chiều** ($k \le C - 1$, với $C$ là số lớp) |
| **Rủi ro / Hạn chế** | Trục có phương sai lớn nhất có thể làm lẫn lộn các lớp (Slide 35) | Nhạy cảm với phân phối lệch; giả định ma trận hiệp phương sai các lớp đồng nhất |

---

# 3. PCA (PRINCIPAL COMPONENT ANALYSIS) - LÝ THUYẾT & CÔNG THỨC

### 3.1. Ý nghĩa & Bài toán tối ưu (Slide 14-20)
Giả sử tập dữ liệu gồm $N$ quan sát: $\{x_1, x_2, \dots, x_N\}$, mỗi $x_n \in \mathbb{R}^D$.
- Vector trung bình toàn cục:
  $$\mu = \frac{1}{N} \sum_{n=1}^N x_n$$
- Chiếu điểm $x_n$ lên vector đơn vị $u_1$ ($\|u_1\| = u_1^T u_1 = 1$):
  $$y_n = u_1^T x_n$$
- Phương sai của dữ liệu sau khi chiếu:
  $$\sigma^2 = \frac{1}{N} \sum_{n=1}^N (u_1^T x_n - u_1^T \mu)^2 = u_1^T \left[ \frac{1}{N} \sum_{n=1}^N (x_n - \mu)(x_n - \mu)^T \right] u_1 = u_1^T S u_1$$
- Cực đại hóa $u_1^T S u_1$ với ràng buộc $u_1^T u_1 = 1$ bằng nhân tử Lagrange:
  $$\mathcal{L}(u_1, \lambda_1) = u_1^T S u_1 + \lambda_1 (1 - u_1^T u_1)$$
  Lấy đạo hàm theo $u_1$ và cho bằng $0$:
  $$\frac{\partial \mathcal{L}}{\partial u_1} = 2 S u_1 - 2 \lambda_1 u_1 = 0 \implies S u_1 = \lambda_1 u_1$$
- Nhân hai vế với $u_1^T$:
  $$u_1^T S u_1 = \lambda_1 u_1^T u_1 = \lambda_1$$
  $\Rightarrow$ **Phương sai cực đại chính là trị riêng $\lambda_1$ lớn nhất của ma trận $S$, và hướng chiếu thành phần chính thứ nhất chính là vector riêng $u_1$ tương ứng.**

### 3.2. Phương pháp 1: Ma trận hiệp phương sai (Covariance Matrix Method)
1. **Trung tâm hóa dữ liệu:**
   $$\bar{X} = X - \mu$$
2. **Tính ma trận hiệp phương sai mẫu ($D \times D$):**
   $$S = \frac{1}{N - 1} \bar{X}^T \bar{X}$$
3. **Giải phương trình đặc trưng:**
   $$\det(S - \lambda I) = 0 \implies \text{Tìm } \lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D \ge 0$$
4. **Tìm các vector riêng tương ứng và chuẩn hóa ($\|v_i\| = 1$):**
   $$(S - \lambda_i I) v_i = 0$$
5. **Xây dựng ma trận chiếu $W \in \mathbb{R}^{D \times M}$:**
   $$W = [v_1, v_2, \dots, v_M]$$
6. **Chiếu dữ liệu:**
   $$Y = \bar{X} W \in \mathbb{R}^{N \times M}$$
7. **Khôi phục dữ liệu xấp xỉ (Reconstruction):**
   $$\hat{X} = Y W^T + \mu$$

### 3.3. Phương pháp 2: Phân tích kỳ dị SVD (Singular Value Decomposition - Slide 25-30)
- Với ma trận trung tâm hóa $\bar{X} \in \mathbb{R}^{N \times D}$, phân tích SVD:
  $$\bar{X} = U \Sigma V^T$$
  Trong đó:
  - $U \in \mathbb{R}^{N \times N}$ là ma trận trực giao (Left Singular Vectors).
  - $\Sigma \in \mathbb{R}^{N \times D}$ chứa các giá trị kỳ dị $\sigma_i \ge 0$ trên đường chéo.
  - $V \in \mathbb{R}^{D \times D}$ là ma trận trực giao (Right Singular Vectors).
- Mối liên hệ toán học với ma trận hiệp phương sai $S$:
  $$S = \frac{1}{N - 1} \bar{X}^T \bar{X} = \frac{1}{N - 1} (U \Sigma V^T)^T (U \Sigma V^T) = \frac{1}{N - 1} V \Sigma^T U^T U \Sigma V^T = \frac{1}{N - 1} V \Sigma^2 V^T$$
  $\Rightarrow$ **Các cột của $V$ chính là các vector thành phần chính $W = V_{:, :M}$, và trị riêng là $\lambda_i = \frac{\sigma_i^2}{N - 1}$.**

### 3.4. Tỷ lệ phương sai giải thích (Explained Variance Ratio - EVR - Slide 23)
$$\text{EVR}_i = \frac{\lambda_i}{\sum_{j=1}^D \lambda_j}$$
Tỷ lệ tích lũy khi giữ lại $M$ thành phần chính:
$$\text{Cumulative EVR} = \frac{\sum_{i=1}^M \lambda_i}{\sum_{j=1}^D \lambda_j}$$

---

# 4. LDA (LINEAR DISCRIMINANT ANALYSIS) - LÝ THUYẾT & CÔNG THỨC

### 4.1. Ý nghĩa & Tiêu chuẩn Fisher (Slide 34-40)
LDA là phương pháp tìm không gian con sao cho các lớp được phân tách tốt nhất.
- Tiêu chuẩn tỷ số Rayleigh-Fisher:
  $$J(W) = \frac{\det(W^T S_B W)}{\det(W^T S_W W)}$$
- Cực đại hóa tỷ số này dẫn đến bài toán trị riêng suy rộng:
  $$S_W^{-1} S_B w_i = \lambda_i w_i$$

### 4.2. Các ma trận phân tán (Scatter Matrices - Slide 37, 40)
Giả sử có $C$ lớp, lớp $c$ có $N_c$ mẫu, tổng số mẫu $N = \sum_{c=1}^C N_c$.
1. **Vector trung bình từng lớp ($\mu_c$) và trung bình toàn cục ($\mu$):**
   $$\mu_c = \frac{1}{N_c} \sum_{x \in C_c} x, \qquad \mu = \frac{1}{N} \sum_{i=1}^N x_i = \sum_{c=1}^C \frac{N_c}{N} \mu_c$$
2. **Ma trận tán xạ trong lớp (Within-Class Scatter Matrix $S_W$):**
   $$S_c = \sum_{x \in C_c} (x - \mu_c)(x - \mu_c)^T \implies S_W = \sum_{c=1}^C S_c$$
   *(Ghi chú: $S_W$ là tổng ma trận phân tán của từng lớp quanh tâm của chính nó).*
3. **Ma trận tán xạ giữa các lớp (Between-Class Scatter Matrix $S_B$):**
   $$S_B = \sum_{c=1}^C N_c (\mu_c - \mu)(\mu_c - \mu)^T$$
   *(Ghi chú: $S_B$ đo độ phân tán của tâm các lớp quanh tâm toàn cục).*

### 4.3. Giới hạn số chiều cực kỳ quan trọng ($k \le C - 1$)
- Ma trận $S_B$ là tổng của $C$ ma trận có hạng 1: $(\mu_c - \mu)(\mu_c - \mu)^T$.
- Do $\sum_{c=1}^C N_c (\mu_c - \mu) = 0$ (phụ thuộc tuyến tính), nên hạng cực đại:
  $$\text{rank}(S_B) \le C - 1$$
- Vì vậy, ma trận $S_W^{-1} S_B$ chỉ có tối đa $C - 1$ trị riêng khác 0.
  $\Rightarrow$ **Số trục phân biệt tối đa của LDA là $k \le C - 1$**:
  - Với bài toán 2 lớp ($C = 2$): LDA chỉ chiếu được xuống **1 chiều duy nhất** ($k = 1$).
  - Với bài toán 3 lớp ($C = 3$): LDA chiếu được tối đa **2 chiều** ($k \le 2$).

---

# 5. HƯỚNG DẪN GIẢI TÍNH TAY CHI TIẾT TỪNG BƯỚC

---

## 5.1. BÀI TOÁN MẪU TÍNH TAY PCA (Từ 2 chiều xuống 1 chiều)

**Đề bài:** Cho 4 điểm dữ liệu trong không gian 2 chiều ($x_1, x_2$):
$$A(1, 2), \quad B(2, 4), \quad C(4, 4), \quad D(5, 6)$$
Ma trận dữ liệu $X \in \mathbb{R}^{4 \times 2}$:
$$X = \begin{bmatrix} 1 & 2 \\ 2 & 4 \\ 4 & 4 \\ 5 & 6 \end{bmatrix}$$
Hãy dùng PCA giảm chiều dữ liệu về 1 chiều.

---

### Bước 1: Tính vector trung bình $\mu$
$$\mu_{x_1} = \frac{1 + 2 + 4 + 5}{4} = \frac{12}{4} = 3$$
$$\mu_{x_2} = \frac{2 + 4 + 4 + 6}{4} = \frac{16}{4} = 4$$
$$\mu = \begin{bmatrix} 3 & 4 \end{bmatrix}$$

---

### Bước 2: Trung tâm hóa dữ liệu $\bar{X} = X - \mu$
Lấy từng dòng của $X$ trừ đi $[3, 4]$:
$$\bar{X} = \begin{bmatrix} 1-3 & 2-4 \\ 2-3 & 4-4 \\ 4-3 & 4-4 \\ 5-3 & 6-4 \end{bmatrix} = \begin{bmatrix} -2 & -2 \\ -1 & 0 \\ 1 & 0 \\ 2 & 2 \end{bmatrix}$$

---

### Bước 3: Tính ma trận hiệp phương sai $S = \frac{1}{N-1} \bar{X}^T \bar{X}$ (với $N - 1 = 3$)
Tính tích $\bar{X}^T \bar{X}$:
$$\bar{X}^T \bar{X} = \begin{bmatrix} -2 & -1 & 1 & 2 \\ -2 & 0 & 0 & 2 \end{bmatrix} \begin{bmatrix} -2 & -2 \\ -1 & 0 \\ 1 & 0 \\ 2 & 2 \end{bmatrix}$$
- Phần tử $(1, 1)$: $(-2)^2 + (-1)^2 + 1^2 + 2^2 = 4 + 1 + 1 + 4 = 10$
- Phần tử $(1, 2) = (2, 1)$: $(-2)(-2) + (-1)(0) + (1)(0) + (2)(2) = 4 + 0 + 0 + 4 = 8$
- Phần tử $(2, 2)$: $(-2)^2 + 0^2 + 0^2 + 2^2 = 4 + 0 + 0 + 4 = 8$

$$S = \frac{1}{3} \begin{bmatrix} 10 & 8 \\ 8 & 8 \end{bmatrix} = \begin{bmatrix} 3.3333 & 2.6667 \\ 2.6667 & 2.6667 \end{bmatrix}$$

---

### Bước 4: Tìm các trị riêng (Eigenvalues)
Giải phương trình đặc trưng: $\det(S - \lambda I) = 0$:
$$\det \begin{bmatrix} \frac{10}{3} - \lambda & \frac{8}{3} \\ \frac{8}{3} & \frac{8}{3} - \lambda \end{bmatrix} = 0$$
$$\left(\frac{10}{3} - \lambda\right)\left(\frac{8}{3} - \lambda\right) - \left(\frac{8}{3}\right)^2 = 0$$
$$\lambda^2 - \left(\frac{10}{3} + \frac{8}{3}\right)\lambda + \left(\frac{80}{9} - \frac{64}{9}\right) = 0$$
$$\lambda^2 - 6\lambda + \frac{16}{9} = 0$$
Áp dụng công thức nghiệm bậc 2 ($\Delta' = b'^2 - ac$):
$$\Delta' = (-3)^2 - 1 \times \frac{16}{9} = 9 - 1.7778 = 7.2222 \implies \sqrt{\Delta'} \approx 2.6874$$
Hai trị riêng thu được:
$$\lambda_1 = 3 + 2.6874 = 5.6874 \quad (\text{Trị riêng lớn nhất } \to \text{PC1})$$
$$\lambda_2 = 3 - 2.6874 = 0.3126$$

> **Tỷ lệ phương sai giữ lại khi giảm xuống 1 chiều:**
> $$\text{EVR}_1 = \frac{\lambda_1}{\lambda_1 + \lambda_2} = \frac{5.6874}{6.0000} \approx 94.79\%$$

---

### Bước 5: Tìm vector riêng tương ứng với $\lambda_1 = 5.6874$
Giải hệ thuần nhất: $(S - \lambda_1 I) v_1 = 0$:
$$\begin{bmatrix} 3.3333 - 5.6874 & 2.6667 \\ 2.6667 & 2.6667 - 5.6874 \end{bmatrix} \begin{bmatrix} v_{11} \\ v_{12} \end{bmatrix} = \begin{bmatrix} -2.3541 & 2.6667 \\ 2.6667 & -3.0207 \end{bmatrix} \begin{bmatrix} v_{11} \\ v_{12} \end{bmatrix} = 0$$
Từ hàng 1:
$$-2.3541 v_{11} + 2.6667 v_{12} = 0 \implies v_{12} = \frac{2.3541}{2.6667} v_{11} \approx 0.8828 v_{11}$$
Chọn $v_{11} = 1 \implies v_{12} \approx 0.8828$. Vector riêng chưa chuẩn hóa:
$$v_1' = \begin{bmatrix} 1 \\ 0.8828 \end{bmatrix}$$
**Chuẩn hóa về vector đơn vị $(\|v_1\| = 1)$:**
$$\|v_1'\| = \sqrt{1^2 + (0.8828)^2} = \sqrt{1 + 0.7793} = \sqrt{1.7793} \approx 1.3339$$
$$v_1 = \begin{bmatrix} 1 / 1.3339 \\ 0.8828 / 1.3339 \end{bmatrix} \approx \begin{bmatrix} 0.7497 \\ 0.6618 \end{bmatrix}$$
Ma trận chiếu 1 chiều:
$$W = \begin{bmatrix} 0.7497 \\ 0.6618 \end{bmatrix}$$

---

### Bước 6: Chiếu dữ liệu xuống 1 chiều: $Y = \bar{X} W$
Tọa độ chiếu của từng điểm:
- **Điểm A:** $y_A = (-2)(0.7497) + (-2)(0.6618) = -1.4994 - 1.3236 = -2.823$
- **Điểm B:** $y_B = (-1)(0.7497) + (0)(0.6618) = -0.750$
- **Điểm C:** $y_C = (1)(0.7497) + (0)(0.6618) = +0.750$
- **Điểm D:** $y_D = (2)(0.7497) + (2)(0.6618) = +1.4994 + 1.3236 = +2.823$

---

## 5.2. BÀI TOÁN MẪU TÍNH TAY LDA (2 lớp từ 2 chiều xuống 1 chiều)

**Đề bài:** Cho 2 lớp dữ liệu trong không gian 2 chiều:
- **Lớp 1 ($C_1$):** $x_1 = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad x_2 = \begin{bmatrix} 2 \\ 3 \end{bmatrix}$ ($N_1 = 2$)
- **Lớp 2 ($C_2$):** $x_3 = \begin{bmatrix} 4 \\ 4 \end{bmatrix}, \quad x_4 = \begin{bmatrix} 5 \\ 5 \end{bmatrix}$ ($N_2 = 2$)
Hãy dùng LDA tìm trục chiếu phân biệt tối ưu $w$ và tọa độ chiếu của các điểm.

---

### Bước 1: Tính trung bình từng lớp và trung bình toàn cục
- Tâm lớp 1:
  $$\mu_1 = \frac{1}{2} \left( \begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 2 \\ 3 \end{bmatrix} \right) = \begin{bmatrix} 1.5 \\ 2.5 \end{bmatrix}$$
- Tâm lớp 2:
  $$\mu_2 = \frac{1}{2} \left( \begin{bmatrix} 4 \\ 4 \end{bmatrix} + \begin{bmatrix} 5 \\ 5 \end{bmatrix} \right) = \begin{bmatrix} 4.5 \\ 4.5 \end{bmatrix}$$
- Tâm toàn cục:
  $$\mu = \frac{1}{4} (x_1 + x_2 + x_3 + x_4) = \begin{bmatrix} 3.0 \\ 3.5 \end{bmatrix}$$

---

### Bước 2: Tính ma trận tán xạ trong lớp ($S_W$)
- **Với lớp 1:**
  $$x_1 - \mu_1 = \begin{bmatrix} 1 - 1.5 \\ 2 - 2.5 \end{bmatrix} = \begin{bmatrix} -0.5 \\ -0.5 \end{bmatrix}, \qquad x_2 - \mu_1 = \begin{bmatrix} 2 - 1.5 \\ 3 - 2.5 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}$$
  $$S_1 = \begin{bmatrix} -0.5 \\ -0.5 \end{bmatrix} \begin{bmatrix} -0.5 & -0.5 \end{bmatrix} + \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix} \begin{bmatrix} 0.5 & 0.5 \end{bmatrix} = \begin{bmatrix} 0.25 & 0.25 \\ 0.25 & 0.25 \end{bmatrix} + \begin{bmatrix} 0.25 & 0.25 \\ 0.25 & 0.25 \end{bmatrix} = \begin{bmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{bmatrix}$$
- **Với lớp 2:**
  $$x_3 - \mu_2 = \begin{bmatrix} 4 - 4.5 \\ 4 - 4.5 \end{bmatrix} = \begin{bmatrix} -0.5 \\ -0.5 \end{bmatrix}, \qquad x_4 - \mu_2 = \begin{bmatrix} 5 - 4.5 \\ 5 - 4.5 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}$$
  $$S_2 = \begin{bmatrix} 0.25 & 0.25 \\ 0.25 & 0.25 \end{bmatrix} + \begin{bmatrix} 0.25 & 0.25 \\ 0.25 & 0.25 \end{bmatrix} = \begin{bmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{bmatrix}$$
- **Tổng ma trận tán xạ trong lớp:**
  $$S_W = S_1 + S_2 = \begin{bmatrix} 1.0 & 1.0 \\ 1.0 & 1.0 \end{bmatrix}$$

*(Lưu ý thi cử: Nếu ma trận suy biến do số điểm ít, đề thi thường cho ma trận $S_W$ khả nghịch, ví dụ $S_W = \begin{bmatrix} 1.1 & 0.9 \\ 0.9 & 1.1 \end{bmatrix}$ hoặc thêm thành phần chính quy hóa $\epsilon I$)*.

---

### Bước 3: Tìm trục chiếu tối ưu Fisher với bài toán 2 lớp
Với bài toán 2 lớp ($C=2$), hướng chiếu tối ưu Fisher có công thức nghiệm rút gọn kinh điển:
$$w \propto S_W^{-1} (\mu_1 - \mu_2)$$
- Vector hiệu tâm:
  $$\mu_1 - \mu_2 = \begin{bmatrix} 1.5 - 4.5 \\ 2.5 - 4.5 \end{bmatrix} = \begin{bmatrix} -3 \\ -2 \end{bmatrix}$$
- Giả sử $S_W = \begin{bmatrix} 1.1 & 0.9 \\ 0.9 & 1.1 \end{bmatrix}$:
  $$\det(S_W) = (1.1)^2 - (0.9)^2 = 1.21 - 0.81 = 0.40$$
  $$S_W^{-1} = \frac{1}{0.40} \begin{bmatrix} 1.1 & -0.9 \\ -0.9 & 1.1 \end{bmatrix} = \begin{bmatrix} 2.75 & -2.25 \\ -2.25 & 2.75 \end{bmatrix}$$
- Nhân với $(\mu_1 - \mu_2)$:
  $$w = \begin{bmatrix} 2.75 & -2.25 \\ -2.25 & 2.75 \end{bmatrix} \begin{bmatrix} -3 \\ -2 \end{bmatrix} = \begin{bmatrix} (2.75)(-3) + (-2.25)(-2) \\ (-2.25)(-3) + (2.75)(-2) \end{bmatrix} = \begin{bmatrix} -8.25 + 4.50 \\ 6.75 - 5.50 \end{bmatrix} = \begin{bmatrix} -3.75 \\ 1.25 \end{bmatrix}$$
- Chuẩn hóa $\|w\| = \sqrt{(-3.75)^2 + (1.25)^2} = \sqrt{14.0625 + 1.5625} \approx 3.9528$:
  $$w = \begin{bmatrix} -3.75 / 3.9528 \\ 1.25 / 3.9528 \end{bmatrix} \approx \begin{bmatrix} -0.9487 \\ 0.3162 \end{bmatrix}$$

---

### Bước 4: Chiếu các điểm dữ liệu $y = w^T x$
- **Lớp 1:**
  - $y_1 = w^T x_1 = (-0.9487)(1) + (0.3162)(2) = -0.3163$
  - $y_2 = w^T x_2 = (-0.9487)(2) + (0.3162)(3) = -0.9488$
  *(Tâm lớp 1 sau khi chiếu: $\approx -0.632$)*
- **Lớp 2:**
  - $y_3 = w^T x_3 = (-0.9487)(4) + (0.3162)(4) = -2.5300$
  - $y_4 = w^T x_4 = (-0.9487)(5) + (0.3162)(5) = -3.1625$
  *(Tâm lớp 2 sau khi chiếu: $\approx -2.846$)*

> **Kết luận:** Lớp 1 phân bố trong đoạn $[-0.95, -0.32]$, Lớp 2 phân bố trong đoạn $[-3.16, -2.53]$. Khoảng cách giữa 2 lớp rất xa, không có bất kỳ điểm nào bị chồng lấn!

---

# 6. HƯỚNG DẪN CẤU HÌNH VÀ ÁP DỤNG CODE NOTEBOOK KHI ĐỔI DATASET

Cả 2 file notebook trong thư mục `Midterm/Review/` đều được viết theo chuẩn **From Scratch** từng bước và hỗ trợ thay đổi tập dữ liệu tức thì.

### 6.1. Khi đổi Dataset trong [C1_PCA.ipynb](file:///Users/nguyenhuuhoangluan/Git_Code/Machine_Learning_PCT/Midterm/Review/C1_PCA.ipynb)
Chỉ cần sửa **Cell 2**:
```python
# ==============================================================================
# ⚙️ KHỐI CẤU HÌNH DATASET & THAM SỐ (CHỈ CẦN SỬA Ở ĐÂY)
# ==============================================================================

# 1. Đường dẫn file CSV mới (trong dataset/Sampledata/ có sẵn iris.csv, penguins.csv, tips.csv, ...)
DATA_PATH = "dataset/Sampledata/iris.csv"

# 2. Danh sách các cột đặc trưng số cần giảm chiều (Để None để TỰ ĐỘNG LẤY TẤT CẢ các cột số trong dataset, bất kể 4, 7, 10 hay 30 cột!)
FEATURE_COLS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]  # hoặc None

# 3. Cột nhãn dùng để tô màu trực quan hóa (để None nếu là Unsupervised hoàn toàn không nhãn)
LABEL_COL = "species"

# 4. Số thành phần chính cần giữ lại (mặc định 2 để vẽ biểu đồ 2D)
N_COMPONENTS = 2
```
*Các Cell sau sẽ tự động:*
- Cell 3: Điền giá trị thiếu (median) và chuẩn hóa Z-Score.
- Cell 4: Tính Covariance Matrix $S$, tìm trị riêng $\lambda$, vector riêng $W$, chiếu $Y = \bar{X} W$ và in bảng EVR.
- Cell 5: Tính bằng SVD $\bar{X} = U S V^T$ và kiểm tra đối chiếu sai số cực đại với phương pháp Covariance.
- Cell 6: Khôi phục $\hat{X} = Y W^T + \mu$ và tính sai số tái tạo (MSE).
- Cell 7+: Vẽ biểu đồ Scree Plot, Biplot phân tán 2D.

---

### 6.2. Khi đổi Dataset trong [C1_LDA.ipynb](file:///Users/nguyenhuuhoangluan/Git_Code/Machine_Learning_PCT/Midterm/Review/C1_LDA.ipynb)
Chỉ cần sửa **Cell 2**:
```python
# ==============================================================================
# ⚙️ KHỐI CẤU HÌNH DATASET & THAM SỐ (CHỈ CẦN SỬA Ở ĐÂY)
# ==============================================================================

# 1. Đường dẫn file CSV mới
DATA_PATH = "dataset/Sampledata/iris.csv"

# 2. Danh sách đặc trưng số đầu vào (Để None để TỰ ĐỘNG LẤY TẤT CẢ các cột số trong dataset, không giới hạn!)
FEATURE_COLS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]  # hoặc None

# 3. Tên cột nhãn mục tiêu phân lớp (BẮT BUỘC vì LDA là Supervised)
TARGET_COL = "species"

# 4. Số trục phân biệt cần giữ lại (LƯU Ý QUAN TRỌNG: k <= C - 1 với C là số lớp)
# Ví dụ: Iris có 3 loài (C = 3) => N_COMPONENTS tối đa là 2
N_COMPONENTS = 2
```
*Các Cell sau sẽ tự động thực hiện chuẩn 5 bước:*
- **Bước 1:** Tính tâm từng lớp $\mu_c$ và ma trận trong lớp $S_W = \sum S_c$.
- **Bước 2:** Tính tâm toàn cục $\mu$ và ma trận giữa các lớp $S_B = \sum N_c (\mu_c - \mu)(\mu_c - \mu)^T$.
- **Bước 3:** Giải trị riêng suy rộng $S_W^{-1} S_B w = \lambda w$ (có dùng `pinv` chống ma trận suy biến).
- **Bước 4:** Lấy $k$ vector riêng lớn nhất làm ma trận chiếu $W$.
- **Bước 5:** Chiếu dữ liệu $Y = X W$ và trực quan hóa phân biệt các lớp.

---

# 7. CHECKLIST GHI NHỚ NHANH TRƯỚC KHI VÀO PHÒNG THI

1. **PCA là Unsupervised, LDA là Supervised.**
2. **PCA tìm hướng phương sai lớn nhất:**
   - Dùng ma trận hiệp phương sai $S = \frac{1}{N-1}\bar{X}^T \bar{X}$.
   - Giải phương trình đặc trưng $\det(S - \lambda I) = 0$.
   - SVD: $\bar{X} = U \Sigma V^T \implies$ vector riêng chính là các cột của $V$, trị riêng $\lambda_i = \sigma_i^2 / (N-1)$.
3. **LDA tìm hướng phân tách lớp tối đa:**
   - Dùng 2 ma trận: $S_W$ (tán xạ trong lớp) và $S_B$ (tán xạ giữa các lớp).
   - Tối đa hóa tỷ số Fisher $\frac{w^T S_B w}{w^T S_W w} \implies S_W^{-1} S_B w = \lambda w$.
   - **Giới hạn số chiều:** Tối đa $k = C - 1$ chiều (với $C$ là số lớp mục tiêu).
4. **Quy tắc nhân ma trận chiếu:**
   - Nếu $X$ có kích thước $N \times D$ (mỗi dòng là 1 mẫu): $Y = X W$ (với $W$ kích thước $D \times k$).
   - Nếu $x$ là vector cột $D \times 1$: $y = W^T x$ (với $W = [w_1, w_2, \dots, w_k]$).
