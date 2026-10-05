# TỔNG ÔN TRỌNG TÂM: K-MEANS, FUZZY C-MEANS VÀ K-NN
> **Học phần:** Học máy và ứng dụng (Machine Learning & Applications)  
> **Chương 3:** Học không giám sát - Phân cụm dữ liệu (K-Means, Fuzzy C-Means)  
> **Chương 4:** Học có giám sát (2) - K láng giềng gần nhất (K-Nearest Neighbors)  
> **Tài liệu tham khảo chính:** Slide bài giảng PGS. TS. Phạm Công Thắng - ĐH Bách Khoa Đà Nẵng  
> **Notebook thực hành liên quan:**  
> - [C2_K_means.ipynb](file:///Users/nguyenhuuhoangluan/Git_Code/Machine_Learning_PCT/Midterm/Review/C2_K_means.ipynb)  
> - [C2_Fuzzy_C_Means.ipynb](file:///Users/nguyenhuuhoangluan/Git_Code/Machine_Learning_PCT/Midterm/Review/C2_Fuzzy_C_Means.ipynb)  
> - [C2_KNN.ipynb](file:///Users/nguyenhuuhoangluan/Git_Code/Machine_Learning_PCT/Midterm/Review/C2_KNN.ipynb)

---

## MỤC LỤC
1. [Bảng so sánh tổng quan K-Means vs Fuzzy C-Means vs K-NN](#1-bảng-so-sánh-tổng-quan-k-means-vs-fuzzy-c-means-vs-k-nn)
2. [Thuật toán K-Means Clustering](#2-thuật-toán-k-means-clustering)
   - 2.1. Bản chất & Hàm mục tiêu WCSS
   - 2.2. Quy trình thuật toán Lloyd 4 bước
   - 2.3. Các độ đo khoảng cách phổ biến
   - 2.4. Phương pháp xác định số cụm tối ưu $K$ (Elbow & Silhouette)
3. [Thuật toán Fuzzy C-Means (FCM) Clustering](#3-thuật-toán-fuzzy-c-means-fcm-clustering)
   - 3.1. Bản chất phân cụm mờ & Ràng buộc ma trận độ thuộc $U$
   - 3.2. Hàm mục tiêu mờ $J_m$
   - 3.3. Công thức cập nhật tâm cụm $v_k$ và ma trận độ thuộc $u_{ik}$
   - 3.4. Tham số mờ hóa $m$ & Tiêu chí đánh giá FPC
4. [Thuật toán K-Nearest Neighbors (K-NN) Classifier](#4-thuật-toán-k-nearest-neighbors-k-nn-classifier)
   - 4.1. Bản chất thuật toán "Lười học" (Lazy Learner)
   - 4.2. Quy trình phân lớp 5 bước
   - 4.3. Các quy tắc bỏ phiếu (Majority Voting vs Distance-Weighted Voting)
   - 4.4. Cách chọn $K$, xử lý hòa phiếu (Tie-breaking) & Tầm quan trọng của chuẩn hóa
5. [Hướng dẫn giải tính tay chi tiết từng bước (Step-by-step Hand Calculation)](#5-hướng-dẫn-giải-tính-tay-chi-tiết-từng-bước)
   - 5.1. Bài toán tính tay K-Means (2 cụm trong $\mathbb{R}^1$)
   - 5.2. Bài toán tính tay Fuzzy C-Means (2 cụm trong $\mathbb{R}^1$, tính 1 vòng lặp)
   - 5.3. Bài toán tính tay K-NN (Phân loại điểm mới trong $\mathbb{R}^2$)
6. [Hướng dẫn cấu hình và áp dụng code Notebook khi đổi Dataset](#6-hướng-dẫn-cấu-hình-và-áp-dụng-code-notebook-khi-đổi-dataset)
   - 6.1. Đổi Dataset trong `C2_K_means.ipynb`
   - 6.2. Đổi Dataset trong `C2_Fuzzy_C_Means.ipynb`
   - 6.3. Đổi Dataset trong `C2_KNN.ipynb`
7. [Checklist công thức nhớ nhanh trước khi vào phòng thi](#7-checklist-công-thức-nhớ-nhanh-trước-khi-vào-phòng-thi)

---

# 1. BẢNG SO SÁNH TỔNG QUAN K-MEANS VS FUZZY C-MEANS VS K-NN

| Tiêu chí | K-Means Clustering | Fuzzy C-Means (FCM) | K-Nearest Neighbors (K-NN) |
| :--- | :--- | :--- | :--- |
| **Loại học máy** | **Học không giám sát** (Unsupervised) | **Học không giám sát** (Unsupervised) | **Học có giám sát** (Supervised) |
| **Mục đích** | Phân nhóm dữ liệu thành các cụm rời rạc | Phân nhóm dữ liệu với độ mờ (xác suất) | Dự đoán nhãn phân lớp (hoặc giá trị hồi quy) |
| **Nhãn lớp ($y$)** | **Không dùng** nhãn | **Không dùng** nhãn | **Bắt buộc có** nhãn để bỏ phiếu |
| **Cách gán cụm / lớp** | **Phân cụm cứng (Hard Clustering):** Mỗi điểm chỉ thuộc đúng 1 cụm ($c_i \in \{1, \dots, K\}$) | **Phân cụm mềm (Soft / Fuzzy Clustering):** Mỗi điểm thuộc nhiều cụm với độ thuộc $u_{ik} \in [0, 1]$ | **Gán nhãn theo láng giềng:** Dựa trên nhãn của $K$ điểm huấn luyện gần nhất |
| **Pha huấn luyện** | Lặp cập nhật tâm cụm và gán cụm cho đến khi hội tụ | Lặp cập nhật ma trận độ thuộc $U$ và tâm cụm $V$ cho đến khi hội tụ | **Không có pha huấn luyện tường minh** ("Lazy Learner" - chỉ lưu tập dữ liệu mẫu) |
| **Hàm mục tiêu / Tiêu chuẩn** | $\min J = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$ (WCSS) | $\min J_m = \sum_{i=1}^N \sum_{k=1}^C u_{ik}^m \|x_i - v_k\|^2$ | Bỏ phiếu bầu (đa số hoặc có trọng số khoảng cách) |
| **Tham số then chốt** | Số cụm $K$ | Số cụm $C$, tham số mờ hóa $m$ ($m > 1$) | Số láng giềng $K$ (thường là số lẻ) |
| **Độ nhạy cảm** | Nhạy với điểm ngoại lai (outliers) và khởi tạo tâm cụm ban đầu | Tốt hơn K-Means khi dữ liệu chồng chéo, nhưng vẫn nhạy với ngoại lai | Nhạy với thang đo đặc trưng (bắt buộc chuẩn hóa) và dữ liệu nhiễu khi $K$ nhỏ |

---

# 2. THUẬT TOÁN K-MEANS CLUSTERING

### 2.1. Bản chất & Hàm mục tiêu WCSS (Slide 7, 8, 9)
- K-Means là thuật toán phân cụm không phân cấp (partitional clustering), thuộc nhóm **Hard Clustering**: mỗi điểm dữ liệu $x_i$ chỉ được gán duy nhất vào một cụm.
- **Hàm mục tiêu WCSS (Within-Cluster Sum of Squares / Inertia):**
  $$J = \text{WCSS} = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|_2^2$$
  Trong đó:
  - $K$ là số cụm định trước.
  - $C_k$ là tập các điểm thuộc cụm thứ $k$.
  - $\mu_k$ là vector tọa độ tâm (trung bình) của cụm $k$.
- **Mục tiêu:** Tìm phân hoạch các cụm $\{C_1, \dots, C_K\}$ và các tâm $\{\mu_1, \dots, \mu_K\}$ sao cho $J$ đạt giá trị nhỏ nhất.

### 2.2. Quy trình thuật toán Lloyd 4 bước (Slide 9)
1. **Khởi tạo (Initialization):** Chọn ngẫu nhiên $K$ điểm dữ liệu làm $K$ tâm cụm ban đầu $\mu_1^{(0)}, \mu_2^{(0)}, \dots, \mu_K^{(0)}$.
2. **Gán cụm (Cluster Assignment - Bước E):**
   Gán mỗi điểm dữ liệu $x_i$ vào cụm có tâm gần nó nhất:
   $$c_i = \arg\min_{k \in \{1, \dots, K\}} \|x_i - \mu_k\|_2$$
3. **Cập nhật tâm cụm (Centroid Update - Bước M):**
   Tính lại tọa độ tâm của từng cụm bằng trung bình cộng tọa độ các điểm thuộc cụm đó:
   $$\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$
4. **Hội tụ (Convergence):**
   Lặp lại Bước 2 và Bước 3 cho đến khi:
   - Tâm các cụm không thay đổi vị trí: $\max_k \|\mu_k^{(t+1)} - \mu_k^{(t)}\| < \epsilon$.
   - Hoặc gán cụm của tất cả các điểm không còn thay đổi.
   - Hoặc đạt số vòng lặp tối đa (`max_iter`).

### 2.3. Các độ đo khoảng cách phổ biến
- **Khoảng cách Euclidean ($L_2$ norm):**
  $$d_E(x_i, \mu_k) = \sqrt{\sum_{j=1}^D (x_{ij} - \mu_{kj})^2}$$
- **Khoảng cách Manhattan ($L_1$ norm):**
  $$d_M(x_i, \mu_k) = \sum_{j=1}^D |x_{ij} - \mu_{kj}|$$

### 2.4. Phương pháp xác định số cụm tối ưu $K$
1. **Phương pháp khuỷu tay (Elbow Method):**
   - Vẽ đồ thị hàm mục tiêu $J$ (WCSS / Inertia) theo các giá trị $K = 1, 2, 3, \dots$
   - Giá trị $K$ tối ưu thường nằm tại điểm "khuỷu tay" (Elbow point) — nơi mà tốc độ suy giảm của WCSS bắt đầu chững lại đáng kể.
2. **Hệ số Silhouette (Silhouette Score):**
   - Với mỗi điểm $i$, tính:
     - $a(i)$: khoảng cách trung bình từ $i$ đến các điểm khác trong cùng cụm.
     - $b(i)$: khoảng cách trung bình từ $i$ đến các điểm thuộc cụm lân cận gần nhất.
     $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, 1]$$
   - Điểm càng gần $+1$ nghĩa là điểm nằm rất sâu trong cụm và cách xa cụm khác; chọn $K$ có Silhouette Score trung bình cao nhất.

---

# 3. THUẬT TOÁN FUZZY C-MEANS (FCM) CLUSTERING

### 3.1. Bản chất phân cụm mờ & Ràng buộc ma trận độ thuộc $U$ (Slide 11, 12, 13)
- Trong thực tế, ranh giới giữa các nhóm dữ liệu thường chồng chéo nhau. Một mẫu có thể mang đặc tính của nhiều cụm.
- Fuzzy C-Means (FCM) mở rộng K-Means bằng cách cho phép mỗi điểm $x_i$ thuộc về cụm $k$ với một **độ thuộc mờ (membership degree)** $u_{ik} \in [0, 1]$.
- **Ma trận độ thuộc $U = [u_{ik}] \in \mathbb{R}^{N \times C}$ phải thỏa mãn 2 ràng buộc:**
  $$\sum_{k=1}^C u_{ik} = 1, \quad \forall i = 1, \dots, N \qquad (\text{Tổng độ thuộc vào các cụm của mỗi điểm bằng 1})$$
  $$0 < \sum_{i=1}^N u_{ik} < N, \quad \forall k = 1, \dots, C \qquad (\text{Không có cụm nào rỗng hay chiếm trọn tập dữ liệu})$$

### 3.2. Hàm mục tiêu mờ $J_m$ (Slide 13)
$$J_m(U, V) = \sum_{i=1}^N \sum_{k=1}^C u_{ik}^m \|x_i - v_k\|^2$$
Trong đó:
- $N$ là tổng số mẫu dữ liệu, $C$ là số cụm mờ.
- $u_{ik}$ là độ thuộc của mẫu $x_i$ vào cụm $k$.
- $v_k$ là vector tâm mờ của cụm thứ $k$.
- $m > 1$ là **tham số mờ hóa (fuzzifier)**; chuẩn mực thường chọn $m = 2.0$. Khi $m \to 1^+$, FCM suy biến về thuật toán K-Means cứng.

### 3.3. Công thức cập nhật tâm cụm $v_k$ và ma trận độ thuộc $u_{ik}$ (Slide 13)
Bằng phương pháp tối ưu có ràng buộc (nhân tử Lagrange), nghiệm tối ưu tại mỗi vòng lặp là:
1. **Cập nhật tâm cụm $v_k$:**
   $$v_k = \frac{\sum_{i=1}^N u_{ik}^m x_i}{\sum_{i=1}^N u_{ik}^m}$$
   *(Tâm cụm mờ chính là trung bình có trọng số mũ $m$ của toàn bộ các điểm dữ liệu).*
2. **Cập nhật ma trận độ thuộc $u_{ik}$:**
   $$u_{ik} = \frac{1}{\sum_{j=1}^C \left(\frac{\|x_i - v_k\|}{\|x_i - v_j\|}\right)^{\frac{2}{m-1}}}$$
   *(Độ thuộc tỷ lệ nghịch với khoảng cách: điểm càng gần tâm $v_k$ thì độ thuộc vào cụm $k$ càng lớn).*
3. **Điều kiện hội tụ:**
   Thuật toán dừng khi mức dịch chuyển lớn nhất của tâm cụm nhỏ hơn ngưỡng $\epsilon$:
   $$\max_{k=1,\dots,C} \|v_k^{(t+1)} - v_k^{(t)}\| < \epsilon \quad \text{hoặc} \quad \|U^{(t+1)} - U^{(t)}\|_\infty < \epsilon$$

### 3.4. Hệ số phân hoạch mờ FPC (Fuzzy Partition Coefficient)
- Dùng để đánh giá chất lượng phân cụm mờ và chọn số cụm $C$:
  $$\text{FPC} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^C u_{ik}^2 \in \left[\frac{1}{C}, 1\right]$$
- $\text{FPC} = 1$: phân cụm đạt độ phân định tuyệt đối như phân cụm cứng.
- $\text{FPC} = 1/C$: mờ hoàn toàn (mọi điểm có độ thuộc bằng nhau vào tất cả các cụm). Chọn $C$ tối ưu tại đỉnh cao nhất của FPC.

---

# 4. THUẬT TOÁN K-NEAREST NEIGHBORS (K-NN) CLASSIFIER

### 4.1. Bản chất thuật toán "Lười học" (Lazy Learner - Slide 4, 5)
- K-NN là thuật toán **Học có giám sát (Supervised Learning)** phi tham số (Non-parametric):
  - Không giả định trước phân phối xác suất của dữ liệu (như phân phối chuẩn, Poisson,...).
  - Không có giai đoạn huấn luyện để tìm hàm trọng số tối ưu (như Linear Regression, SVM hay Neural Network).
  - Toàn bộ tập huấn luyện được lưu trữ trong bộ nhớ. Khi có một mẫu mới cần dự đoán, mô hình mới bắt đầu tính khoảng cách và truy vấn $K$ lân cận. Do đó được gọi là **Instance-based Learning** hoặc **Lazy Learner**.

### 4.2. Quy trình phân lớp 5 bước (Slide 8, 9)
1. **Bước 1:** Chọn số láng giềng $K$ (thường là số nguyên lẻ: $K = 1, 3, 5, 7, \dots$).
2. **Bước 2:** Tính khoảng cách từ điểm kiểm tra $x$ đến **tất cả** các điểm dữ liệu $x_i$ trong tập huấn luyện ($i = 1, \dots, N$).
3. **Bước 3:** Sắp xếp khoảng cách theo thứ tự tăng dần và chọn ra $K$ điểm có khoảng cách nhỏ nhất (gọi là tập $N_K(x)$).
4. **Bước 4:** Thống kê nhãn của $K$ điểm láng giềng này.
5. **Bước 5:** Áp dụng quy tắc bỏ phiếu để gán nhãn dự đoán $\hat{y}$ cho điểm $x$.

### 4.3. Các quy tắc bỏ phiếu (Voting Rules)
- **Quy tắc bỏ phiếu đa số (Majority Voting):**
  Lớp nào chiếm số lượng láng giềng nhiều nhất trong top $K$ thì điểm $x$ được gán nhãn lớp đó:
  $$\hat{y} = \arg\max_{c} \sum_{i \in N_K(x)} \mathbb{I}(y_i = c)$$
- **Quy tắc bỏ phiếu có trọng số nghịch đảo khoảng cách (Distance-Weighted Voting):**
  Điểm láng giềng ở càng gần điểm $x$ thì tiếng nói (trọng số) càng lớn, giảm thiểu ảnh hưởng của các láng giềng ở xa:
  $$w_i = \frac{1}{d(x, x_i) + \epsilon} \implies \hat{y} = \arg\max_{c} \sum_{i \in N_K(x)} w_i \cdot \mathbb{I}(y_i = c)$$

### 4.4. Cách chọn $K$, xử lý hòa phiếu & Chuẩn hóa dữ liệu (Slide 10, 11)
- **Ảnh hưởng của tham số $K$:**
  - $K$ quá nhỏ (ví dụ $K=1$): Mô hình rất nhạy cảm với nhiễu và điểm ngoại lai $\implies$ **Dễ bị Overfitting**.
  - $K$ quá lớn: Vùng láng giềng bao trùm nhiều lớp khác, quyết định phân lớp bị san phẳng bởi lớp chiếm đa số $\implies$ **Dễ bị Underfitting**.
  - Chọn $K$ tối ưu thông qua **Cross-Validation (Kiểm định chéo K-Fold)**.
- **Xử lý hòa phiếu (Tie-breaking):**
  Nếu 2 lớp có số phiếu bầu bằng nhau:
  1. Dùng tổng trọng số khoảng cách $W_c = \sum \frac{1}{d(x, x_i)}$ của từng lớp để quyết định.
  2. Nếu vẫn hòa: Dùng quy tắc 1-NN (lớp của điểm gần nhất tuyệt đối sẽ thắng).
- **Tầm quan trọng của chuẩn hóa dữ liệu:**
  Vì K-NN dựa hoàn toàn vào khoảng cách Euclidean, nếu một đặc trưng có thang đo lớn (ví dụ: Thu nhập $10,000,000$) sẽ lấn át hoàn toàn đặc trưng có thang đo nhỏ (ví dụ: Số con $1 - 3$). Do đó **bắt buộc phải chuẩn hóa dữ liệu (Z-Score hoặc Min-Max)** trước khi chạy K-NN.

---

# 5. HƯỚNG DẪN GIẢI TÍNH TAY CHI TIẾT TỪNG BƯỚC

---

## 5.1. BÀI TOÁN TÍNH TAY K-MEANS

**Đề bài:** Cho 4 điểm dữ liệu 1 chiều:
$$x_1 = 2, \quad x_2 = 4, \quad x_3 = 10, \quad x_4 = 12$$
Áp dụng K-Means với $K = 2$. Giả sử khởi tạo tâm ban đầu: $\mu_1^{(0)} = 2, \quad \mu_2^{(0)} = 10$.
Thực hiện các bước lặp của thuật toán cho đến khi hội tụ.

---

### Vòng lặp 1:
- **Bước 1: Tính khoảng cách và gán cụm:**
  Khoảng cách từ các điểm tới 2 tâm $\mu_1 = 2, \mu_2 = 10$:
  - Điểm $x_1 = 2$: $d(2, 2) = 0, \quad d(2, 10) = 8 \implies \min = 0 \to \text{Cụm 1}$
  - Điểm $x_2 = 4$: $d(4, 2) = 2, \quad d(4, 10) = 6 \implies \min = 2 \to \text{Cụm 1}$
  - Điểm $x_3 = 10$: $d(10, 2) = 8, \quad d(10, 10) = 0 \implies \min = 0 \to \text{Cụm 2}$
  - Điểm $x_4 = 12$: $d(12, 2) = 10, \quad d(12, 10) = 2 \implies \min = 2 \to \text{Cụm 2}$

  Phân hoạch cụm vòng 1:
  $$C_1 = \{x_1, x_2\} = \{2, 4\}, \qquad C_2 = \{x_3, x_4\} = \{10, 12\}$$

- **Bước 2: Cập nhật lại tâm cụm:**
  $$\mu_1^{(1)} = \frac{2 + 4}{2} = 3$$
  $$\mu_2^{(1)} = \frac{10 + 12}{2} = 11$$

---

### Vòng lặp 2:
- **Tính khoảng cách với tâm mới $\mu_1 = 3, \mu_2 = 11$:**
  - Điểm $x_1 = 2$: $d(2, 3) = 1, \quad d(2, 11) = 9 \implies \text{Cụm 1}$
  - Điểm $x_2 = 4$: $d(4, 3) = 1, \quad d(4, 11) = 7 \implies \text{Cụm 1}$
  - Điểm $x_3 = 10$: $d(10, 3) = 7, \quad d(10, 11) = 1 \implies \text{Cụm 2}$
  - Điểm $x_4 = 12$: $d(12, 3) = 9, \quad d(12, 11) = 1 \implies \text{Cụm 2}$

  Phân hoạch cụm vòng 2:
  $$C_1 = \{2, 4\}, \qquad C_2 = \{10, 12\}$$

- **Kiểm tra điều kiện dừng:**
  Phân hoạch cụm ở vòng 2 hoàn toàn trùng khớp với vòng 1, tâm cụm không thay đổi ($\mu_1 = 3, \mu_2 = 11$).
  $\Rightarrow$ **Thuật toán hội tụ!**

- **Tính hàm mục tiêu WCSS cuối cùng:**
  $$J = \left[(2 - 3)^2 + (4 - 3)^2\right] + \left[(10 - 11)^2 + (12 - 11)^2\right] = (1 + 1) + (1 + 1) = 4$$

---

## 5.2. BÀI TOÁN TÍNH TAY FUZZY C-MEANS (FCM)

**Đề bài:** Cho 3 điểm dữ liệu 1 chiều:
$$x_1 = 1, \quad x_2 = 5, \quad x_3 = 9$$
Phân cụm mờ với $C = 2$ cụm, tham số mờ $m = 2.0$.
Giả sử tại thời điểm bắt đầu vòng lặp, ta có 2 tâm cụm:
$$v_1 = 2, \quad v_2 = 8$$
Hãy tính:
1. Ma trận độ thuộc $U$ của 3 điểm đối với 2 cụm.
2. Tọa độ tâm cụm mới $v_1^{(1)}, v_2^{(1)}$ sau 1 vòng cập nhật.

---

### Bước 1: Tính ma trận khoảng cách $d_{ik} = |x_i - v_k|$
- Với $x_1 = 1$: $d_{11} = |1 - 2| = 1, \quad d_{12} = |1 - 8| = 7$
- Với $x_2 = 5$: $d_{21} = |5 - 2| = 3, \quad d_{22} = |5 - 8| = 3$
- Với $x_3 = 9$: $d_{31} = |9 - 2| = 7, \quad d_{32} = |9 - 8| = 1$

---

### Bước 2: Tính ma trận độ thuộc $U$
Công thức với $m = 2 \implies \frac{2}{m - 1} = \frac{2}{2 - 1} = 2$:
$$u_{ik} = \frac{1}{\sum_{j=1}^2 \left(\frac{d_{ik}}{d_{ij}}\right)^2}$$

- **Điểm $x_1 = 1$:**
  $$u_{11} = \frac{1}{\left(\frac{1}{1}\right)^2 + \left(\frac{1}{7}\right)^2} = \frac{1}{1 + \frac{1}{49}} = \frac{49}{50} = 0.98$$
  $$u_{12} = 1 - u_{11} = 0.02$$

- **Điểm $x_2 = 5$ (nằm chính giữa hai tâm):**
  $$u_{21} = \frac{1}{\left(\frac{3}{3}\right)^2 + \left(\frac{3}{3}\right)^2} = \frac{1}{1 + 1} = 0.50$$
  $$u_{22} = 1 - u_{21} = 0.50$$

- **Điểm $x_3 = 9$:**
  $$u_{31} = \frac{1}{\left(\frac{7}{7}\right)^2 + \left(\frac{7}{1}\right)^2} = \frac{1}{1 + 49} = \frac{1}{50} = 0.02$$
  $$u_{32} = 1 - u_{31} = 0.98$$

Ma trận độ thuộc thu được:
$$U = \begin{bmatrix} 0.98 & 0.02 \\ 0.50 & 0.50 \\ 0.02 & 0.98 \end{bmatrix}$$

---

### Bước 3: Cập nhật tọa độ tâm cụm mới $v_k$ (với $m = 2 \implies u_{ik}^2$)
- Bình phương các độ thuộc:
  - Cụm 1: $u_{11}^2 = 0.98^2 = 0.9604, \quad u_{21}^2 = 0.50^2 = 0.25, \quad u_{31}^2 = 0.02^2 = 0.0004$
    $$\sum_{i=1}^3 u_{i1}^2 = 0.9604 + 0.25 + 0.0004 = 1.2108$$
  - Cụm 2: $u_{12}^2 = 0.0004, \quad u_{22}^2 = 0.25, \quad u_{32}^2 = 0.9604$
    $$\sum_{i=1}^3 u_{i2}^2 = 1.2108$$

- **Tọa độ tâm mới $v_1^{(1)}$:**
  $$v_1^{(1)} = \frac{u_{11}^2 x_1 + u_{21}^2 x_2 + u_{31}^2 x_3}{\sum u_{i1}^2} = \frac{0.9604(1) + 0.25(5) + 0.0004(9)}{1.2108} = \frac{0.9604 + 1.25 + 0.0036}{1.2108} = \frac{2.2140}{1.2108} \approx 1.8285$$

- **Tọa độ tâm mới $v_2^{(1)}$:**
  $$v_2^{(1)} = \frac{u_{12}^2 x_1 + u_{22}^2 x_2 + u_{32}^2 x_3}{\sum u_{i2}^2} = \frac{0.0004(1) + 0.25(5) + 0.9604(9)}{1.2108} = \frac{0.0004 + 1.25 + 8.6436}{1.2108} = \frac{9.8940}{1.2108} \approx 8.1715$$

> **Nhận xét:** Tâm $v_1$ dịch chuyển từ $2 \to 1.83$, tâm $v_2$ dịch chuyển từ $8 \to 8.17$, phản ánh đúng bản chất hút về phía các điểm dữ liệu mờ.

---

## 5.3. BÀI TOÁN TÍNH TAY K-NEAREST NEIGHBORS (K-NN)

**Đề bài:** Cho tập dữ liệu huấn luyện gồm 5 điểm trong không gian 2 chiều với nhãn lớp thuộc hai loại `A` và `B`:
- $x_1 = (1, 2) \to \text{Lớp A}$
- $x_2 = (2, 3) \to \text{Lớp A}$
- $x_3 = (3, 1) \to \text{Lớp A}$
- $x_4 = (6, 5) \to \text{Lớp B}$
- $x_5 = (7, 7) \to \text{Lớp B}$

Có một điểm kiểm tra mới cần phân lớp: $x_{test} = (3, 3)$.
Áp dụng thuật toán K-NN với $K = 3$ (khoảng cách Euclidean):
1. Tìm $K=3$ láng giềng gần nhất.
2. Dự đoán nhãn theo quy tắc Bỏ phiếu đa số (Majority Voting).
3. Dự đoán nhãn theo quy tắc Bỏ phiếu có trọng số (Distance-Weighted Voting).

---

### Bước 1: Tính khoảng cách Euclidean từ $x_{test}(3, 3)$ đến tất cả các điểm
Công thức: $d(x, x_i) = \sqrt{(x_1 - x_{i1})^2 + (x_2 - x_{i2})^2}$

- Điểm $x_1(1, 2)$: $d_1 = \sqrt{(3 - 1)^2 + (3 - 2)^2} = \sqrt{2^2 + 1^2} = \sqrt{5} \approx 2.236$
- Điểm $x_2(2, 3)$: $d_2 = \sqrt{(3 - 2)^2 + (3 - 3)^2} = \sqrt{1^2 + 0^2} = \sqrt{1} = 1.000$
- Điểm $x_3(3, 1)$: $d_3 = \sqrt{(3 - 3)^2 + (3 - 1)^2} = \sqrt{0^2 + 2^2} = \sqrt{4} = 2.000$
- Điểm $x_4(6, 5)$: $d_4 = \sqrt{(3 - 6)^2 + (3 - 5)^2} = \sqrt{(-3)^2 + (-2)^2} = \sqrt{9 + 4} = \sqrt{13} \approx 3.606$
- Điểm $x_5(7, 7)$: $d_5 = \sqrt{(3 - 7)^2 + (3 - 7)^2} = \sqrt{(-4)^2 + (-4)^2} = \sqrt{16 + 16} = \sqrt{32} \approx 5.657$

---

### Bước 2: Sắp xếp khoảng cách và chọn Top $K = 3$ láng giềng
Bảng sắp xếp tăng dần theo khoảng cách:

| Thứ tự gần | Điểm mẫu | Tọa độ | Khoảng cách $d$ | Nhãn thực tế |
| :---: | :---: | :---: | :---: | :---: |
| **1st NN** | $x_2$ | $(2, 3)$ | **1.000** | **Lớp A** |
| **2nd NN** | $x_3$ | $(3, 1)$ | **2.000** | **Lớp A** |
| **3rd NN** | $x_1$ | $(1, 2)$ | **2.236** | **Lớp A** |
| 4th | $x_4$ | $(6, 5)$ | 3.606 | Lớp B |
| 5th | $x_5$ | $(7, 7)$ | 5.657 | Lớp B |

$\Rightarrow$ Tập 3 láng giềng gần nhất là: $\{x_2, x_3, x_1\}$.

---

### Bước 3: Dự đoán nhãn
- **Cách 1: Bỏ phiếu đa số (Majority Voting):**
  - Số phiếu lớp A: $3$ phiếu ($x_2, x_3, x_1$).
  - Số phiếu lớp B: $0$ phiếu.
  $$\hat{y}_{test} = \text{Lớp A}$$

- **Cách 2: Bỏ phiếu có trọng số ($w_i = 1 / d_i$):**
  - Trọng số $x_2$: $w_2 = \frac{1}{1.000} = 1.000$
  - Trọng số $x_3$: $w_3 = \frac{1}{2.000} = 0.500$
  - Trọng số $x_1$: $w_1 = \frac{1}{2.236} \approx 0.447$
  
  Tổng trọng số:
  - Tổng trọng số lớp A: $W_A = 1.000 + 0.500 + 0.447 = 1.947$
  - Tổng trọng số lớp B: $W_B = 0$
  $$\hat{y}_{test} = \text{Lớp A}$$

---

# 6. HƯỚNG DẪN CẤU HÌNH VÀ ÁP DỤNG CODE NOTEBOOK KHI ĐỔI DATASET

Cả 3 notebook đều được thiết kế theo kiến trúc chuẩn modular: **toàn bộ thuật toán chạy từ đầu (From Scratch)** và **chỉ cần sửa duy nhất Cell 2 (Khối cấu hình)** khi chuyển sang dataset mới.

### 6.1. Đổi Dataset trong [`C2_K_means.ipynb`](file:///Users/nguyenhuuhoangluan/Git_Code/Machine_Learning_PCT/Midterm/Review/C2_K_means.ipynb)
Mở **Cell 2** và chỉnh sửa:
```python
# 1. Đường dẫn file dữ liệu mới
DATA_PATH = "dataset/Sampledata/iris.csv"

# 2. Danh sách cột đặc trưng số đầu vào (để None nếu muốn tự động lấy các cột số)
FEATURE_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# 3. Số cụm K mong muốn (đặt số cụ thể như 3, hoặc đặt None để tự chọn K theo đỉnh Silhouette)
CHOSEN_K = 3

# 4. Độ đo khoảng cách ('euclidean' hoặc 'manhattan')
METRIC = "euclidean"

# 5. Dải số cụm K khảo sát đồ thị Elbow và Silhouette
K_SEARCH_RANGE = range(2, 9)
```

---

### 6.2. Đổi Dataset trong [`C2_Fuzzy_C_Means.ipynb`](file:///Users/nguyenhuuhoangluan/Git_Code/Machine_Learning_PCT/Midterm/Review/C2_Fuzzy_C_Means.ipynb)
Mở **Cell 2** và chỉnh sửa:
```python
# 1. Đường dẫn file dữ liệu mới
DATA_PATH = "dataset/Sampledata/iris.csv"

# 2. Danh sách cột đặc trưng số đầu vào (để None nếu muốn tự động lấy các cột số)
FEATURE_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# 3. Số cụm mờ C (đặt số cụ thể như 3, hoặc đặt None để tự chọn C theo đỉnh FPC)
CHOSEN_C = 3

# 4. Tham số mờ hóa m (chuẩn mực chọn 2.0)
FUZZIFIER_M = 2.0

# 5. Dải số cụm C khảo sát đồ thị FPC
C_SEARCH_RANGE = range(2, 9)
```

---

### 6.3. Đổi Dataset trong [`C2_KNN.ipynb`](file:///Users/nguyenhuuhoangluan/Git_Code/Machine_Learning_PCT/Midterm/Review/C2_KNN.ipynb)
Mở **Cell 2** và chỉnh sửa:
```python
# 1. Đường dẫn file dữ liệu mới
DATA_PATH = "dataset/Sampledata/iris.csv"

# 2. Danh sách cột đặc trưng số đầu vào (để None nếu muốn tự động lấy các cột số)
FEATURE_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# 3. Cột nhãn mục tiêu cần phân lớp (BẮT BUỘC có vì K-NN là Supervised)
TARGET_COL = "species"

# 4. Số láng giềng K (chọn số nguyên lẻ cụ thể như 3, 5, 7 hoặc đặt None để tự chọn qua CV)
CHOSEN_K = 5

# 5. Độ đo khoảng cách ('euclidean' hoặc 'manhattan')
METRIC = "euclidean"

# 6. Tỷ lệ tập kiểm thử Test Set
TEST_SIZE = 0.2

# 7. Dải số láng giềng K khảo sát đồ thị Cross-Validation
K_SEARCH_RANGE = range(1, 26)
```

---

# 7. CHECKLIST CÔNG THỨC NHỚ NHANH TRƯỚC KHI VÀO PHÒNG THI

1. **K-Means:**
   - Cập nhật tâm: $\mu_k = \frac{1}{|C_k|} \sum_{x \in C_k} x$
   - Hàm mục tiêu WCSS: $J = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$
   - Chọn $K$: Điểm gãy khuỷu tay (Elbow) hoặc $\max \text{Silhouette Score}$.
2. **Fuzzy C-Means (FCM):**
   - Ràng buộc: $\sum_{k=1}^C u_{ik} = 1$
   - Cập nhật tâm mờ: $v_k = \frac{\sum_{i=1}^N u_{ik}^m x_i}{\sum_{i=1}^N u_{ik}^m}$
   - Cập nhật độ thuộc: $u_{ik} = \frac{1}{\sum_{j=1}^C \left(\frac{\|x_i - v_k\|}{\|x_i - v_j\|}\right)^{\frac{2}{m-1}}}$ (với $m = 2$, số mũ là 2).
   - Chọn $C$: Đỉnh cao nhất của hệ số $\text{FPC} = \frac{1}{N} \sum \sum u_{ik}^2$.
3. **K-Nearest Neighbors (K-NN):**
   - Không có tham số học (Lazy Learner).
   - Bắt buộc chuẩn hóa dữ liệu trước khi tính khoảng cách.
   - Bỏ phiếu đa số: $\hat{y} = \arg\max_c \sum_{i \in N_K(x)} \mathbb{I}(y_i = c)$.
   - Bỏ phiếu trọng số: $w_i = \frac{1}{d(x, x_i) + \epsilon}$.
   - $K$ nhỏ $\to$ Overfitting; $K$ lớn $\to$ Underfitting.
