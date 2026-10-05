# TỔNG ÔN TRỌNG TÂM: PHÂN ĐOẠN ẢNH MÀU BẰNG K-MEANS & K-NN
> **Học phần:** Học máy và ứng dụng (Machine Learning & Applications)  
> **Chủ đề:** Phân đoạn ảnh màu (Color Image Segmentation)  
> **Thuật toán cốt lõi:** K-Means Clustering & K-Nearest Neighbors (K-NN)  
> **Notebook thực hành liên quan:** [C3_Img_Segmentation.ipynb](file:///Users/nguyenhuuhoangluan/Git_Code/Machine_Learning_PCT/Midterm/Review/C3_Img_Segmentation.ipynb)

---

## MỤC LỤC
1. [Bản chất bài toán Phân đoạn ảnh (Image Segmentation)](#1-bản-chất-bài-toán-phân-đoạn-ảnh-image-segmentation)
2. [So sánh bản chất: Phân đoạn ảnh đen trắng (Grayscale) vs Ảnh màu (Color Image)](#2-so-sánh-bản-chất-phân-đoạn-ảnh-đen-trắng-grayscale-vs-ảnh-màu-color-image)
3. [Thuật toán 1: Phân đoạn ảnh màu bằng K-Means Clustering](#3-thuật-toán-1-phân-đoạn-ảnh-màu-bằng-k-means-clustering)
   - 3.1. Vector hóa ma trận ảnh màu $(H, W, 3) \to (N, 3)$
   - 3.2. Hàm mục tiêu WCSS & Cập nhật tâm màu (Color Palette)
   - 3.3. Tái tạo ảnh màu phân đoạn & Tách mặt nạ từng cụm
4. [Thuật toán 2: Phân đoạn ảnh màu bằng K-Nearest Neighbors (K-NN)](#4-thuật-toán-2-phân-đoạn-ảnh-màu-bằng-k-nearest-neighbors-k-nn)
   - 4.1. Cơ chế lấy mẫu hạt giống (Seed-based Supervised Segmentation)
   - 4.2. Huấn luyện bộ phân lớp K-NN trên đặc trưng màu sắc
   - 4.3. Dự đoán toàn ảnh, làm mượt mặt nạ và bóc tách đối tượng tiền cảnh (Matting/Compositing)
5. [Hướng dẫn giải tính tay chi tiết (Step-by-step Hand Calculation)](#5-hướng-dẫn-giải-tính-tay-chi-tiết-step-by-step-hand-calculation)
   - 5.1. Bài toán tính tay K-Means trên ma trận ảnh màu mini $2 \times 2$
   - 5.2. Bài toán tính tay K-NN phân loại pixel màu truy vấn
6. [Hướng dẫn cấu hình & Áp dụng trong Notebook `C3_Img_Segmentation.ipynb`](#6-hướng-dẫn-cấu-hình--áp-dụng-trong-notebook-c3_img_segmentationipynb)
7. [Checklist tổng kết khi đi thi](#7-checklist-tổng-kết-khi-đi-thi)

---

# 1. BẢN CHẤT BÀI TOÁN PHÂN ĐOẠN ẢNH (IMAGE SEGMENTATION)

- **Định nghĩa:** Phân đoạn ảnh là quá trình chia một bức ảnh kỹ thuật số thành nhiều phân vùng (segments / regions / clusters) gồm các tập hợp điểm ảnh (pixels) có chung tính chất trực quan như màu sắc, độ sáng hoặc kết cấu.
- **Mục tiêu:**
  1. Đơn giản hóa cấu trúc biểu diễn của bức ảnh để dễ phân tích hơn (nhận dạng vật thể, định vị ranh giới).
  2. Bóc tách đối tượng mục tiêu (Foreground - Tiền cảnh) ra khỏi phông nền (Background - Hậu cảnh).
  3. Nén ảnh và lượng tử hóa màu sắc (Color Quantization / Palette Extraction).

---

# 2. SO SÁNH BẢN CHẤT: PHÂN ĐOẠN ẢNH ĐEN TRẮNG VS ẢNH MÀU

Nhiều tài liệu hoặc bài mẫu sơ cấp (như trên Kaggle hay GitHub) thường thực hiện trên **ảnh đen trắng (Grayscale / MRI scans)** vì kích thước dữ liệu nhẹ. Tuy nhiên, khi chuyển sang **ảnh màu thực tế**, sự khác biệt về mặt bản chất toán học là rất lớn:

| Đặc điểm | Ảnh đen trắng (Grayscale) | Ảnh màu (Color RGB / CIELAB) |
| :--- | :--- | :--- |
| **Kênh dữ liệu** | 1 kênh duy nhất: $I(x, y) \in [0, 255]$ | 3 kênh màu: $[R(x, y), G(x, y), B(x, y)] \in [0, 255]^3$ |
| **Vector đặc trưng pixel** | Đại lượng vô hướng 1 chiều: $p_i \in \mathbb{R}^1$ | Vector đa chiều 3D: $p_i = [R, G, B]^T \in \mathbb{R}^3$ (hoặc 5D khi thêm tọa độ $[R, G, B, \lambda x, \lambda y]$) |
| **Khoảng cách giữa 2 pixel** | Hiệu đại số độ sáng: $d = \|I_1 - I_2\|$ | Khoảng cách không gian màu Euclidean 3D: $d = \sqrt{(R_1 - R_2)^2 + (G_1 - G_2)^2 + (B_1 - B_2)^2}$ |
| **Khả năng phân tách** | **Rất kém:** Hai vùng có màu sắc hoàn toàn khác nhau (ví dụ: quả táo đỏ và lá cây xanh) nhưng có cùng mức sáng (luminance) sẽ bị gộp thành 1 cụm xám giống nhau. | **Xuất sắc:** Phân biệt rõ ràng dựa trên sắc thái màu (Hue) và độ bão hòa (Saturation), tách biệt trọn vẹn đối tượng khỏi nền. |
| **Kết quả trực quan** | Ảnh phân đoạn mức xám $K$ nấc xám phẳng | Ảnh phân đoạn giữ nguyên màu sắc sống động với $K$ tâm màu đại diện (Color Palette) |

---

# 3. THUẬT TOÁN 1: PHÂN ĐOẠN ẢNH MÀU BẰNG K-MEANS CLUSTERING

### 3.1. Vector hóa ma trận ảnh màu $(H, W, 3) \to (N, 3)$
- Ảnh màu có kích thước $H \times W \times 3$.
- Ta trải phẳng ma trận 2D không gian thành tập dữ liệu gồm $N = H \times W$ điểm dữ liệu:
  $$X = \begin{bmatrix} R_1 & G_1 & B_1 \\ R_2 & G_2 & B_2 \\ \vdots & \vdots & \vdots \\ R_N & G_N & B_N \end{bmatrix} \in \mathbb{R}^{N \times 3}$$

### 3.2. Hàm mục tiêu WCSS & Cập nhật tâm màu (Color Palette)
- **Hàm mục tiêu WCSS (Within-Cluster Sum of Squares):**
  $$J = \sum_{k=1}^K \sum_{p_i \in C_k} \|p_i - \mu_k\|_2^2 = \sum_{k=1}^K \sum_{p_i \in C_k} \left[ (R_i - \mu_{k,R})^2 + (G_i - \mu_{k,G})^2 + (B_i - \mu_{k,B})^2 \right]$$
- **Bước 1 (Gán cụm):** Mỗi pixel $p_i$ chọn cụm có khoảng cách màu nhỏ nhất:
  $$c_i = \arg\min_{k \in \{1, \dots, K\}} \sqrt{(R_i - \mu_{k,R})^2 + (G_i - \mu_{k,G})^2 + (B_i - \mu_{k,B})^2}$$
- **Bước 2 (Cập nhật tâm màu):**
  $$\mu_k = \begin{bmatrix} \mu_{k,R} \\ \mu_{k,G} \\ \mu_{k,B} \end{bmatrix} = \frac{1}{|C_k|} \sum_{p_i \in C_k} \begin{bmatrix} R_i \\ G_i \\ B_i \end{bmatrix}$$
  *Mỗi tâm $\mu_k$ chính là màu trung bình của toàn bộ các pixel trong cụm đó (màu đại diện).*

### 3.3. Tái tạo ảnh phân đoạn & Tách mặt nạ từng cụm
- **Ảnh phân đoạn màu:** Thay thế mỗi pixel $p_i$ bằng màu tâm cụm $\mu_{c_i}$, sau đó reshape về $(H, W, 3)$:
  $$I_{seg}(x, y) = \mu_{c(x, y)}$$
- **Mặt nạ nhị phân của cụm thứ $k$:**
  $$M_k(x, y) = \begin{cases} 1 & \text{nếu } c(x, y) = k \\ 0 & \text{ngược lại} \end{cases}$$
- **Tỷ lệ diện tích của cụm $k$:** $\text{Ratio}_k = \frac{\sum M_k(x, y)}{H \times W} \times 100\%$.

---

# 4. THUẬT TOÁN 2: PHÂN ĐOẠN ẢNH MÀU BẰNG K-NEAREST NEIGHBORS (K-NN)

Trong các bài toán thực tế, khi cần **tách một đối tượng tiền cảnh (Foreground)** khỏi **phông nền (Background)**, thuật toán K-NN hoạt động theo cơ chế **bán giám sát dựa trên hạt giống (Seed-based Supervised Segmentation)**:

### 4.1. Cơ chế lấy mẫu hạt giống (Seed-based Sampling)
- **Hạt giống Tiền cảnh ($\mathcal{S}_{FG}$):** Các pixel thuộc vùng đối tượng (thường là vùng elip/chữ nhật trung tâm ảnh do người dùng chọn hoặc lấy mẫu tự động):
  $$p \in \mathcal{S}_{FG} \implies y = 1$$
- **Hạt giống Hậu cảnh ($\mathcal{S}_{BG}$):** Các pixel thuộc vùng phông nền (thường lấy ở 4 dải viền xung quanh mép ảnh):
  $$p \in \mathcal{S}_{BG} \implies y = 0$$
- Tập dữ liệu huấn luyện: $\mathcal{D}_{train} = \{(p_i, y_i)\}_{i=1}^M$ với $M \ll N$ (thường chọn $M \approx 1,000 - 3,000$ điểm để tối ưu tốc độ).

### 4.2. Huấn luyện bộ phân lớp K-NN trên đặc trưng màu sắc
- Với mỗi pixel $p = [R, G, B]^T$ bất kỳ trên toàn bộ bức ảnh $H \times W$:
  1. Tính khoảng cách Euclidean đến $M$ hạt giống huấn luyện.
  2. Chọn ra $K$ hạt giống gần nhất (ví dụ $K=5$).
  3. Bỏ phiếu có trọng số nghịch đảo khoảng cách ($w_i = \frac{1}{d(p, p_i) + \epsilon}$):
     $$\hat{y}(p) = \arg\max_{c \in \{0, 1\}} \sum_{i \in N_K(p)} w_i \cdot \mathbb{I}(y_i = c)$$

### 4.3. Làm mượt mặt nạ và bóc tách đối tượng tiền cảnh
1. **Làm mượt mặt nạ:** Áp dụng phép toán hình thái học (Morphological Closing & Opening) để triệt tiêu các chấm nhiễu đơn lẻ và lấp đầy các lỗ rỗng trong lòng đối tượng:
   $$M_{clean} = (M \bullet K_{elem}) \circ K_{elem}$$
2. **Bóc tách đối tượng tiền cảnh:**
   - **Nền trắng thuần:** $I_{fg}(x, y) = M_{clean}(x, y) \cdot I(x, y) + (1 - M_{clean}(x, y)) \cdot [255, 255, 255]^T$
   - **Kênh trong suốt RGBA:** Kênh Alpha $A(x, y) = M_{clean}(x, y) \times 255$.
   - **Ghép nền mới (Chroma Green Screen):** Thay thế vùng $M_{clean} = 0$ bằng màu xanh lá phông nền $[30, 220, 60]$.

---

# 5. HƯỚNG DẪN GIẢI TÍNH TAY CHI TIẾT (STEP-BY-STEP HAND CALCULATION)

---

## 5.1. BÀI TOÁN TÍNH TAY K-MEANS TRÊN ẢNH MÀU MINI $2 \times 2$

**Đề bài:** Cho một bức ảnh màu gồm 4 điểm ảnh xếp thành lưới $2 \times 2$:
- Pixel $p_1(0, 0) = [240, 20, 20]$ (Đỏ rực)
- Pixel $p_2(0, 1) = [250, 30, 30]$ (Đỏ nhạt)
- Pixel $p_3(1, 0) = [20, 30, 230]$ (Xanh dương)
- Pixel $p_4(1, 1) = [30, 40, 250]$ (Xanh dương đậm)

Áp dụng K-Means với $K = 2$ cụm màu.
Khởi tạo 2 tâm ban đầu:
$$\mu_1^{(0)} = [240, 20, 20], \qquad \mu_2^{(0)} = [20, 30, 230]$$
Thực hiện vòng lặp thứ nhất của K-Means:

---

### Bước 1: Tính khoảng cách Euclidean màu và gán cụm
Khoảng cách màu giữa $p_i = [R_i, G_i, B_i]$ và tâm $\mu = [R_\mu, G_\mu, B_\mu]$:
$$d(p_i, \mu) = \sqrt{(R_i - R_\mu)^2 + (G_i - G_\mu)^2 + (B_i - B_\mu)^2}$$

- **Pixel $p_1[240, 20, 20]$:**
  - $d(p_1, \mu_1) = \sqrt{0^2 + 0^2 + 0^2} = 0$
  - $d(p_1, \mu_2) = \sqrt{(240-20)^2 + (20-30)^2 + (20-230)^2} = \sqrt{220^2 + (-10)^2 + (-210)^2} = \sqrt{48400 + 100 + 44100} = \sqrt{92600} \approx 304.3$
  $\implies \min = 0 \to \mathbf{Cụm\ 1}$

- **Pixel $p_2[250, 30, 30]$:**
  - $d(p_2, \mu_1) = \sqrt{(250-240)^2 + (30-20)^2 + (30-20)^2} = \sqrt{10^2 + 10^2 + 10^2} = \sqrt{300} \approx 17.32$
  - $d(p_2, \mu_2) = \sqrt{(250-20)^2 + (30-30)^2 + (30-230)^2} = \sqrt{230^2 + 0 + (-200)^2} = \sqrt{52900 + 40000} \approx 304.8$
  $\implies \min = 17.32 \to \mathbf{Cụm\ 1}$

- **Pixel $p_3[20, 30, 230]$:**
  - $d(p_3, \mu_1) \approx 304.3$
  - $d(p_3, \mu_2) = 0 \implies \min = 0 \to \mathbf{Cụm\ 2}$

- **Pixel $p_4[30, 40, 250]$:**
  - $d(p_4, \mu_1) \approx 323.7$
  - $d(p_4, \mu_2) = \sqrt{(30-20)^2 + (40-30)^2 + (250-230)^2} = \sqrt{100 + 100 + 400} = \sqrt{600} \approx 24.49$
  $\implies \min = 24.49 \to \mathbf{Cụm\ 2}$

Phân hoạch cụm: $C_1 = \{p_1, p_2\}$, $C_2 = \{p_3, p_4\}$.

---

### Bước 2: Cập nhật tâm màu mới $\mu_1^{(1)}, \mu_2^{(1)}$
- **Tâm cụm 1 (Đỏ):**
  $$\mu_1^{(1)} = \frac{p_1 + p_2}{2} = \begin{bmatrix} (240 + 250)/2 \\ (20 + 30)/2 \\ (20 + 30)/2 \end{bmatrix} = \begin{bmatrix} 245 \\ 25 \\ 25 \end{bmatrix}$$
- **Tâm cụm 2 (Xanh):**
  $$\mu_2^{(1)} = \frac{p_3 + p_4}{2} = \begin{bmatrix} (20 + 30)/2 \\ (30 + 40)/2 \\ (230 + 250)/2 \end{bmatrix} = \begin{bmatrix} 25 \\ 35 \\ 240 \end{bmatrix}$$

---

### Bước 3: Tái tạo ảnh phân đoạn màu
Mỗi pixel được thay thế bằng màu của tâm cụm:
- $I_{seg}(0, 0) = [245, 25, 25]$ (Đỏ đồng nhất)
- $I_{seg}(0, 1) = [245, 25, 25]$ (Đỏ đồng nhất)
- $I_{seg}(1, 0) = [25, 35, 240]$ (Xanh đồng nhất)
- $I_{seg}(1, 1) = [25, 35, 240]$ (Xanh đồng nhất)

> **Kết luận:** Bức ảnh $2 \times 2$ được phân tách thành 2 vùng màu sắc riêng biệt hoàn hảo: nửa trên là vùng màu đỏ ($50\%$), nửa dưới là vùng màu xanh ($50\%$).

---

## 5.2. BÀI TOÁN TÍNH TAY K-NN PHÂN LOẠI PIXEL MÀU TRUY VẤN

**Đề bài:** Cho 3 hạt giống huấn luyện:
- Hạt giống 1: $s_1 = [240, 20, 20]$ (Đỏ) $\to$ **Tiền cảnh (Nhãn 1)**
- Hạt giống 2: $s_2 = [220, 40, 30]$ (Đỏ sẫm) $\to$ **Tiền cảnh (Nhãn 1)**
- Hạt giống 3: $s_3 = [30, 40, 230]$ (Xanh dương) $\to$ **Hậu cảnh (Nhãn 0)**

Có một pixel mới cần phân loại: $p_{query} = [230, 25, 25]$.
Sử dụng K-NN với $K = 3$:
1. Tính khoảng cách Euclidean đến 3 hạt giống.
2. Bỏ phiếu đa số dự đoán pixel $p_{query}$ là Tiền cảnh hay Hậu cảnh.

---

### Lời giải:
1. **Tính khoảng cách:**
   - $d(p, s_1) = \sqrt{(230-240)^2 + (25-20)^2 + (25-20)^2} = \sqrt{(-10)^2 + 5^2 + 5^2} = \sqrt{100 + 25 + 25} = \sqrt{150} \approx 12.25$
   - $d(p, s_2) = \sqrt{(230-220)^2 + (25-40)^2 + (25-30)^2} = \sqrt{10^2 + (-15)^2 + (-5)^2} = \sqrt{100 + 225 + 25} = \sqrt{350} \approx 18.71$
   - $d(p, s_3) = \sqrt{(230-30)^2 + (25-40)^2 + (25-230)^2} = \sqrt{200^2 + (-15)^2 + (-205)^2} = \sqrt{40000 + 225 + 42025} = \sqrt{82250} \approx 286.79$
2. **Top $K = 3$ láng giềng gần nhất:**
   - Hạng 1: $s_1$ ($d \approx 12.25$) $\to$ Nhãn 1 (Tiền cảnh)
   - Hạng 2: $s_2$ ($d \approx 18.71$) $\to$ Nhãn 1 (Tiền cảnh)
   - Hạng 3: $s_3$ ($d \approx 286.79$) $\to$ Nhãn 0 (Hậu cảnh)
3. **Bỏ phiếu đa số:**
   - Nhãn 1 (Tiền cảnh): 2 phiếu ($s_1, s_2$).
   - Nhãn 0 (Hậu cảnh): 1 phiếu ($s_3$).
   $$\hat{y}(p_{query}) = \mathbf{1\ (Tiền\ cảnh)}$$

---

# 6. HƯỚNG DẪN CẤU HÌNH & ÁP DỤNG TRONG NOTEBOOK `C3_Img_Segmentation.ipynb`

Toàn bộ notebook [C3_Img_Segmentation.ipynb](file:///Users/nguyenhuuhoangluan/Git_Code/Machine_Learning_PCT/Midterm/Review/C3_Img_Segmentation.ipynb) đã được chuẩn hóa lại hoàn chỉnh và kiểm tra thực thi thành công $100\%$ không lỗi.

### Khối Cấu hình tại Cell 2:
```python
# ==============================================================================
# ⚙️ KHỐI CẤU HÌNH DATASET & THAM SỐ (CHỈ CẦN SỬA Ở ĐÂY)
# ==============================================================================

# 1. Đường dẫn file ảnh màu đầu vào (PNG, JPG, JPEG)
IMAGE_PATH = "dataset/image/Siamese_161.jpg"  # hoặc 'dataset/image/cat.png'

# 2. Số lượng cụm màu sắc K-Means (Khuyến nghị: 3 đến 6 cụm)
N_CLUSTERS_KMEANS = 4

# 3. Số láng giềng K trong K-NN (Khuyến nghị số lẻ: 3, 5, 7)
KNN_N_NEIGHBORS = 5

# 4. Kích thước chuẩn hóa cạnh lớn nhất để tối ưu tốc độ xử lý (pixel)
MAX_IMAGE_DIM = 600

# 5. Tỷ lệ lấy mẫu hạt giống cho K-NN
SAMPLE_SEEDS_COUNT = 1500

# 6. Random seed
RANDOM_STATE = 42
```

### Các tính năng trực quan hóa đã được tích hợp sẵn:
1. **Cell 4:** Biểu đồ tán xạ 3D màu sắc (3D RGB Scatter Plot) trực quan hóa phân bố các pixel màu.
2. **Cell 5 & 6:** 
   - Phân đoạn ảnh màu K-Means.
   - Trích xuất bảng màu thống kê tỷ lệ diện tích (DataFrame gồm RGB, mã HEX, số lượng pixel, tỷ lệ %).
   - Vẽ thanh bảng màu đại diện (Palette Bar).
   - Tách từng lớp phân đoạn màu độc lập.
   - Khảo sát biến thiên kết quả với $K = 2, 3, 5, 8$.
3. **Cell 7 & 8:** 
   - Phân đoạn tách nền đối tượng tiền cảnh bằng K-NN.
   - Hiển thị vị trí các hạt giống (Seeds overlay).
   - Mặt nạ phân đoạn nhị phân (Binary Mask).
   - Ảnh bóc tách đối tượng tiền cảnh trên nền trắng thuần.
   - Ghép đối tượng lên phông xanh điện ảnh (Chroma Green Screen).
4. **Cell 9:** 
   - **Đối chứng trực diện 4 khung hình:** Ảnh đen trắng gốc vs Phân đoạn đen trắng (như code Kaggle/GitHub) vs Ảnh màu gốc vs Phân đoạn màu RGB.
   - Thể hiện rõ nét tại sao phân đoạn màu vượt trội hoàn toàn.

---

# 7. CHECKLIST TỔNG KẾT KHI ĐI THI

1. **Đặc trưng:** Ảnh đen trắng là vô hướng 1D ($I$), ảnh màu là vector 3D ($[R, G, B]$) hoặc 5D ($[R, G, B, x, y]$).
2. **K-Means màu:** Gom $N = H \times W$ vector màu thành $K$ cụm $\implies$ Trích xuất bảng màu (Color Palette), nén màu sắc.
3. **K-NN màu:** Dùng mẫu hạt giống màu (Seeds) $\implies$ Phân loại toàn bộ pixel để bóc tách đối tượng tiền cảnh (Foreground) khỏi hậu cảnh (Background).
4. **Khoảng cách:** Luôn dùng khoảng cách Euclidean 3D:
   $$d = \sqrt{\Delta R^2 + \Delta G^2 + \Delta B^2}$$
