# Tối Ưu Hóa Vị Trí Mạng Lưới Trạm Quan Trắc Chất Lượng Không Khí Đô Thị Bằng Thuật Toán Thỏ Nhân Tạo Cải Tiến Với Quầng Halo Đa Tầng (HARO)

Hệ thống mã nguồn tối ưu hóa vị trí mạng lưới trạm quan trắc chất lượng không khí đô thị (Optimal Monitoring Network Problem - OMNP). Bài toán hướng đến việc cực đại hóa mức độ bao phủ trường nồng độ ô nhiễm suy giảm liên tục theo khoảng cách, đồng thời thỏa mãn ràng buộc khoảng cách tối thiểu giữa các cặp trạm ($d_{\min}$) và giới hạn nghiêm ngặt về ngân sách đánh giá hàm mục tiêu (Strict Function Evaluation Budgeting).

---

## 1. Kiến Trúc Hệ Thống & Chức Năng Từng Khối Module

Toàn bộ hệ thống được thiết kế theo kiến trúc module hóa phân tầng, đảm bảo tính độc lập giữa mô hình môi trường lan truyền vật lý, định nghĩa toán học bài toán tối ưu, các giải thuật siêu phỏng sinh học và khung thực nghiệm đối chứng.

```
├── main.py                     # Điểm thực thi chính (CLI Runner)
├── requirements.txt            # Danh sách phụ thuộc môi trường Python
├── pytest.ini                 # Cấu hình kiểm thử đơn vị tự động
├── LICENSE                     # Giấy phép mã nguồn mở MIT
│
├── environment/                # Module mô phỏng trường phân tán ô nhiễm
│   ├── types.py                # Cấu trúc dữ liệu: GridSpec, PointSource, EnvironmentResult
│   ├── grid.py                 # Khởi tạo lưới không gian 2D rời rạc hóa (M x N)
│   ├── sources.py              # Định nghĩa nguồn phát thải điểm, diện và đường
│   ├── gaussian.py             # Mô hình lan truyền khói ô nhiễm Gaussian 2D đẳng hướng
│   ├── scenario.py             # Đóng gói và tổng hợp các kịch bản phát thải
│   └── test_environment.py     # Kiểm thử đơn vị cho module môi trường
│
├── problem/                    # Định nghĩa bài toán OMNP và xử lý ràng buộc
│   ├── candidates.py           # Quản lý tập vị trí ứng viên khả thi trên lưới
│   ├── encoding.py             # Lớp biểu diễn giải pháp mạng trạm (StationSolution)
│   ├── objective.py            # Hàm mục tiêu độ bao phủ suy giảm liên tục
│   ├── constraints.py          # Bộ kiểm tra ràng buộc không gian và hàm phạt
│   ├── problem.py              # MonitoringProblem, đếm FE và Backtracking Spatial Repair
│   └── test_problem.py         # Kiểm thử tính hợp lệ và cơ chế sửa nghiệm
│
├── optimizers/                 # Các thuật toán tối ưu hóa đối chuẩn
│   ├── base.py                 # BaseOptimizer: kiểm soát FE, ghi vết hội tụ và snap nghiệm
│   ├── haro.py                 # Thuật toán đề xuất: Halo-enhanced ARO
│   ├── aro.py                  # Thuật toán gốc: Artificial Rabbits Optimization
│   ├── pso.py                  # Thuật toán đối chuẩn: Particle Swarm Optimization
│   ├── mvo.py                  # Thuật toán đối chuẩn: Multi-Verse Optimizer
│   ├── ga.py                   # Thuật toán đối chuẩn: Genetic Algorithm
│   ├── test_optimizers.py      # Kiểm thử khả năng tối ưu hóa của các thuật toán
│   └── test_fe_accounting.py   # Kiểm thử kiểm toán ngân sách đánh giá hàm mục tiêu
│
├── experiments/                # Khung thực nghiệm Monte Carlo và kiểm định thống kê
│   ├── scenario_builder.py     # Thiết lập kịch bản thực nghiệm chuẩn hóa theo kích thước lưới
│   ├── runner.py               # Điều phối chạy thực nghiệm đơn lẻ và tham số hóa
│   ├── run_experiments.py      # Bộ điều phối đa tiến trình 450 runs và kiểm định thống kê
│   └── ablation_and_resilience.py # Thực nghiệm bóc tách thành phần và độ bền vững
│
├── visualization/              # Module trực quan hóa kết quả
│   └── plots.py                # Vẽ đồ thị hội tụ (FE/Iteration) và bản đồ nhiệt nồng độ
│
└── ket_qua_thi_nghiem/         # Dữ liệu thực nghiệm thực tế và đồ thị học thuật
    ├── raw_runs_results.csv    # Dữ liệu chi tiết 450 lượt chạy độc lập
    ├── convergence_traces.csv  # Vết hội tụ chi tiết theo từng thế hệ và FE
    ├── bang_thong_ke_mo_ta.csv # Thống kê mô tả (Mean, Std, Median, IQR, Best, Time)
    ├── bang_fe_audit.csv       # Kiểm toán ngân sách đánh giá FE = 3.030
    ├── bang_kiem_dinh_thong_ke.csv # Kiểm định cặp đôi: Wilcoxon, Holm, A12, Cliff's delta
    ├── bang_kiem_dinh_across_instances.csv # Kiểm định đa kịch bản: Friedman & Wilcoxon
    ├── bang_ablation_study.csv # Kết quả bóc tách từng cơ chế trong HARO
    ├── bang_resilience_test.csv# Kết quả kiểm định độ bền vững khi hỏng hóc trạm
    └── *.png                   # Toàn bộ bản đồ nhiệt trạm và đồ thị hội tụ
```

### 1.1. Khối Môi Trường (`environment/`)
- **Vai trò:** Mô phỏng hiện tượng phát tán chất ô nhiễm không khí trong không gian đô thị 2D dựa trên cơ sở vật lý khí quyển.
- **`grid.py`:** Quản lý lưới tọa độ không gian kích thước $M \times N$ với độ phân giải bước lưới $\Delta x, \Delta y$. Chuyển đổi qua lại giữa chỉ số hàng/cột $(i, j)$ và tọa độ thực tế $(x, y)$.
- **`sources.py`:** Đại diện cho các nguồn phát thải ô nhiễm. Mỗi nguồn được đặc trưng bởi vị trí tâm $(x_s, y_s)$, cường độ phát thải $Q_s > 0$, và độ phân tán $\sigma$.
- **`gaussian.py`:** Triển khai mô hình khuếch tán Gaussian 2D đẳng hướng:
  $$C(x, y) = \sum_{s=1}^{S} \frac{Q_s}{2\pi \sigma^2} \exp\left(-\frac{(x - x_s)^2 + (y - y_s)^2}{2\sigma^2}\right)$$
  Tính toán ma trận nồng độ ô nhiễm $C \in \mathbb{R}^{M \times N}$ trên toàn bộ miền khảo sát.
- **`scenario.py`:** Khởi tạo các kịch bản ô nhiễm chuẩn hóa, bao gồm bố trí phân tán đa nguồn nhằm phản ánh đặc trưng các vùng phát thải công nghiệp và giao thông đô thị.
- **`types.py`:** Định nghĩa các dataclass biểu diễn thông số lưới (`GridSpec`), nguồn phát thải (`PointSource`), và kết quả mô phỏng môi trường (`EnvironmentResult`).

### 1.2. Khối Bài Toán Tối Ưu (`problem/`)
- **Vai trò:** Thiết lập không gian tìm kiếm, hàm thích nghi, kiểm tra các ràng buộc kỹ thuật và cơ chế phục hồi nghiệm khả thi.
- **`candidates.py`:** Quản lý tập hợp các vị trí ứng viên hợp lệ $\mathcal{C} \subset \mathbb{R}^2$ mà trạm quan trắc có thể được xây dựng, hỗ trợ loại trừ các vùng cấm địa lý.
- **`encoding.py`:** Lớp `StationSolution` biểu diễn một cấu hình mạng lưới gồm danh sách tọa độ của $K$ trạm: $S = \{s_1, s_2, \dots, s_K\}$.
- **`objective.py`:** Hàm mục tiêu đo lường mức độ bao phủ ô nhiễm theo khoảng cách suy giảm hàm mũ:
  $$\text{Coverage}(x, y, S) = \max_{k=1..K} \exp\left(-\lambda \cdot \|(x, y) - s_k\|\right)$$
  với hệ số suy giảm $\lambda = \frac{\ln(10)}{R_{\text{sens}}}$, đảm bảo giá trị bao phủ giảm về $10\%$ tại biên bán kính cảm biến $R_{\text{sens}}$. Hàm thích nghi cần cực tiểu hóa:
  $$f_{\text{fitness}}(S) = - \frac{\sum_{x, y} C(x, y) \cdot \text{Coverage}(x, y, S)}{\sum_{x, y} C(x, y)}$$
- **`constraints.py`:** Đánh giá tính khả thi của giải pháp $S$:
  - Số lượng trạm chính xác bằng $K$.
  - Mọi tọa độ $s_k$ phải thuộc tập ứng viên $\mathcal{C}$.
  - Ràng buộc khoảng cách tối thiểu giữa các cặp trạm $\|s_i - s_j\| \ge d_{\min}, \forall i \neq j$. Vi phạm được lượng hóa qua hàm phạt bậc hai:
    $$\text{Penalty}(S) = \gamma \sum_{i < j} \max\left(0, \, d_{\min} - \|s_i - s_j\|\right)^2$$
- **`problem.py`:** Lớp điều phối trung tâm `MonitoringProblem`. Tích hợp:
  - Bộ đếm số lần gọi hàm mục tiêu `actual_fe`.
  - Nghiệm tham chiếu an toàn `_feasible_witness` được sinh tiền nghiệm.
  - Cơ chế sửa nghiệm không gian quay lui tiền tố (*Backtracking Spatial Repair*): Khi nghiệm vi phạm khoảng cách $d_{\min}$, thuật toán duyệt từng trạm từ $1$ đến $K$. Nếu trạm thứ $k$ xung đột với các trạm đã chọn trước đó, thuật toán tìm kiếm vị trí thay thế an toàn trong tập ứng viên. Nếu không thể tìm thấy ứng viên khả thi, thuật toán quay lui rút ngắn tiền tố và lấp đầy các trạm còn lại bằng nghiệm nhân chứng đã biết chắc chắn khả thi. Đảm bảo tỷ lệ nghiệm hợp lệ $100\%$ không gây tắc nghẽn vòng lặp.

### 1.3. Khối Thuật Toán Tối Ưu (`optimizers/`)
- **Vai trò:** Tìm kiếm cấu hình mạng trạm tối ưu trong không gian tìm kiếm rời rạc thông qua việc dẫn hướng các cá thể liên tục và ánh xạ về lưới khả thi.
- **`base.py`:** Lớp trừu tượng `BaseOptimizer`. Đảm bảo mọi thuật toán:
  - Tuân thủ cùng giới hạn ngân sách đánh giá hàm mục tiêu $\text{FE}_{\max} = N_{\text{pop}} \times (T + 1) = 3.030$.
  - Tự động dừng vòng lặp ngay khi chạm ngưỡng ngân sách FE.
  - Lưu vết lịch sử tiến hóa: `history_best_fitness`, `history_mean_fitness`, `history_fe`.
- **`haro.py`:** Thuật toán đề xuất Halo-enhanced Artificial Rabbits Optimization:
  - Tích hợp hàm suy giảm năng lượng phi tuyến $A(t)$ điều phối quá trình thăm dò và khai thác.
  - Cơ chế quầng Halo đa tầng (3 tầng quầng thích nghi theo tỷ lệ vàng và khoảng cách Mahalanobis) tạo nhiễu định hướng xung quanh cá thể tốt nhất, giúp giải phóng khỏi các cực trị địa phương.
  - Cơ chế nhảy lượng tử ngẫu nhiên (Quantum Jump) tái khởi động các cá thể đình trệ.
- **`aro.py`:** Thuật toán Thỏ nhân tạo nguyên bản (Wang et al., 2022) với hành vi tìm kiếm thức ăn vòng tròn và ẩn náu ngẫu nhiên.
- **`pso.py`:** Thuật toán Tối ưu hóa bầy đàn hạt (Particle Swarm Optimization) với hệ số quán tính giảm tuyến tính và hệ số học tập nhận thức/xã hội.
- **`mvo.py`:** Thuật toán Đa vũ trụ (Multi-Verse Optimizer) dựa trên mô hình lỗ trắng, lỗ đen và hố sâu không gian.
- **`ga.py`:** Giải thuật Di truyền chuẩn (Genetic Algorithm) với chọn lọc bánh xe roulette, lai ghép điểm cắt ngẫu nhiên và đột biến đồng đều.

### 1.4. Khối Thực Nghiệm & Kiểm Định Thống Kê (`experiments/`)
- **Vai trò:** Thực thi các chiến dịch đo lường thực nghiệm, đảm bảo tính công bằng và kiểm chứng bằng phương pháp thống kê nghiêm ngặt.
- **`scenario_builder.py`:** Xây dựng cấu hình chuẩn hóa cho 3 quy mô bài toán: Lưới $50 \times 50$ với số lượng trạm $K \in \{5, 10, 15\}$, bán kính cảm biến $R_{\text{sens}} = 8.0$, khoảng cách tối thiểu $d_{\min} = 10.0$.
- **`run_experiments.py`:** Điều phối thực nghiệm Monte Carlo song song:
  - 450 lượt chạy độc lập (3 kịch bản $\times$ 5 thuật toán $\times$ 30 hạt giống ngẫu nhiên).
  - Xuất dữ liệu thô ra `raw_runs_results.csv` và quỹ đạo hội tụ ra `convergence_traces.csv`.
  - Phân tích thống kê mô tả: Mean, Standard Deviation, Median, Interquartile Range (IQR), Best Fitness, Runtime.
  - Kiểm định phi tham số đa tầng: Kruskal-Wallis test (so sánh tổng thể), Wilcoxon signed-rank test (so sánh cặp đôi), hiệu chỉnh giá trị $p$ theo phương pháp Holm-Bonferroni, tính toán kích thước hiệu ứng Vargha-Delaney ($A_{12}$) và Cliff's delta ($\delta$), kiểm định xếp hạng Friedman qua nhiều bài toán.
- **`ablation_and_resilience.py`:**
  - *Ablation Study:* Đánh giá đóng góp độc lập của từng cơ chế (loại bỏ Halo, loại bỏ cơ chế năng lượng, loại bỏ bước nhảy ngẫu nhiên) so với HARO toàn phần trên lưới $70 \times 70, K=10$.
  - *Resilience Test:* Kiểm tra độ bền vững của mạng lưới khi gặp sự cố ngắt kết nối $10\%$, $20\%$, $30\%$ số lượng trạm ngẫu nhiên qua 100 lần mô phỏng Monte Carlo.

### 1.5. Khối Trực Quan Hóa (`visualization/`)
- **Vai trò:** Tạo lập đồ thị biểu diễn động thái hội tụ và bản đồ nhiệt không gian chất lượng xuất bản học thuật.
- **`plots.py`:** 
  - Đồ thị hội tụ theo số thế hệ (*Generations*) và theo ngân sách hàm mục tiêu thực tế (*Function Evaluations*).
  - Bản đồ nhiệt phân bố nồng độ ô nhiễm $C(x, y)$ lồng ghép vị trí các trạm quan trắc tối ưu kèm vòng tròn bán kính bảo vệ $R_{\text{sens}}$.
  - Hỗ trợ xuất biểu đồ đơn lẻ từng kịch bản và biểu đồ tổng hợp $3 \times 1$ cho các kịch bản $K = 5, 10, 15$.

### 1.6. Khối Dữ Liệu Thực Nghiệm (`ket_qua_thi_nghiem/`)
Toàn bộ kết quả thực nghiệm sau khi chạy hoàn tất được lưu trữ tại thư mục này:
- `raw_runs_results.csv`: 450 dòng kết quả đo lường đầy đủ giá trị thích nghi, độ bao phủ, số FE thực tế, thời gian chạy và tọa độ mạng trạm.
- `convergence_traces.csv`: Dữ liệu vết hội tụ của tất cả các thuật toán qua từng thế hệ.
- `bang_fe_audit.csv`: Bảng đối soát ngân sách đánh giá (xác nhận $100\%$ các thuật toán dừng chính xác tại $\text{FE} = 3.030$, không vượt quá ngân sách).
- `bang_thong_ke_mo_ta.csv`: Tổng hợp các chỉ số trung bình, độ lệch chuẩn, trung vị và khoảng tứ phân vị.
- `bang_kiem_dinh_thong_ke.csv`: Kết quả các kiểm định cặp đôi kèm giá trị hiệu chỉnh Holm.
- `bang_kiem_dinh_across_instances.csv`: Kết quả kiểm định thống kê trên toàn bộ các kịch bản thực nghiệm.
- `bang_ablation_study.csv` & `bang_resilience_test.csv`: Bảng số liệu phân tích bóc tách và khả năng chống chịu sự cố.
- Các tệp `.png`: Đồ thị hội tụ và bản đồ phân bố trạm độ phân giải cao.

---

## 2. Hướng Dẫn Cài Đặt & Sử Dụng

### 2.1. Yêu cầu hệ thống
- Python $\ge$ 3.10
- Khuyến nghị sử dụng môi trường ảo (`venv` hoặc `conda`).

Cài đặt các gói phụ thuộc:
```bash
pip install -r requirements.txt
```

### 2.2. Chạy kiểm thử tự động
Thực thi toàn bộ 33 ca kiểm thử đơn vị bao phủ các module:
```bash
pytest
```

### 2.3. Chạy thử nghiệm nhanh (Quick Test)
Kiểm tra hoạt động của thuật toán và quá trình sinh dữ liệu:
```bash
python3 main.py --quick-test
```

### 2.4. Chạy toàn bộ thực nghiệm đối chứng chuẩn hóa
Thực thi 450 runs song song cho 5 thuật toán trên 3 kịch bản:
```bash
python3 experiments/run_experiments.py
```
Toàn bộ bảng dữ liệu `.csv` và hình ảnh đồ thị `.png` sẽ được tự động cập nhật vào thư mục `ket_qua_thi_nghiem/`.

### 2.5. Chạy phân tích bóc tách và kiểm tra độ bền vững mạng lưới
```bash
python3 experiments/ablation_and_resilience.py
```

---

## 3. Giấy Phép
Dự án được phân phối dưới điều khoản của giấy phép mã nguồn mở [MIT License](LICENSE).
# haro-sensor-placement-demo-
