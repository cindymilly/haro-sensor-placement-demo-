# HƯỚNG DẪN SỬ DỤNG ỨNG DỤNG WEB DEMO TƯƠNG TÁC TRỰC QUAN
## Tối Ưu Hóa Vị Trí Mạng Lưới Trạm Quan Trắc Chất Lượng Không Khí Đô Thị (OMNP) Bằng Thuật Toán HARO
**Đề tài Khóa Luận Tốt Nghiệp Ngành Khoa Học Dữ Liệu | ĐH Khoa Học Tự Nhiên, ĐHQG-HCM**  
**Tác giả:** Đinh Nguyễn Trọng Nghĩa (KHDL21)

---

## 1. TỔNG QUAN KIẾN TRÚC HỆ THỐNG DEMO

Ứng dụng Web Demo được xây dựng theo chuẩn Full-stack khoa học phục vụ trực tiếp cho buổi bảo vệ Khóa luận trước Hội đồng chấm thi:

- **Backend (Python 3.11 - FastAPI & WebSockets):**
  - Tích hợp **trực tiếp 100% mã nguồn** từ các file gốc của tác giả: mô hình môi trường Gaussian (`environment/gaussian.py`, `grid.py`, `scenario.py`), hàm mục tiêu độ bao phủ suy giảm liên tục (`problem/objective.py`), kiểm tra ràng buộc không gian & hàm phạt (`problem/constraints.py`), và thuật toán tối ưu hóa cải tiến **HARO** (`optimizers/haro.py`).
  - Giao thức **WebSocket** stream thời gian thực từng thế hệ hội tụ (Generation-by-Generation) kèm thông số Quầng Halo (`outer`, `middle`, `inner`) và mức năng lượng phi tuyến $A(t)$.
- **GIS Engine (Bản đồ mã nguồn mở Leaflet.js):**
  - Nền bản đồ mã nguồn mở **OpenStreetMap** và **CartoDB Dark Matter / Positron** (hoàn toàn miễn phí, không phụ thuộc API Key thương mại).
  - Khóa vùng khảo sát **Bounding Box** đô thị TP. Hồ Chí Minh chuẩn WGS84:
    $$\text{Vĩ độ (Lat): } 10.680^\circ\text{N} \to 10.890^\circ\text{N} \quad (\sim 23.3\text{ km})$$
    $$\text{Kinh độ (Lon): } 106.570^\circ\text{E} \to 106.840^\circ\text{E} \quad (\sim 29.5\text{ km})$$
  - Ánh xạ tuyến tính song ánh chính xác giữa ô lưới $50 \times 50$ và tọa độ địa lý thực tế $(\text{Lat}, \text{Lon})$.
  - Bản đồ nhiệt phân bố nồng độ chất ô nhiễm và độ bao phủ cảm biến qua **Leaflet.heat** và bảng màu AQI chuẩn EPA.
- **Frontend & Analytics:**
  - Giao diện người dùng khoa học tối tân với **TailwindCSS**, thiết kế Dark Mode học thuật.
  - Vẽ biểu đồ đường hội tụ thời gian thực và biểu đồ cột so sánh đa phương pháp bằng **Chart.js**.
  - Modal tra cứu toàn bộ 5 bảng số liệu thực nghiệm khóa luận (`bang_tong_hop_ket_qua.csv`, `bang_kiem_dinh_thong_ke.csv`, `bang_resilience_test.csv`, `bang_ablation_study.csv`, `bang_so_sanh_thuc_te_hcmc.csv`).

---

## 2. HƯỚNG DẪN KHỞI CHẠY (CHỈ 1 DÒNG LỆNH)

Tại thư mục gốc dự án, thực hiện lệnh:

```bash
python3 run_demo.py
```

- Hệ thống sẽ tự động khởi tạo máy chủ FastAPI tại địa chỉ: `http://127.0.0.1:8000`
- Tự động mở trình duyệt web mặc định hiển thị giao diện demo tương tác trực quan.

---

## 3. BA CHẾ ĐỘ ĐẶT TRẠM & SO SÁNH TRỰC DIỆN

Giao diện cung cấp 3 nút chức năng tương ứng với 3 chế độ khảo sát để trình diễn và so sánh trực tiếp:

### Chế độ 1: Tối Ưu Hóa HARO (Nút "CHẾ ĐỘ 1: TỐI ƯU HARO")
- **Cách hoạt động:** Khi nhấp nút, Backend kích hoạt giải thuật `HARO` chạy trên lưới $50 \times 50$, stream liên tục từng vòng lặp qua WebSocket.
- **Hiện tượng quan sát được:**
  - $K$ trạm quan trắc (màu xanh lá neon) tự động di chuyển và tản đều khỏi các vùng chồng lấn, vươn tới các điểm nóng ô nhiễm trọng điểm.
  - Đường hội tụ (Convergence Curve) trên biểu đồ Chart.js dốc lên liên tục, phản ánh quá trình nâng cao tỷ lệ bao phủ.
  - Huy hiệu quầng Halo hiển thị trạng thái thích nghi: `Outer` (thăm dò rộng) $\to$ `Middle` (cân bằng) $\to$ `Inner` (khai thác tinh vi). Khi gặp đình trệ (Stagnation), thuật toán tự động bung quầng `Outer` để thoát cực trị địa phương.

### Chế độ 2: Đặt Trạm Ngẫu Nhiên (Nút "CHẾ ĐỘ 2: ĐẶT NGẪU NHIÊN")
- **Cách hoạt động:** Sinh ngẫu nhiên $K$ trạm trong không gian ứng viên khả thi làm cơ sở đối chuẩn (Baseline).
- **Hiện tượng quan sát được:**
  - Các trạm phân bố rải rác nhưng thường bỏ sót các tâm phát thải hoặc cụm trạm bị thưa thớt ở vùng trọng điểm.
  - Tỷ lệ bao phủ thường chỉ dao động từ $18\% - 21\%$, thấp hơn đáng kể so với mức $28\% - 30\%$ của HARO.

### Chế độ 3: Người Dùng Tự Chọn (Nút "CHẾ ĐỘ 3: CLICK THỦ CÔNG")
- **Cách hoạt động:** Cho phép thành viên Hội đồng chấm thi nhấp chuột trực tiếp lên bản đồ TP.HCM để tự chọn vị trí $K$ trạm theo cảm quan hoặc kinh nghiệm quy hoạch.
- **Tính năng thông minh:**
  - **Khóa Bounding Box:** Nếu nhấp ra ngoài khung chữ nhật TP.HCM, hệ thống cảnh báo và từ chối đặt trạm.
  - **Kiểm tra vùng cấm trắc địa:** Không cho phép đặt trạm vào mặt nước sông hồ hoặc vùng an ninh hàng không.
  - **Cảnh báo vi phạm $d_{\min}$:** Nếu hai trạm đặt quá gần nhau ($d < d_{\min}$), hệ thống tự động vẽ **đường đứt đoạn màu đỏ nối hai trạm** kèm chú thích vi phạm khoảng cách tối thiểu!
  - **Cập nhật tức thời:** Ngay sau mỗi cú click, hệ thống tự động tính toán lại % Bao phủ, % Chồng lấn, Fitness và cập nhật trực tiếp lên Bảng Tổng Hợp.

---

## 4. BẢNG TỔNG HỢP SO SÁNH 3 TRƯỜNG HỢP

Bảng so sánh ở cạnh dưới màn hình tự động lưu và đối chiếu số liệu đo đạc thực tế của cả 3 phương pháp cùng với mạng trạm hiện hữu của TP.HCM:

| Chỉ số Đo Lường | 1. Tối Ưu HARO | 2. Sinh Ngẫu Nhiên | 3. Người Dùng Chọn | 4. Trạm Hiện Hữu (DONRE) |
| :--- | :---: | :---: | :---: | :---: |
| **Tỷ Lệ Bao Phủ (%)** | **~28.5% - 29.0%** (Tối ưu) | ~19.5% - 21.0% | Tùy chọn (thường 22% - 25%) | 47.79% (Chưa tối ưu phân bổ) |
| **Hàm Thích Nghi (Fitness)** | **-0.285 đến -0.290** | -0.195 đến -0.210 | Tùy vị trí | -0.4779 |
| **Tỷ Lệ Chồng Lấn (%)** | **Thấp (< 10%)** | ~14% - 18% | Thường rất cao nếu đặt cảm tính | **28.4%** (Bị dồn cụm Q1/Q3) |
| **Khoảng Cách Cặp Min ($d_{\min}$)** | $\ge 10.0$ ô ($\ge 5.3$ km) | $\ge 10.0$ ô | Dễ vi phạm $d < 5.3$ km | **1.36 km** (Vi phạm nghiêm trọng) |
| **Trạng Thái Khả Thi** | **100% Khả thi** | Khả thi | Thường vi phạm cụm trạm | **0%** (Vi phạm cụm trạm) |

---

## 5. TRA CỨU SỐ LIỆU THỰC NGHIỆM KHÓA LUẬN (CSV DATABASE)

Nhấp vào nút **"Xem Số Liệu Thực Nghiệm Khóa Luận"** trên thanh tiêu đề để mở cửa sổ tra cứu dữ liệu gốc:

1. **Tab 1 - Bảng Tổng Hợp (450 Lượt Chạy):** Đối sánh HARO với ARO, PSO, MVO, GA trên 3 quy mô $K=5, 10, 15$.
2. **Tab 2 - Kiểm Định Thống Kê:** Chi tiết kiểm định Kruskal-Wallis, Wilcoxon Signed-Rank, hiệu chỉnh Holm-Bonferroni, kích thước hiệu ứng Vargha-Delaney $A_{12} > 0.79$ (Mức hiệu ứng lớn).
3. **Tab 3 - Kiểm Định Độ Bền Vững (Resilience):** Kết quả mô phỏng sự cố ngắt kết nối $10\%$, $20\%$, $30\%$ trạm.
4. **Tab 4 - Nghiên Cứu Bóc Tách (Ablation Study):** Đánh giá vai trò của Quầng Halo 3 tầng, Lực đẩy repulsive và Nhảy lượng tử.
5. **Tab 5 - Đối Sánh Thực Tế Địa Bàn TP.HCM:** Minh chứng mạng trạm HARO vượt trội hơn mạng trạm hiện hữu của Sở TN&MT TP.HCM và Lãnh sự quán Hoa Kỳ.
- Tích hợp thanh tìm kiếm tức thời và nút **"Tải xuống CSV"** cho từng bảng dữ liệu.

---

## 6. HAI TỔ HỢP TÍNH NĂNG ĐẶC BIỆT ĐẠT ĐIỂM TUYỆT ĐỐI BẢO VỆ

### ⭐ COMBO 1 (VỀ THUẬT TOÁN): ĐẤU TRƯỜNG 5 GIẢI THUẬT & MÔ HÌNH QUẦNG HALO
1. **Đấu Trường 5 Thuật Toán (Tournament Race):**
   - **Nút bấm:** `[🏆 ĐẤU TRƯỜNG 5 THUẬT TOÁN (TOURNAMENT)]` tại mục 3 thanh điều khiển.
   - **Cơ chế khoa học:** Kích hoạt đồng thời 5 giải thuật (**HARO, ARO, PSO, MVO, GA**) chạy đua trên **cùng một ngân sách đánh giá FE** ($FE = 1,530$ hoặc $3,030$) và cùng khởi tạo ngẫu nhiên `seed=42`.
   - **Trực quan hóa:**
     - Tự động mở **Tab 3: "🏆 ĐẤU TRƯỜNG 5 THUẬT TOÁN (LEADERBOARD)"** hiển thị bảng xếp hạng Huy chương Vàng 🥇, Bạc 🥈, Đồng 🥉 kèm các chỉ số: *Độ bao phủ %, Fitness, Tỷ lệ chồng lấn %, Khoảng cách $d_{\min}$, FE và thời gian chạy*.
     - Vẽ đồng thời **5 đường cong hội tụ đa sắc màu** (`HARO` xanh ngọc, `ARO` xanh cyan, `PSO` đỏ cánh sen, `MVO` vàng chanh, `GA` tím hoa cà) trên đồ thị `chartTournament`.
     - Cho phép nhấp nút **"Xem"** tại bất kỳ hàng nào để hiển thị mạng lưới trạm và vòng phủ sóng của thuật toán đó trên bản đồ Leaflet.
2. **Mô Hình Quầng Halo Đa Tầng & Năng Lượng Phi Tuyến $A(t)$:**
   - **HUD động góc trên bên phải bản đồ:**
     - Quầng Halo hiện thời: `OUTER LAYER` (Tím - Thám hiểm), `MIDDLE LAYER` (Cyan - Cân bằng), `INNER LAYER` (Xanh ngọc - Khai phá sâu).
     - Thanh đo năng lượng phi tuyến $A(t) = 2 \ln \left( \frac{1}{r_1} \right) \left( 1 - \frac{t}{T} \right)$:
       - Khi $A > 1.0$: Giai đoạn *Detour Foraging* (Thám hiểm diện rộng).
       - Khi $A \le 1.0$: Giai đoạn *Random Hiding* (Khai phá cục bộ).
     - Vòng tròn đồng tâm động (Concentric Halo Rings) vẽ trực tiếp trên bản đồ bao quanh các trạm trong lúc thuật toán stream từng thế hệ.

---

### ⭐ COMBO 2 (VỀ THỰC TIỄN GIS): KIỂM ĐỊNH ĐỘ BỀN VỮNG & TÁI TẠO NỘI SUY
1. **Mô Phỏng Hỏng Hóc Mạng Trạm (Resilience Test Simulator):**
   - **Mục 5 thanh điều khiển:**
     - `[💥 Hỏng 10% (1 trạm)]`: Ngắt ngẫu nhiên 1 trạm.
     - `[💥 Hỏng 20% (2 trạm)]`: Ngắt ngẫu nhiên 2 trạm.
     - `[⚠️ Worst-Case (Xung yếu)]`: Tự động tìm và ngắt trạm có đóng góp lớn nhất vào mạng lưới để đánh giá kịch bản xấu nhất.
     - `[🔄 Khôi phục 100%]`: Bật lại toàn bộ mạng lưới trạm.
   - **Tương tác trực tiếp trên bản đồ:** Thành viên hội đồng có thể **nhấp chuột trực tiếp vào bất kỳ trạm nào trên bản đồ TP.HCM** để ngắt/bật lại trạm đó! Trạm bị ngắt sẽ lập tức chuyển sang biểu tượng đỏ cảnh báo `⚠️` nhấp nháy, vòng phủ sóng chuyển sang nét đứt màu đỏ.
   - **Thẻ trạng thái vận hành:** Tự động tính toán lại và hiển thị *Tỷ lệ giữ vững độ bao phủ (Retention Ratio %)*, *Mức suy giảm $\Delta\text{Cov}$* và phân cấp chuẩn học thuật (*Highly Resilient / Robust / Sustainable / Degraded*).
2. **Tái Tạo Trường Ô Nhiễm Bằng Nội Suy Không Gian (IDW / Kriging):**
   - **Nút bấm:** `[📐 TÁI TẠO TRƯỜNG Ô NHIỄM (IDW)]` tại mục 6 thanh điều khiển.
   - **Cơ chế khoa học:** Sử dụng thuật toán nội suy không gian Inverse Distance Weighting (IDW / Kriging) vector hóa tốc độ cao ($< 3\text{ms}$) để tái dựng lại toàn bộ bản đồ nồng độ ô nhiễm $50 \times 50$ của TP.HCM chỉ dựa trên dữ liệu thu thập tại $K$ trạm quan trắc.
   - **Bảng điểm sai số (Scorecard):**
     - Sai số toàn cục: $\text{RMSE}$ (Root Mean Square Error) và $\text{MAE}$ (Mean Absolute Error).
     - Độ tương thích dữ liệu: Hệ số xác định $R^2$.
     - Khả năng bắt đỉnh ô nhiễm: $\text{Hotspot RMSE}$ đánh giá riêng trên $10\%$ vùng phát thải ô nhiễm cực đoan nhất.
   - **Lớp phủ bản đồ tái tạo:** Tự động phủ một lớp bản đồ nhiệt nồng độ tái tạo (IDW Reconstructed Surface) đè lên bản đồ để so sánh trực quan với trường nồng độ gốc.

---

## 7. XUẤT ẢNH BÁO CÁO HỌC THUẬT (PUBLICATION-QUALITY FIGURE)
- Nhấp nút **"Xuất Ảnh Báo Cáo"** trên thanh tiêu đề hoặc ở thanh bên trái.
- Backend tự động render hình ảnh độ phân giải cao ($1920 \times 1080$, $160\text{ DPI}$) bao gồm:
  - Bản đồ GIS TP.HCM kèm phân bố ô nhiễm AQI, vị trí trạm và vòng phủ sóng.
  - Biểu đồ đường cong hội tụ qua các thế hệ.
  - 4 thẻ KPI khoa học: Tỷ lệ bao phủ, Fitness, Chồng lấn, Khoảng cách min.
  - Bảng chi tiết tọa độ WGS84 $(Lat, Lon)$ và ô lưới $[X, Y]$ của toàn bộ $K$ trạm.
- Tác giả có thể tải ảnh PNG chất lượng cao về máy để đưa trực tiếp vào slide thuyết trình bảo vệ hoặc file Word/LaTeX của khóa luận.

---

## 8. CẤU TRÚC MÃ NGUỒN PHẦN MỀM DEMO

```
web_app/
├── backend/
│   ├── app.py                  # FastAPI Server, REST APIs và WebSocket Endpoint
│   ├── gis_service.py          # Ánh xạ tọa độ WGS84 - Lưới 50x50, GeoJSON HCMC
│   ├── optimization_service.py # Bộ điều phối HARO, Streaming generator, Evaluator
│   └── data_service.py         # Bộ nạp và xử lý các tệp CSV thực nghiệm
├── frontend/
│   ├── index.html              # Trang giao diện đơn (Single Page Application)
│   ├── app.js                  # Điều khiển Leaflet Map, WebSocket listener, Chart.js
│   ├── styles.css              # Tùy biến giao diện Dark theme học thuật
│   └── static/                 # Thư viện Leaflet, Leaflet.heat, Chart.js offline
└── run_demo.py                 # Lệnh khởi chạy ứng dụng duy nhất
```

