#!/usr/bin/env bash
# =============================================================================
# SCRIPT TẠO LINK DEMO WEB CÔNG KHAI ĐỂ GỬI CHO THẦY CÔ & HỘI ĐỒNG CHẤM THI
# Đề tài Khóa Luận Tốt Nghiệp: Tối ưu hóa vị trí đặt trạm quan trắc (HARO)
# Tác giả: Đinh Nguyễn Trọng Nghĩa (KHDL21 - ĐH Khoa học Tự nhiên, ĐHQG-HCM)
# =============================================================================

echo "==============================================================================="
echo "   ĐANG KHỞI TẠO ĐƯỜNG LINK CÔNG KHAI (INTERNET) CHO BẢN DEMO WEB KHÓA LUẬN    "
echo "==============================================================================="

# 1. Kiểm tra máy chủ local có đang chạy trên port 8000 hay không
if ! lsof -i :8000 > /dev/null 2>&1; then
    echo "[1/3] Máy chủ local chưa chạy. Đang tự động khởi chạy backend FastAPI..."
    python3 run_demo.py > /dev/null 2>&1 &
    sleep 3
else
    echo "[1/3] Máy chủ local FastAPI đang chạy sẵn sàng trên cổng 8000."
fi

# 2. Kiểm tra công cụ cloudflared
if ! command -v cloudflared &> /dev/null; then
    echo "[Lỗi] Chưa tìm thấy cloudflared. Vui lòng cài đặt: brew install cloudflared"
    exit 1
fi

echo "[2/3] Đang kết nối mạng đám mây Cloudflare để cấp link HTTPS miễn phí..."
TUNNEL_LOG="/tmp/cf_tunnel_haro.log"
rm -f "$TUNNEL_LOG"

# Khởi chạy tunnel nền
cloudflared tunnel --url http://127.0.0.1:8000 > "$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!

# Đợi lấy link URL
PUBLIC_URL=""
for i in {1..30}; do
    sleep 1
    PUBLIC_URL=$(grep -o "https://[a-zA-Z0-9.-]*trycloudflare\.com" "$TUNNEL_LOG" | head -n 1)
    if [ -n "$PUBLIC_URL" ]; then
        break
    fi
done

if [ -z "$PUBLIC_URL" ]; then
    echo "[Lỗi] Không thể lấy link từ Cloudflare Tunnel. Chi tiết log tại $TUNNEL_LOG"
    kill $TUNNEL_PID 2>/dev/null
    exit 1
fi

echo "==============================================================================="
echo "🎉 LINK DEMO WEB CỦA BẠN ĐÃ SẴN SÀNG! COPY ĐỂ GỬI CHO THẦY CÔ:"
echo ""
echo "   👉  $PUBLIC_URL"
echo ""
echo "-------------------------------------------------------------------------------"
echo "💡 THÔNG TIN DÀNH CHO BẠN:"
echo "• Thầy cô có thể mở link này trên MỌI THIẾT BỊ (Laptop, Điện thoại, Máy tính bảng)"
echo "  mà KHÔNG CẦN cài đặt Python hay bất kỳ phần mềm nào."
echo "• Hỗ trợ đầy đủ tính năng: Tối ưu HARO, Đấu trường 5 giải thuật, Click trạm,"
echo "  Mô phỏng hỏng hóc trạm, Tái tạo nội suy IDW, và Xuất ảnh báo cáo học thuật."
echo "• Giữ cửa sổ Terminal này mở trong lúc thầy cô trải nghiệm."
echo "• Nhấn Ctrl + C để dừng chia sẻ link."
echo "==============================================================================="

# Bắt tín hiệu Ctrl+C để dọn dẹp
trap "echo ''; echo 'Đã đóng link chia sẻ.'; kill $TUNNEL_PID 2>/dev/null; exit 0" SIGINT SIGTERM

wait $TUNNEL_PID

