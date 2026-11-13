#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIỆN ÍCH CHẠY DEMO XUẤT VIDEO MINH HỌA CƠ CHẾ THUẬT TOÁN (OMNP - HARO)
=============================================================================
Chạy lệnh trực tiếp từ thư mục gốc dự án:
    python demo_video_thuat_toan.py

Tùy chọn:
    python demo_video_thuat_toan.py --mode haro         (Chuyên sâu 5 cơ chế HARO)
    python demo_video_thuat_toan.py --mode comparative  (Đối sánh 5 thuật toán)
    python demo_video_thuat_toan.py --mode all          (Sinh cả hai video)
=============================================================================
"""

import os
import sys

# Thêm thư mục gốc dự án vào sys.path
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from visualization.demo_algorithm_mechanisms_video import main

if __name__ == "__main__":
    main()
