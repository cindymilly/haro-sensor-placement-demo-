"""Thesis Experimental Data Loader and Statistical Inspection Service.

Loads, formats, and serves all official thesis results from CSV tables:
- `bang_tong_hop_ket_qua.csv`: Comprehensive benchmark of 450 runs (HARO vs ARO vs PSO vs MVO vs GA).
- `bang_kiem_dinh_thong_ke.csv`: Non-parametric statistical tests (Wilcoxon, Holm, A12, Cliff's delta).
- `bang_resilience_test.csv`: Network resilience against 10%, 20%, 30% node failures.
- `bang_ablation_study.csv`: Contribution of each core mechanism in HARO.
- `bang_so_sanh_thuc_te_hcmc.csv`: Real-world HCMC municipal baseline vs HARO deployment.
"""

from __future__ import annotations

import os
import csv
from typing import Dict, Any, List, Optional

# Base directories to search for CSV files
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

CSV_PATHS = {
    "tong_hop": [
        os.path.join(BASE_DIR, "KHDL_KLTN21_DinhNguyenTrongNghia", "ket_qua_thi_nghiem", "bang_tong_hop_ket_qua.csv"),
        os.path.join(BASE_DIR, "ket_qua_thi_nghiem", "bang_tong_hop_ket_qua.csv"),
    ],
    "kiem_dinh": [
        os.path.join(BASE_DIR, "KHDL_KLTN21_DinhNguyenTrongNghia", "ket_qua_thi_nghiem", "bang_kiem_dinh_thong_ke.csv"),
        os.path.join(BASE_DIR, "ket_qua_thi_nghiem", "bang_kiem_dinh_thong_ke.csv"),
    ],
    "resilience": [
        os.path.join(BASE_DIR, "KHDL_KLTN21_DinhNguyenTrongNghia", "ket_qua_thi_nghiem", "bang_resilience_test.csv"),
        os.path.join(BASE_DIR, "ket_qua_thi_nghiem", "bang_resilience_test.csv"),
    ],
    "ablation": [
        os.path.join(BASE_DIR, "KHDL_KLTN21_DinhNguyenTrongNghia", "ket_qua_thi_nghiem", "bang_ablation_study.csv"),
        os.path.join(BASE_DIR, "ket_qua_thi_nghiem", "bang_ablation_study.csv"),
    ],
    "thuc_te_hcmc": [
        os.path.join(BASE_DIR, "tai_lieu_tham_khao", "archive_legacy_gis_hcmc", "bang_so_sanh_thuc_te_hcmc.csv"),
        os.path.join(BASE_DIR, "KHDL_KLTN21_DinhNguyenTrongNghia", "tai_lieu_tham_khao", "archive_legacy_gis_hcmc", "bang_so_sanh_thuc_te_hcmc.csv"),
    ],
    "thong_ke_mo_ta": [
        os.path.join(BASE_DIR, "KHDL_KLTN21_DinhNguyenTrongNghia", "ket_qua_thi_nghiem", "bang_thong_ke_mo_ta.csv"),
        os.path.join(BASE_DIR, "ket_qua_thi_nghiem", "bang_thong_ke_mo_ta.csv"),
    ],
}

TABLE_METADATA = {
    "tong_hop": {
        "title": "Bảng Tổng Hợp Kết Quả Thực Nghiệm Đối Chuẩn (450 Lượt Chạy)",
        "subtitle": "So sánh 5 thuật toán (HARO, ARO, PSO, MVO, GA) trên lưới 50x50 với K = 5, 10, 15",
        "key_insight": "Thuật toán HARO đề xuất vượt trội so với ARO gốc và các thuật toán kinh điển, đạt tỷ lệ bao phủ trung bình cao nhất và duy trì 100% tính khả thi mà không cần sửa nghiệm dự phòng.",
        "badge": "450 Independent Runs",
    },
    "kiem_dinh": {
        "title": "Bảng Kết Quả Kiểm Định Thống Kê Phi Tham Số & Kích Thước Hiệu Ứng",
        "subtitle": "Kiểm định Kruskal-Wallis, Wilcoxon Signed-Rank kèm hiệu chỉnh Holm-Bonferroni, Vargha-Delaney A12 và Cliff's delta",
        "key_insight": "HARO thắng áp đảo ARO và PSO với p-value < 10^-4 và kích thước hiệu ứng A12 > 0.79 (mức hiệu ứng lớn Large Effect Size theo chuẩn học thuật).",
        "badge": "Wilcoxon & Holm p-value",
    },
    "resilience": {
        "title": "Bảng Kiểm Định Độ Bền Vững Mạng Trạm Khi Hỏng Hóc (Resilience Test)",
        "subtitle": "Khảo sát suy giảm độ bao phủ khi ngắt kết nối ngẫu nhiên và trường hợp xấu nhất (10%, 20%, 30% số trạm)",
        "key_insight": "Khi mất 1 trạm (90% trạm hoạt động), tỷ lệ giữ vững độ bao phủ đạt 92.19% (ngẫu nhiên) và 89.72% (xấu nhất), chứng minh tính dự phòng phân tán cao của mạng trạm HARO.",
        "badge": "Monte Carlo Failure Sim",
    },
    "ablation": {
        "title": "Bảng Nghiên Cứu Bóc Tách Thành Phần Cốt Lõi HARO (Ablation Study)",
        "subtitle": "Đánh giá đóng góp độc lập của Quầng Halo đa tầng, Lực đẩy repulsive và Cơ chế suy giảm phi tuyến",
        "key_insight": "Việc tích hợp đầy đủ 3 cơ chế (Halo 3 tầng, Quỹ đạo lượng tử và Năng lượng phi tuyến) mang lại tỷ lệ hội tụ và độ phủ ô nhiễm tối ưu toàn diện.",
        "badge": "Mechanism Contribution",
    },
    "thuc_te_hcmc": {
        "title": "Bảng Đối Sánh Thực Tế Địa Bàn TP.HCM: Mạng Trạm Hiện Hữu vs Tối Ưu HARO",
        "subtitle": "So sánh 10 trạm quy chuẩn hiện có (US Consulate + HealthyAir + DONRE) với mạng trạm tối ưu hóa",
        "key_insight": "Mạng trạm hiện hữu chỉ đạt 47.79% độ bao phủ và vi phạm nghiêm trọng ràng buộc khoảng cách (1.36 km < 3.0 km d_min). Trong khi đó, HARO đạt 61.78% độ bao phủ và 100% thỏa mãn trắc địa (khoảng cách 6.05 km).",
        "badge": "HCMC Real GIS Benchmark",
    },
}


def _find_csv(dataset_key: str) -> Optional[str]:
    candidates = CSV_PATHS.get(dataset_key, [])
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def get_dataset(dataset_key: str) -> Dict[str, Any]:
    """Tải và phân tích dữ liệu CSV thành định dạng JSON chuẩn cho frontend."""
    file_path = _find_csv(dataset_key)
    meta = TABLE_METADATA.get(dataset_key, {
        "title": dataset_key,
        "subtitle": "",
        "key_insight": "",
        "badge": "CSV Data",
    })

    if not file_path:
        return {
            "status": "not_found",
            "key": dataset_key,
            "metadata": meta,
            "headers": [],
            "rows": [],
            "row_count": 0,
            "error": f"Không tìm thấy tệp CSV cho '{dataset_key}'.",
        }

    try:
        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, [])
            rows = []
            for r in reader:
                if any(cell.strip() for cell in r):
                    rows.append(r)

        return {
            "status": "success",
            "key": dataset_key,
            "file_name": os.path.basename(file_path),
            "metadata": meta,
            "headers": headers,
            "rows": rows,
            "row_count": len(rows),
        }
    except Exception as e:
        return {
            "status": "error",
            "key": dataset_key,
            "metadata": meta,
            "headers": [],
            "rows": [],
            "row_count": 0,
            "error": str(e),
        }


def get_all_datasets_summary() -> List[Dict[str, Any]]:
    """Lấy danh mục tóm tắt tất cả các bộ số liệu thực nghiệm để render tab."""
    summaries = []
    for key, meta in TABLE_METADATA.items():
        path = _find_csv(key)
        available = path is not None and os.path.exists(path)
        summaries.append({
            "key": key,
            "title": meta["title"],
            "subtitle": meta["subtitle"],
            "key_insight": meta["key_insight"],
            "badge": meta["badge"],
            "available": available,
        })
    return summaries

