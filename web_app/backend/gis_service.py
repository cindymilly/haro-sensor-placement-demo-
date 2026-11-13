"""GIS Mapping and Geospatial Services for HCMC Air Quality Monitoring Optimization.

Provides:
- Exact WGS84 Bounding Box for Ho Chi Minh City metropolitan core.
- Bidirectional Linear Coordinate Mapping: Grid (x, y) <-> Geographic (Lat, Lon).
- Scale calculations: Conversion between grid cells and real-world meters/kilometers.
- Integration of 10 official existing stations and 12 real hotspots in HCMC.
- GeoJSON export for Leaflet layers (Bounding box, forbidden zones, emission sources).
"""

from __future__ import annotations

import math
from typing import Tuple, List, Dict, Any, Optional
import numpy as np

# Bounding box địa lý khu vực khảo sát trung tâm và vành đai đô thị TP.HCM
# Tọa độ WGS84 chuẩn:
# Vĩ độ (Latitude):    10.680°N đến 10.890°N (Nam -> Bắc ~ 23.3 km)
# Kinh độ (Longitude): 106.570°E đến 106.840°E (Tây -> Đông ~ 29.5 km)
HCMC_BOUNDS = {
    "lat_min": 10.680,
    "lat_max": 10.890,
    "lon_min": 106.570,
    "lon_max": 106.840,
}

# 10 trạm quan trắc thực tế hiện hữu tại TP.HCM (Baseline từ HealthyAir Mendeley, US Consulate, DONRE)
HCMC_EXISTING_STATIONS = [
    {
        "id": "STA-01",
        "name": "US Consulate General HCMC (Quận 1)",
        "source": "US Department of State / EPA AirNow",
        "type": "Reference Grade (BAM-1020)",
        "lat": 10.7828,
        "lon": 106.7001,
        "district": "Quận 1",
    },
    {
        "id": "STA-02",
        "name": "ĐHQG-HCM Linh Trung (TP. Thủ Đức)",
        "source": "HealthyAir Project (Mendeley Data)",
        "type": "Urban Background",
        "lat": 10.8699,
        "lon": 106.7960,
        "district": "TP. Thủ Đức",
    },
    {
        "id": "STA-03",
        "name": "An Lạc (Bình Tân - Nguyễn Trọng Trí)",
        "source": "HealthyAir Project (Mendeley Data)",
        "type": "Traffic",
        "lat": 10.7410,
        "lon": 106.6171,
        "district": "Bình Tân",
    },
    {
        "id": "STA-04",
        "name": "KCN Tân Bình (Tây Thạnh)",
        "source": "HealthyAir Project (Mendeley Data)",
        "type": "Industrial",
        "lat": 10.8162,
        "lon": 106.6204,
        "district": "Tân Phú / Tân Bình",
    },
    {
        "id": "STA-05",
        "name": "Bán đảo Thanh Đa (Bình Thạnh)",
        "source": "HealthyAir Project (Mendeley Data)",
        "type": "Residential",
        "lat": 10.8158,
        "lon": 106.7174,
        "district": "Bình Thạnh",
    },
    {
        "id": "STA-06",
        "name": "Nguyễn Đình Chiểu (Quận 3)",
        "source": "HealthyAir Project (Mendeley Data)",
        "type": "Traffic",
        "lat": 10.7764,
        "lon": 106.6878,
        "district": "Quận 3",
    },
    {
        "id": "STA-07",
        "name": "Trường Sơn (Quận 10)",
        "source": "HealthyAir Project (Mendeley Data)",
        "type": "Traffic & Residential",
        "lat": 10.7805,
        "lon": 106.6595,
        "district": "Quận 10",
    },
    {
        "id": "STA-08",
        "name": "Nguyễn Văn Cừ (Quận 5)",
        "source": "Sở TN&MT TP.HCM / CEM",
        "type": "Urban Mixed",
        "lat": 10.7583,
        "lon": 106.6825,
        "district": "Quận 5",
    },
    {
        "id": "STA-09",
        "name": "Ngã tư An Sương (Quận 12)",
        "source": "Sở TN&MT TP.HCM / CEM",
        "type": "Heavy Traffic Hub",
        "lat": 10.8492,
        "lon": 106.6153,
        "district": "Quận 12",
    },
    {
        "id": "STA-10",
        "name": "Cụm Cảng Cát Lái (TP. Thủ Đức)",
        "source": "Sở TN&MT TP.HCM / CEM",
        "type": "Heavy Industrial & Port",
        "lat": 10.7601,
        "lon": 106.7932,
        "district": "TP. Thủ Đức",
    },
]

# 12 Điểm nóng ô nhiễm thực tế trọng điểm tại TP.HCM
HCMC_REAL_HOTSPOTS = [
    {"id": "S01", "name": "Cảng & KCN Cát Lái (TP. Thủ Đức)", "type": "industrial", "lat": 10.7620, "lon": 106.7950, "emission_rate": 100.0, "sigma_km": 2.5},
    {"id": "S02", "name": "KCN Hiệp Phước (Nhà Bè)", "type": "industrial", "lat": 10.6850, "lon": 106.7550, "emission_rate": 95.0, "sigma_km": 3.0},
    {"id": "S03", "name": "KCN Tân Bình & Vĩnh Lộc", "type": "industrial", "lat": 10.8150, "lon": 106.6180, "emission_rate": 88.0, "sigma_km": 2.2},
    {"id": "S04", "name": "Khu Công nghệ cao SHTP & KCN Linh Trung", "type": "industrial", "lat": 10.8550, "lon": 106.7900, "emission_rate": 80.0, "sigma_km": 2.2},
    {"id": "S05", "name": "Nút giao Ngã tư An Sương (QL1A - QL22)", "type": "traffic", "lat": 10.8500, "lon": 106.6160, "emission_rate": 90.0, "sigma_km": 1.8},
    {"id": "S06", "name": "Hành lang Xa lộ Hà Nội - Cầu Sài Gòn", "type": "traffic", "lat": 10.8030, "lon": 106.7280, "emission_rate": 85.0, "sigma_km": 1.8},
    {"id": "S07", "name": "Trục Võ Văn Kiệt - Hầm Thủ Thiêm - Mai Chí Thọ", "type": "traffic", "lat": 10.7680, "lon": 106.7100, "emission_rate": 82.0, "sigma_km": 1.6},
    {"id": "S08", "name": "Nút giao Vòng xoay Mỹ Thủy & Cầu Phú Mỹ", "type": "traffic", "lat": 10.7580, "lon": 106.7650, "emission_rate": 86.0, "sigma_km": 2.0},
    {"id": "S09", "name": "Khu đại công trường KĐT Mới Thủ Thiêm", "type": "construction", "lat": 10.7720, "lon": 106.7250, "emission_rate": 65.0, "sigma_km": 1.5},
    {"id": "S10", "name": "Hành lang Bến xe Miền Đông & QL13 (Bình Thạnh)", "type": "traffic", "lat": 10.8140, "lon": 106.7120, "emission_rate": 75.0, "sigma_km": 1.6},
    {"id": "S11", "name": "Sân bay Quốc tế Tân Sơn Nhất", "type": "traffic", "lat": 10.8180, "lon": 106.6600, "emission_rate": 78.0, "sigma_km": 2.0},
    {"id": "S12", "name": "Lõi Trung tâm Tài chính Quận 1 - Quận 3", "type": "traffic", "lat": 10.7750, "lon": 106.6980, "emission_rate": 70.0, "sigma_km": 1.5},
]


def grid_to_geo(x: float, y: float, grid_m: int = 50, grid_n: int = 50) -> Tuple[float, float]:
    """Chuyển đổi chỉ số lưới (x, y) sang tọa độ địa lý WGS84 (Lat, Lon).
    
    x: trục kinh độ (West -> East), 0 .. grid_m - 1
    y: trục vĩ độ (South -> North), 0 .. grid_n - 1
    """
    b = HCMC_BOUNDS
    denom_x = max(1.0, float(grid_m - 1))
    denom_y = max(1.0, float(grid_n - 1))
    lon = b["lon_min"] + (float(x) / denom_x) * (b["lon_max"] - b["lon_min"])
    lat = b["lat_min"] + (float(y) / denom_y) * (b["lat_max"] - b["lat_min"])
    return round(lat, 6), round(lon, 6)


def geo_to_grid(lat: float, lon: float, grid_m: int = 50, grid_n: int = 50) -> Tuple[int, int, bool]:
    """Chuyển đổi tọa độ địa lý (Lat, Lon) sang chỉ số lưới (x, y).
    
    Trả về (x, y, is_inside).
    Nếu nằm ngoài Bounding Box HCMC, is_inside = False.
    """
    b = HCMC_BOUNDS
    is_inside = (b["lat_min"] <= lat <= b["lat_max"]) and (b["lon_min"] <= lon <= b["lon_max"])
    
    norm_x = (lon - b["lon_min"]) / (b["lon_max"] - b["lon_min"])
    norm_y = (lat - b["lat_min"]) / (b["lat_max"] - b["lat_min"])
    
    x = int(round(np.clip(norm_x * (grid_m - 1), 0, grid_m - 1)))
    y = int(round(np.clip(norm_y * (grid_n - 1), 0, grid_n - 1)))
    return x, y, is_inside


def get_meters_per_cell(grid_m: int = 50, grid_n: int = 50) -> Dict[str, float]:
    """Tính toán quy đổi kích thước ô lưới sang mét và kilômét.
    
    HCMC Bounding Box:
    - Chiều ngang (Lon): 29.5 km
    - Chiều dọc (Lat): 23.3 km
    """
    km_x = 29.5 / grid_m
    km_y = 23.3 / grid_n
    km_avg = (km_x + km_y) / 2.0
    return {
        "km_x": round(km_x, 3),
        "km_y": round(km_y, 3),
        "km_avg": round(km_avg, 3),
        "meters_x": round(km_x * 1000.0, 1),
        "meters_y": round(km_y * 1000.0, 1),
        "meters_avg": round(km_avg * 1000.0, 1),
    }


def cells_to_meters(cells: float, grid_m: int = 50, grid_n: int = 50) -> float:
    """Chuyển đổi bán kính theo ô lưới sang bán kính theo mét thực địa."""
    scale = get_meters_per_cell(grid_m, grid_n)
    return float(cells * scale["meters_avg"])


def meters_to_cells(meters: float, grid_m: int = 50, grid_n: int = 50) -> float:
    """Chuyển đổi khoảng cách thực địa (mét) sang số ô lưới."""
    scale = get_meters_per_cell(grid_m, grid_n)
    return float(meters / max(1.0, scale["meters_avg"]))


def get_bounding_box_geojson() -> Dict[str, Any]:
    """Tạo GeoJSON Polygon đại diện cho Bounding Box TP.HCM."""
    b = HCMC_BOUNDS
    # Tọa độ GeoJSON chuẩn theo quy ước [lon, lat]
    coords = [
        [b["lon_min"], b["lat_min"]],
        [b["lon_max"], b["lat_min"]],
        [b["lon_max"], b["lat_max"]],
        [b["lon_min"], b["lat_max"]],
        [b["lon_min"], b["lat_min"]],
    ]
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords],
        },
        "properties": {
            "name": "Khu vực khảo sát TP. Hồ Chí Minh (Bounding Box)",
            "bounds": b,
            "width_km": 29.5,
            "height_km": 23.3,
        },
    }


def get_standard_forbidden_zones_geojson(grid_m: int = 50, grid_n: int = 50) -> List[Dict[str, Any]]:
    """Tạo GeoJSON cho 2 vùng cấm trắc địa trong kịch bản thực nghiệm chuẩn của khóa luận:
    
    1. Vùng cấm 1 (Hồ nước / công viên trung tâm): [0.33M - 0.50M, 0.33N - 0.50N]
    2. Vùng cấm 2 (Khu quân sự / an ninh):          [0.08M - 0.20M, 0.58N - 0.70N]
    """
    zones = []
    
    def make_zone(x0_frac, x1_frac, y0_frac, y1_frac, name, desc):
        x0, x1 = x0_frac * (grid_m - 1), x1_frac * (grid_m - 1)
        y0, y1 = y0_frac * (grid_n - 1), y1_frac * (grid_n - 1)
        
        sw_lat, sw_lon = grid_to_geo(x0, y0, grid_m, grid_n)
        se_lat, se_lon = grid_to_geo(x1, y0, grid_m, grid_n)
        ne_lat, ne_lon = grid_to_geo(x1, y1, grid_m, grid_n)
        nw_lat, nw_lon = grid_to_geo(x0, y1, grid_m, grid_n)
        
        # GeoJSON coordinates are [lon, lat]
        coords = [
            [sw_lon, sw_lat],
            [se_lon, se_lat],
            [ne_lon, ne_lat],
            [nw_lon, nw_lat],
            [sw_lon, sw_lat],
        ]
        return {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {
                "name": name,
                "description": desc,
                "grid_bounds": {"x0": x0, "x1": x1, "y0": y0, "y1": y1},
                "color": "#ef4444",
            },
        }

    zones.append(
        make_zone(
            0.33, 0.50, 0.33, 0.50,
            "Vùng cấm 1: Vùng lõi Mặt nước / Hồ cảnh quan Trung tâm",
            "Vùng cấm trắc địa tự nhiên không thể xây dựng trạm quan trắc mặt đất.",
        )
    )
    zones.append(
        make_zone(
            0.08, 0.20, 0.58, 0.70,
            "Vùng cấm 2: Khu vực An ninh Quốc phòng / Quân sự",
            "Vùng hạn chế trắc địa đặc biệt, loại trừ khỏi tập ứng viên khả thi.",
        )
    )
    return zones


def get_hcmc_river_and_airport_mask_geojson(grid_m: int = 50, grid_n: int = 50) -> List[Dict[str, Any]]:
    """Tạo GeoJSON biểu diễn dòng chảy Sông Sài Gòn và vành đai an toàn Sân bay Tân Sơn Nhất."""
    features = []
    
    # Sân bay Tân Sơn Nhất safety zone
    tsn_coords = [
        [106.655, 10.810],
        [106.672, 10.810],
        [106.672, 10.828],
        [106.655, 10.828],
        [106.655, 10.810],
    ]
    features.append({
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [tsn_coords]},
        "properties": {
            "name": "Vành đai An toàn Hàng không Sân bay Tân Sơn Nhất",
            "description": "Vùng cấm lắp đặt thiết bị cao tầng và cảm biến trong phễu bay.",
            "color": "#f97316",
        },
    })
    
    # Hành lang Sông Sài Gòn
    river_line = []
    for lat in np.linspace(10.70, 10.88, 30):
        river_lon = 106.710 + 0.05 * math.sin((lat - 10.70) / 0.15 * math.pi)
        river_line.append([river_lon, lat])
        
    features.append({
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": river_line},
        "properties": {
            "name": "Hành lang Mặt nước Sông Sài Gòn",
            "description": "Vệt dòng chảy sông tự nhiên uốn khúc qua trung tâm TP.HCM.",
            "color": "#0284c7",
        },
    })
    
    return features

