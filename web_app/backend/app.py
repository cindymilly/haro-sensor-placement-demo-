"""FastAPI Application Server for HCMC Air Quality Monitoring Sensor Placement Demo.

Exposes REST APIs and a real-time WebSocket endpoint for streaming HARO convergence.
Serves the modern responsive Leaflet.js and TailwindCSS Single-Page Application.
"""

from __future__ import annotations

import os
import sys
import asyncio
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response

# Add paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_APP_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
FRONTEND_DIR = os.path.join(WEB_APP_DIR, "frontend")

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from .gis_service import (
        HCMC_BOUNDS,
        HCMC_EXISTING_STATIONS,
        HCMC_REAL_HOTSPOTS,
        get_bounding_box_geojson,
        get_standard_forbidden_zones_geojson,
        get_hcmc_river_and_airport_mask_geojson,
        grid_to_geo,
        geo_to_grid,
        get_meters_per_cell,
    )
    from .optimization_service import OptimizationService
    from .data_service import get_dataset, get_all_datasets_summary
except ImportError:
    from gis_service import (
        HCMC_BOUNDS,
        HCMC_EXISTING_STATIONS,
        HCMC_REAL_HOTSPOTS,
        get_bounding_box_geojson,
        get_standard_forbidden_zones_geojson,
        get_hcmc_river_and_airport_mask_geojson,
        grid_to_geo,
        geo_to_grid,
        get_meters_per_cell,
    )
    from optimization_service import OptimizationService
    from data_service import get_dataset, get_all_datasets_summary

app = FastAPI(
    title="HCMC Air Quality Sensor Network Optimization (HARO)",
    description="Web Demo Tương tác Trực quan Tối ưu hóa Vị trí Đặt Trạm Quan trắc Chất lượng Không khí bằng Thuật toán HARO",
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optimization service singleton
service = OptimizationService()


# --------------------------------------------------------------------------- #
# Request Schemas
# --------------------------------------------------------------------------- #
class ProblemConfigRequest(BaseModel):
    scenario: str = Field(default="standard_50x50", description="Kịch bản ô nhiễm")
    grid_m: int = Field(default=50, ge=30, le=100)
    grid_n: int = Field(default=50, ge=30, le=100)
    k: int = Field(default=10, ge=3, le=30, description="Số lượng trạm K")
    coverage_radius: Optional[float] = Field(default=None, description="Bán kính phủ cảm biến (ô lưới)")
    min_station_distance: Optional[float] = Field(default=None, description="Khoảng cách trạm tối thiểu (ô lưới)")
    penalty: float = Field(default=10.0, ge=0.0)
    seed: Optional[int] = Field(default=42)


class OptimizeRequest(ProblemConfigRequest):
    max_iterations: int = Field(default=50, ge=10, le=200)
    pop_size: int = Field(default=30, ge=10, le=60)


class UserStationsRequest(ProblemConfigRequest):
    stations: List[Dict[str, Any]] = Field(default_factory=list, description="Danh sách tọa độ lat/lon hoặc grid_x/grid_y")


class TournamentRequest(ProblemConfigRequest):
    max_iterations: int = Field(default=50, ge=10, le=100)
    pop_size: int = Field(default=30, ge=10, le=60)


class SingleAlgorithmRequest(ProblemConfigRequest):
    algorithm: str = Field(default="HARO")
    max_iterations: int = Field(default=50, ge=10, le=100)
    pop_size: int = Field(default=30, ge=10, le=60)


class ResilienceRequest(ProblemConfigRequest):
    stations: Optional[List[Dict[str, Any]]] = None
    failure_type: str = Field(default="random_1", description="random_1, random_2, worst_case, custom")
    disabled_station_ids: Optional[List[int]] = None


class ReconstructionRequest(BaseModel):
    scenario: str = Field(default="standard_50x50")
    grid_m: int = Field(default=50)
    grid_n: int = Field(default=50)
    stations: Optional[List[Dict[str, Any]]] = None
    power: float = Field(default=2.0, ge=0.5, le=5.0)


class ExportReportRequest(BaseModel):
    scenario: str = Field(default="standard_50x50")
    grid_m: int = Field(default=50)
    grid_n: int = Field(default=50)
    k: int = Field(default=10)
    coverage_radius: Optional[float] = None
    min_station_distance: Optional[float] = None
    stations: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[Dict[str, Any]] = None
    convergence: Optional[List[float]] = None


# --------------------------------------------------------------------------- #
# REST Endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/environment")
async def get_environment(
    scenario: str = Query(default="standard_50x50"),
    grid_m: int = Query(default=50, ge=30, le=100),
    grid_n: int = Query(default=50, ge=30, le=100),
    k: int = Query(default=10, ge=3, le=30),
    r_cover: Optional[float] = Query(default=None),
    d_min: Optional[float] = Query(default=None),
):
    """Lấy dữ liệu môi trường, phân bố nồng độ ô nhiễm, nguồn phát và mặt nạ vùng cấm."""
    try:
        env_data = service.get_environment_data(
            scenario=scenario,
            grid_m=grid_m,
            grid_n=grid_n,
            k=k,
        )
        return env_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/geojson/bounds")
async def get_bounds_geojson():
    """Lấy GeoJSON Bounding Box khảo sát TP.HCM."""
    return get_bounding_box_geojson()


@app.get("/api/geojson/forbidden_zones")
async def get_forbidden_zones_geojson(
    scenario: str = Query(default="standard_50x50"),
    grid_m: int = Query(default=50),
    grid_n: int = Query(default=50),
):
    """Lấy GeoJSON các vùng cấm trắc địa tương ứng với kịch bản được chọn."""
    if scenario == "standard_50x50":
        return {
            "type": "FeatureCollection",
            "features": get_standard_forbidden_zones_geojson(grid_m, grid_n),
        }
    else:
        return {
            "type": "FeatureCollection",
            "features": get_hcmc_river_and_airport_mask_geojson(grid_m, grid_n),
        }


@app.post("/api/random_placement")
async def run_random(req: ProblemConfigRequest):
    """Chế độ 2: Sinh ngẫu nhiên K trạm quan trắc (Baseline đối chứng)."""
    try:
        res = service.run_random_placement(
            scenario=req.scenario,
            grid_m=req.grid_m,
            grid_n=req.grid_n,
            k=req.k,
            coverage_radius=req.coverage_radius,
            min_station_distance=req.min_station_distance,
            seed=req.seed,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evaluate_user_stations")
async def evaluate_user_click(req: UserStationsRequest):
    """Chế độ 3: Người dùng tự click chọn vị trí trạm trên bản đồ trực quan."""
    try:
        res = service.evaluate_user_stations(
            stations_geo=req.stations,
            scenario=req.scenario,
            grid_m=req.grid_m,
            grid_n=req.grid_n,
            k=req.k,
            coverage_radius=req.coverage_radius,
            min_station_distance=req.min_station_distance,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export_report_figure")
async def export_report_figure(req: ExportReportRequest):
    """Trích xuất hình ảnh báo cáo học thuật độ phân giải cao kèm thông số và bản đồ."""
    try:
        img_bytes = service.generate_academic_report_figure(
            scenario=req.scenario,
            grid_m=req.grid_m,
            grid_n=req.grid_n,
            k=req.k,
            coverage_radius=req.coverage_radius,
            min_station_distance=req.min_station_distance,
            stations_data=req.stations,
            metrics=req.metrics,
            convergence=req.convergence,
        )
        return Response(content=img_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/existing_stations")
async def get_existing_stations():
    """Lấy và đánh giá mạng lưới 10 trạm quan trắc thực tế hiện hữu TP.HCM."""
    try:
        res = service.get_existing_hcmc_stations_evaluation(grid_m=50, grid_n=50)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/run_tournament")
async def run_tournament(req: TournamentRequest):
    """Combo 1: Tổ chức Đấu trường 5 Thuật toán (HARO vs ARO vs PSO vs MVO vs GA)."""
    try:
        res = service.run_algorithm_tournament(
            scenario=req.scenario,
            grid_m=req.grid_m,
            grid_n=req.grid_n,
            k=req.k,
            coverage_radius=req.coverage_radius,
            min_station_distance=req.min_station_distance,
            max_iterations=req.max_iterations,
            pop_size=req.pop_size,
            seed=req.seed if req.seed is not None else 42,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/run_single_algorithm")
async def run_single_algo(req: SingleAlgorithmRequest):
    """Chạy độc lập 1 thuật toán bất kỳ trong 5 giải thuật."""
    try:
        res = service.run_single_algorithm(
            algo_code=req.algorithm,
            scenario=req.scenario,
            grid_m=req.grid_m,
            grid_n=req.grid_n,
            k=req.k,
            coverage_radius=req.coverage_radius,
            min_station_distance=req.min_station_distance,
            max_iterations=req.max_iterations,
            pop_size=req.pop_size,
            seed=req.seed if req.seed is not None else 42,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/resilience_simulation")
async def run_resilience(req: ResilienceRequest):
    """Combo 2: Mô phỏng kiểm định độ bền vững (Resilience Test) khi xảy ra sự cố ngắt trạm."""
    try:
        res = service.run_resilience_simulation(
            scenario=req.scenario,
            grid_m=req.grid_m,
            grid_n=req.grid_n,
            k=req.k,
            coverage_radius=req.coverage_radius,
            min_station_distance=req.min_station_distance,
            stations_data=req.stations,
            failure_type=req.failure_type,
            disabled_station_ids=req.disabled_station_ids,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/spatial_reconstruction")
async def run_spatial_recon(req: ReconstructionRequest):
    """Combo 2: Tái tạo trường nồng độ ô nhiễm toàn thành phố bằng nội suy IDW / Kriging."""
    try:
        res = service.run_spatial_reconstruction(
            scenario=req.scenario,
            grid_m=req.grid_m,
            grid_n=req.grid_n,
            stations_data=req.stations,
            power=req.power,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/thesis_data")
async def get_thesis_data(dataset: str = Query(default="tong_hop")):
    """Lấy dữ liệu bảng thực nghiệm từ các file CSV của khóa luận."""
    res = get_dataset(dataset)
    return res


@app.get("/api/thesis_datasets_summary")
async def get_thesis_summary():
    """Lấy danh mục tóm tắt tất cả các bảng thực nghiệm."""
    return get_all_datasets_summary()


# --------------------------------------------------------------------------- #
# WebSocket Streaming for HARO Optimization
# --------------------------------------------------------------------------- #
@app.websocket("/ws/optimize")
async def websocket_optimize(websocket: WebSocket):
    """WebSocket endpoint để stream tiến trình hội tụ và vị trí trạm HARO trực tiếp từng thế hệ."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action", "start")

            if action == "start":
                scenario = data.get("scenario", "standard_50x50")
                grid_m = int(data.get("grid_m", 50))
                grid_n = int(data.get("grid_n", 50))
                k = int(data.get("k", 10))
                coverage_radius = data.get("coverage_radius")
                if coverage_radius is not None:
                    coverage_radius = float(coverage_radius)
                min_station_distance = data.get("min_station_distance")
                if min_station_distance is not None:
                    min_station_distance = float(min_station_distance)
                max_iterations = int(data.get("max_iterations", 50))
                pop_size = int(data.get("pop_size", 30))
                seed = int(data.get("seed", 42))
                delay_ms = int(data.get("delay_ms", 25))

                await websocket.send_json({
                    "type": "started",
                    "scenario": scenario,
                    "k": k,
                    "max_iterations": max_iterations,
                    "pop_size": pop_size,
                })

                stream = service.run_haro_streaming(
                    scenario=scenario,
                    grid_m=grid_m,
                    grid_n=grid_n,
                    k=k,
                    coverage_radius=coverage_radius,
                    min_station_distance=min_station_distance,
                    max_iterations=max_iterations,
                    pop_size=pop_size,
                    seed=seed,
                )

                for msg in stream:
                    await websocket.send_json(msg)
                    if delay_ms > 0:
                        await asyncio.sleep(delay_ms / 1000.0)

            elif action == "stop":
                await websocket.send_json({"type": "stopped", "message": "Thuật toán đã được dừng bởi người dùng."})
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Static Files & Single Page App Serving
# --------------------------------------------------------------------------- #
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def serve_index():
    """Phục vụ file giao diện HTML5 chính."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(
        status_code=404,
        content={"error": "File frontend/index.html chưa được khởi tạo."},
    )

