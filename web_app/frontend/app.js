/**
 * Application Controller for HCMC Air Quality Monitoring Sensor Placement Demo.
 * 
 * Features:
 * - Leaflet Map with OpenStreetMap and CartoDB base layers.
 * - Exact Linear Coordinate Mapping between Grid 50x50 and HCMC WGS84 bounds.
 * - Mode 1: HARO Optimization with real-time WebSocket generation streaming.
 * - Mode 2: Random Uniform Baseline generation.
 * - Mode 3: Interactive User Click placement with live constraint and coverage evaluation.
 * - 3-Case Comparison Table & Side-by-Side Chart.js graphs.
 * - Thesis Experimental CSV Database Viewer modal with live search & export.
 */

// Global State
const STATE = {
  scenario: 'standard_50x50',
  gridM: 50,
  gridN: 50,
  k: 10,
  rCover: 8.0,
  dMin: 10.0,
  maxIterations: 50,
  bounds: {
    lat_min: 10.680,
    lat_max: 10.890,
    lon_min: 106.570,
    lon_max: 106.840,
  },
  metersPerCell: {
    meters_avg: 528.0,
    km_avg: 0.528,
  },
  activeMode: null, // 'haro' | 'random' | 'user' | 'tournament' | 'single'
  isUserClickMode: false,
  userStations: [],
  currentStations: [], // Trạm hiển thị hiện tại trên bản đồ
  disabledStationIds: new Set(), // Set các station_index bị ngắt kết nối
  environmentData: null,
  socket: null,
  charts: {
    convergence: null,
    comparison: null,
    tournament: null,
  },
  thesisCsvData: {
    currentKey: 'tong_hop',
    headers: [],
    rows: [],
  },
  tournamentResults: null,
  results: {
    haro: null,
    random: null,
    user: null,
    official: null,
  }
};

// Map & Layer References
let map = null;
let currentTileLayer = null;
const layers = {
  boundingBox: null,
  forbiddenZones: null,
  sources: null,
  pollutionHeat: null,
  coverageHeat: null,
  haroStations: null,
  randomStations: null,
  userStations: null,
  officialStations: null,
  violatingLines: null,
  haloRings: null,
  reconHeat: null,
};

// ========================================================================= //
// 1. COORDINATE CONVERSION UTILITIES                                        //
// ========================================================================= //
function gridToGeo(x, y) {
  const b = STATE.bounds;
  const lon = b.lon_min + (x / (STATE.gridM - 1)) * (b.lon_max - b.lon_min);
  const lat = b.lat_min + (y / (STATE.gridN - 1)) * (b.lat_max - b.lat_min);
  return [Number(lat.toFixed(6)), Number(lon.toFixed(6))];
}

function geoToGrid(lat, lon) {
  const b = STATE.bounds;
  const isInside = (lat >= b.lat_min && lat <= b.lat_max && lon >= b.lon_min && lon <= b.lon_max);
  const normX = (lon - b.lon_min) / (b.lon_max - b.lon_min);
  const normY = (lat - b.lat_min) / (b.lat_max - b.lat_min);
  const x = Math.round(Math.max(0, Math.min(STATE.gridM - 1, normX * (STATE.gridM - 1))));
  const y = Math.round(Math.max(0, Math.min(STATE.gridN - 1, normY * (STATE.gridN - 1))));
  return { x, y, isInside };
}

function cellsToMeters(cells) {
  return cells * STATE.metersPerCell.meters_avg;
}

// ========================================================================= //
// 2. INITIALIZATION                                                         //
// ========================================================================= //
document.addEventListener('DOMContentLoaded', async () => {
  if (window.lucide) {
    lucide.createIcons();
  }

  initLeafletMap();
  initCharts();
  initEventListeners();

  await loadEnvironmentData();
  await loadOfficialBaseline();
});

// ========================================================================= //
// 3. LEAFLET MAP INITIALIZATION & TILE LAYERS                               //
// ========================================================================= //
function initLeafletMap() {
  const centerLat = (STATE.bounds.lat_min + STATE.bounds.lat_max) / 2.0;
  const centerLon = (STATE.bounds.lon_min + STATE.bounds.lon_max) / 2.0;

  map = L.map('map', {
    center: [centerLat, centerLon],
    zoom: 12,
    zoomControl: true,
    attributionControl: false,
  });

  // Base tile layers
  setTileLayer('dark');

  // Initialize Layer Groups
  layers.boundingBox = L.layerGroup().addTo(map);
  layers.forbiddenZones = L.layerGroup().addTo(map);
  layers.sources = L.layerGroup().addTo(map);
  layers.haroStations = L.layerGroup().addTo(map);
  layers.randomStations = L.layerGroup().addTo(map);
  layers.userStations = L.layerGroup().addTo(map);
  layers.officialStations = L.layerGroup().addTo(map);
  layers.violatingLines = L.layerGroup().addTo(map);
  layers.haloRings = L.layerGroup().addTo(map);

  // Map Mouse Tracking (HUD)
  map.on('mousemove', (e) => {
    const lat = e.latlng.lat;
    const lon = e.latlng.lng;
    const { x, y, isInside } = geoToGrid(lat, lon);

    document.getElementById('hudCoords').textContent = `${lat.toFixed(4)}°N, ${lon.toFixed(4)}°E`;
    document.getElementById('hudGridCoords').textContent = `X: ${x}, Y: ${y}`;

    const badge = document.getElementById('hudBoundsBadge');
    if (isInside) {
      badge.textContent = 'Trong vùng khảo sát';
      badge.className = 'px-1.5 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-300';
    } else {
      badge.textContent = 'Ngoài vùng khảo sát TP.HCM';
      badge.className = 'px-1.5 py-0.5 rounded text-[10px] bg-rose-500/20 text-rose-300';
    }
  });

  // Map Click Handler for Mode 3 (User Click Type)
  map.on('click', handleMapClick);
}

function setTileLayer(type) {
  if (currentTileLayer) {
    map.removeLayer(currentTileLayer);
  }

  let url = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';
  if (type === 'light') {
    url = 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png';
  } else if (type === 'osm') {
    url = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
  }

  currentTileLayer = L.tileLayer(url, { maxZoom: 19 }).addTo(map);
}

// ========================================================================= //
// 4. LOAD ENVIRONMENT & MAP LAYERS                                          //
// ========================================================================= //
async function loadEnvironmentData() {
  try {
    updateStatus('Đang tải kịch bản môi trường...', 'loading');
    const resp = await fetch(`/api/environment?scenario=${STATE.scenario}&grid_m=${STATE.gridM}&grid_n=${STATE.gridN}&k=${STATE.k}`);
    const data = await resp.json();

    STATE.environmentData = data;
    STATE.bounds = data.bounds;
    STATE.metersPerCell = data.scale;
    STATE.rCover = data.R_cover_cells;
    STATE.dMin = data.d_min_cells;

    // Update Slider text
    updateSliderDisplays();

    // Render Bounding Box
    renderBoundingBox();

    // Render Forbidden Zones
    await renderForbiddenZones();

    // Render Emission Sources
    renderEmissionSources(data.sources);

    // Render Pollution Heatmap
    renderPollutionHeatmap(data.heat_points);

    updateStatus('Hệ thống sẵn sàng', 'ready');
  } catch (err) {
    console.error('Failed to load environment:', err);
    updateStatus('Lỗi tải dữ liệu kịch bản', 'error');
  }
}

function renderBoundingBox() {
  layers.boundingBox.clearLayers();
  const b = STATE.bounds;
  const bounds = [[b.lat_min, b.lon_min], [b.lat_max, b.lon_max]];

  const rect = L.rectangle(bounds, {
    color: '#3b82f6',
    weight: 2,
    dashArray: '6, 6',
    fillColor: '#3b82f6',
    fillOpacity: 0.03,
  }).addTo(layers.boundingBox);

  rect.bindTooltip('VÙNG KHẢO SÁT TRUNG TÂM TP. HỒ CHÍ MINH (BOUNDING BOX 50x50)', {
    sticky: true,
    className: 'bg-gray-900 text-blue-400 text-xs px-2 py-1 rounded border border-blue-500/30',
  });

  map.fitBounds(bounds, { padding: [30, 30] });
}

async function renderForbiddenZones() {
  layers.forbiddenZones.clearLayers();
  try {
    const resp = await fetch(`/api/geojson/forbidden_zones?scenario=${STATE.scenario}&grid_m=${STATE.gridM}&grid_n=${STATE.gridN}`);
    const geojson = await resp.json();

    L.geoJSON(geojson, {
      style: (feature) => ({
        color: feature.properties.color || '#ef4444',
        weight: 2,
        fillColor: feature.properties.color || '#ef4444',
        fillOpacity: 0.25,
        dashArray: '4, 4',
      }),
      onEachFeature: (feature, layer) => {
        layer.bindTooltip(`<b>${feature.properties.name}</b><br><span class="text-[11px] text-gray-300">${feature.properties.description}</span>`, {
          className: 'bg-gray-900 text-rose-300 text-xs p-2 rounded border border-rose-500/30',
        });
      }
    }).addTo(layers.forbiddenZones);
  } catch (err) {
    console.error('Failed to load forbidden zones:', err);
  }
}

function renderEmissionSources(sources) {
  layers.sources.clearLayers();
  if (!sources) return;

  const typeConfig = {
    traffic: { color: '#ffffff', icon: 'car', label: 'Giao thông' },
    industrial: { color: '#38bdf8', icon: 'factory', label: 'Công nghiệp' },
    construction: { color: '#fbbf24', icon: 'hammer', label: 'Xây dựng' },
    noise: { color: '#9ca3af', icon: 'radio', label: 'Nhiễu nền' },
  };

  sources.forEach((s) => {
    const conf = typeConfig[s.type] || typeConfig.traffic;
    const marker = L.circleMarker([s.lat, s.lon], {
      radius: 6,
      fillColor: conf.color,
      color: '#000000',
      weight: 1.5,
      opacity: 0.9,
      fillOpacity: 0.9,
    }).addTo(layers.sources);

    marker.bindPopup(`
      <div class="text-xs space-y-1">
        <div class="font-bold text-white flex items-center gap-1.5">
          <span class="w-2 h-2 rounded-full" style="background:${conf.color}"></span>
          ${s.name}
        </div>
        <div class="text-gray-300">Phân loại: <span class="text-emerald-400">${conf.label}</span></div>
        <div class="text-gray-300">Cường độ phát thải Q: <span class="font-mono text-amber-400">${s.emission_rate}</span></div>
        <div class="text-gray-300">Độ phân tán σ: <span class="font-mono text-cyan-400">${s.sigma}</span></div>
        <div class="text-gray-400 text-[10px]">Tọa độ: (${s.lat.toFixed(4)}, ${s.lon.toFixed(4)}) | Ô: [${s.grid_x}, ${s.grid_y}]</div>
      </div>
    `);
  });
}

function renderPollutionHeatmap(heatPoints) {
  if (layers.pollutionHeat) {
    map.removeLayer(layers.pollutionHeat);
  }
  if (!heatPoints || !heatPoints.length) return;

  // EPA Colormap standard gradient
  layers.pollutionHeat = L.heatLayer(heatPoints, {
    radius: 28,
    blur: 18,
    maxZoom: 16,
    max: 1.0,
    gradient: {
      0.15: '#00A65A', // Green (Good)
      0.35: '#7CD100', // Lime
      0.55: '#FFF200', // Yellow (Moderate)
      0.75: '#FF8C00', // Orange (Unhealthy)
      0.90: '#E8291C', // Red (Very Unhealthy)
      1.00: '#7A1FA2', // Purple (Hazardous)
    }
  });

  if (document.getElementById('chkHeatmap').checked) {
    layers.pollutionHeat.addTo(map);
  }
}

// ========================================================================= //
// 5. MODE 1: HARO OPTIMIZATION (REAL-TIME WEBSOCKET STREAMING)               //
// ========================================================================= //
function startHaroOptimization() {
  if (STATE.socket && STATE.socket.readyState === WebSocket.OPEN) {
    STATE.socket.close();
  }

  // Switch UI states
  STATE.activeMode = 'haro';
  disableUserClickMode();
  updateModeBadge('HARO: Đang tối ưu hóa vị trí...', 'haro');

  document.getElementById('btnRunHaro').classList.add('hidden');
  document.getElementById('btnStopHaro').classList.remove('hidden');

  // Reset convergence chart data
  STATE.charts.convergence.data.labels = [];
  STATE.charts.convergence.data.datasets[0].data = [];
  STATE.charts.convergence.update();

  // Clear previous layers
  layers.haroStations.clearLayers();
  layers.randomStations.clearLayers();
  layers.userStations.clearLayers();
  layers.violatingLines.clearLayers();

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/optimize`;

  STATE.socket = new WebSocket(wsUrl);

  STATE.socket.onopen = () => {
    updateStatus('Đang stream tiến trình hội tụ HARO...', 'loading');
    STATE.socket.send(JSON.stringify({
      action: 'start',
      scenario: STATE.scenario,
      grid_m: STATE.gridM,
      grid_n: STATE.gridN,
      k: STATE.k,
      coverage_radius: STATE.rCover,
      min_station_distance: STATE.dMin,
      max_iterations: STATE.maxIterations,
      pop_size: 30,
      seed: 42,
      delay_ms: 25,
    }));
  };

  STATE.socket.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'iteration') {
      handleHaroIteration(data);
    } else if (data.type === 'completed') {
      handleHaroCompleted(data);
    } else if (data.type === 'error') {
      alert(`Lỗi tối ưu hóa: ${data.error}`);
      stopHaroOptimization();
    }
  };

  STATE.socket.onclose = () => {
    document.getElementById('btnRunHaro').classList.remove('hidden');
    document.getElementById('btnStopHaro').classList.add('hidden');
  };

  STATE.socket.onerror = (err) => {
    console.error('WebSocket error:', err);
    stopHaroOptimization();
  };
}

function stopHaroOptimization() {
  if (STATE.socket) {
    STATE.socket.send(JSON.stringify({ action: 'stop' }));
    STATE.socket.close();
    STATE.socket = null;
  }
  document.getElementById('btnRunHaro').classList.remove('hidden');
  document.getElementById('btnStopHaro').classList.add('hidden');
  updateStatus('Hệ thống sẵn sàng', 'ready');
  updateModeBadge('HARO: Đã hoàn tất hoặc dừng', 'haro');
}

function handleHaroIteration(data) {
  // 1. Update Convergence Chart
  STATE.charts.convergence.data.labels.push(`G${data.iteration}`);
  STATE.charts.convergence.data.datasets[0].data.push(data.coverage_percent);
  STATE.charts.convergence.update('none');

  // 2. Update Dashboard Pills
  document.getElementById('dashFeCount').textContent = `FE: ${data.fe.toLocaleString()} / 3,030`;
  const haloBadge = document.getElementById('dashHaloLayerBadge');
  haloBadge.classList.remove('hidden');
  haloBadge.textContent = `Quầng Halo: ${data.halo_layer.toUpperCase()} (E = ${data.energy})`;

  // 3. Update Map HUD Widget for Halo & Energy
  updateHaloHUD(data.halo_layer, data.energy);

  // 4. Render Halo Concentric Rings
  renderHaloRings(data.stations, data.halo_layer);

  // 5. Render Stations on Leaflet
  renderStationsOnMap(data.stations, 'haro');

  // 6. Update Row 1 in Comparison Table
  document.getElementById('haroK').textContent = STATE.k;
  document.getElementById('haroCov').textContent = `${data.coverage_percent.toFixed(2)}%`;
  document.getElementById('haroFit').textContent = data.best_fitness.toFixed(5);
  document.getElementById('haroOverlap').textContent = `${data.overlap_percent.toFixed(1)}%`;
  document.getElementById('haroMinDist').textContent = `${data.min_pairwise_distance_cells} ô (${data.min_pairwise_distance_km} km)`;
  
  const feasibleBadge = document.getElementById('haroFeasible');
  if (data.feasible) {
    feasibleBadge.innerHTML = `<span class="px-2 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-400 font-bold">100% Khả thi</span>`;
  } else {
    feasibleBadge.innerHTML = `<span class="px-2 py-0.5 rounded text-[10px] bg-amber-500/20 text-amber-400 font-bold">Vi phạm d_min</span>`;
  }
  document.getElementById('haroTime').textContent = `${data.runtime}s`;
}

function handleHaroCompleted(data) {
  STATE.results.haro = data;
  STATE.currentStations = data.stations;
  STATE.disabledStationIds.clear();
  stopHaroOptimization();
  updateComparisonBarChart();
  updateStatus('Tối ưu HARO hoàn tất thành công!', 'ready');
  updateModeBadge(`HARO Hoàn Tất: Độ bao phủ ${data.coverage_percent.toFixed(2)}%`, 'haro');
}

// ========================================================================= //
// 6. MODE 2: RANDOM PLACEMENT (BASELINE ĐỐI CHỨNG)                          //
// ========================================================================= //
async function runRandomPlacement() {
  try {
    STATE.activeMode = 'random';
    disableUserClickMode();
    updateStatus('Đang sinh mạng trạm ngẫu nhiên...', 'loading');
    updateModeBadge('Random: Đang tính toán độ bao phủ...', 'random');

    // Clear previous layers
    layers.haroStations.clearLayers();
    layers.randomStations.clearLayers();
    layers.userStations.clearLayers();
    layers.violatingLines.clearLayers();

    const resp = await fetch('/api/random_placement', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        scenario: STATE.scenario,
        grid_m: STATE.gridM,
        grid_n: STATE.gridN,
        k: STATE.k,
        coverage_radius: STATE.rCover,
        min_station_distance: STATE.dMin,
      })
    });

    const data = await resp.json();
    STATE.results.random = data;

    // Render stations on Map
    renderStationsOnMap(data.stations, 'random');

    // Update Row 2 in Comparison Table
    document.getElementById('randK').textContent = STATE.k;
    document.getElementById('randCov').textContent = `${data.coverage_percent.toFixed(2)}%`;
    document.getElementById('randFit').textContent = data.penalized_fitness.toFixed(5);
    document.getElementById('randOverlap').textContent = `${data.overlap_percent.toFixed(1)}%`;
    document.getElementById('randMinDist').textContent = `${data.min_pairwise_distance_cells} ô (${data.min_pairwise_distance_km} km)`;

    const feasibleBadge = document.getElementById('randFeasible');
    if (data.feasible) {
      feasibleBadge.innerHTML = `<span class="px-2 py-0.5 rounded text-[10px] bg-blue-500/20 text-blue-400 font-bold">Khả thi</span>`;
    } else {
      feasibleBadge.innerHTML = `<span class="px-2 py-0.5 rounded text-[10px] bg-rose-500/20 text-rose-400 font-bold">Vi phạm d_min</span>`;
    }
    document.getElementById('randTime').textContent = `${data.runtime_seconds}s`;

    updateComparisonBarChart();
    updateStatus('Sinh mạng trạm ngẫu nhiên hoàn tất', 'ready');
    updateModeBadge(`Random Hoàn Tất: Độ bao phủ ${data.coverage_percent.toFixed(2)}%`, 'random');
  } catch (err) {
    console.error('Random placement error:', err);
    updateStatus('Lỗi sinh ngẫu nhiên', 'error');
  }
}

// ========================================================================= //
// 7. MODE 3: USER CLICK PLACEMENT (CLICK THỦ CÔNG)                          //
// ========================================================================= //
function toggleUserClickMode() {
  if (STATE.isUserClickMode) {
    disableUserClickMode();
  } else {
    enableUserClickMode();
  }
}

function enableUserClickMode() {
  STATE.isUserClickMode = true;
  STATE.activeMode = 'user';

  document.getElementById('btnToggleUserClick').classList.add('ring-2', 'ring-purple-400');
  document.getElementById('btnUserClickText').textContent = 'CHẾ ĐỘ CLICK: ĐANG BẬT (CLICK TRÊN BẢN ĐỒ)';
  document.getElementById('userClickControls').classList.remove('hidden');
  document.getElementById('map').classList.add('cursor-crosshair-map');

  updateModeBadge(`Click Mode: Nhấp vào bản đồ để đặt trạm (${STATE.userStations.length}/${STATE.k})`, 'user');

  // Clear other mode stations to highlight user manual placement
  layers.haroStations.clearLayers();
  layers.randomStations.clearLayers();
}

function disableUserClickMode() {
  STATE.isUserClickMode = false;
  document.getElementById('btnToggleUserClick').classList.remove('ring-2', 'ring-purple-400');
  document.getElementById('btnUserClickText').textContent = 'CHẾ ĐỘ 3: CLICK THỦ CÔNG (USER CLICK)';
  document.getElementById('userClickControls').classList.add('hidden');
  document.getElementById('map').classList.remove('cursor-crosshair-map');
}

async function handleMapClick(e) {
  if (!STATE.isUserClickMode) return;

  const lat = e.latlng.lat;
  const lon = e.latlng.lng;
  const { x, y, isInside } = geoToGrid(lat, lon);

  // Check if click is inside HCMC bounding box
  if (!isInside) {
    alert(`⚠️ Tọa độ (${lat.toFixed(4)}, ${lon.toFixed(4)}) nằm ngoài vùng khảo sát TP.HCM!\nVui lòng nhấp chuột bên trong khung chữ nhật giới hạn.`);
    return;
  }

  // Add station to user list
  STATE.userStations.push({ lat, lon, grid_x: x, grid_y: y });
  await evaluateUserStations();
}

async function evaluateUserStations() {
  const count = STATE.userStations.length;
  document.getElementById('userStationsCountBadge').textContent = `Đã đặt: ${count}/${STATE.k}`;

  if (count === 0) {
    layers.userStations.clearLayers();
    layers.violatingLines.clearLayers();
    document.getElementById('userK').textContent = '--';
    document.getElementById('userCov').textContent = '--%';
    document.getElementById('userFit').textContent = '--';
    document.getElementById('userOverlap').textContent = '--%';
    document.getElementById('userMinDist').textContent = '--';
    document.getElementById('userFeasible').innerHTML = `<span class="px-2 py-0.5 rounded text-[10px] bg-gray-800 text-gray-400">Chưa đặt trạm</span>`;
    document.getElementById('userTime').textContent = '--s';
    return;
  }

  try {
    updateStatus('Đang tính toán độ bao phủ trạm thủ công...', 'loading');
    const resp = await fetch('/api/evaluate_user_stations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        scenario: STATE.scenario,
        grid_m: STATE.gridM,
        grid_n: STATE.gridN,
        k: STATE.k,
        coverage_radius: STATE.rCover,
        min_station_distance: STATE.dMin,
        stations: STATE.userStations,
      })
    });

    const data = await resp.json();
    STATE.results.user = data;

    // Render stations on Map
    renderStationsOnMap(data.stations, 'user');

    // Draw violating distance lines between pairs if d < d_min
    renderViolatingPairsLines(data.violating_pairs, data.stations);

    // Update Row 3 in Comparison Table
    document.getElementById('userK').textContent = `${count}/${STATE.k}`;
    document.getElementById('userCov').textContent = `${data.coverage_percent.toFixed(2)}%`;
    document.getElementById('userFit').textContent = data.penalized_fitness.toFixed(5);
    document.getElementById('userOverlap').textContent = `${data.overlap_percent.toFixed(1)}%`;
    document.getElementById('userMinDist').textContent = `${data.min_pairwise_distance_cells} ô (${data.min_pairwise_distance_km} km)`;

    const feasibleBadge = document.getElementById('userFeasible');
    if (data.feasible && count === STATE.k) {
      feasibleBadge.innerHTML = `<span class="px-2 py-0.5 rounded text-[10px] bg-purple-500/20 text-purple-400 font-bold">100% Khả thi</span>`;
    } else if (!data.feasible) {
      feasibleBadge.innerHTML = `<span class="px-2 py-0.5 rounded text-[10px] bg-rose-500/20 text-rose-400 font-bold">Vi phạm (${data.violating_pairs_count} cặp < d_min)</span>`;
    } else {
      feasibleBadge.innerHTML = `<span class="px-2 py-0.5 rounded text-[10px] bg-amber-500/20 text-amber-400 font-bold">Đang đặt ${count}/${STATE.k}</span>`;
    }
    document.getElementById('userTime').textContent = `${data.runtime_seconds}s`;

    updateComparisonBarChart();
    updateStatus(`Đã cập nhật ${count} trạm thủ công`, 'ready');
    updateModeBadge(`User Click: ${count}/${STATE.k} trạm | Độ phủ ${data.coverage_percent.toFixed(2)}%`, 'user');
  } catch (err) {
    console.error('Error evaluating user stations:', err);
    updateStatus('Lỗi tính toán trạm thủ công', 'error');
  }
}

function undoLastUserStation() {
  if (STATE.userStations.length > 0) {
    STATE.userStations.pop();
    evaluateUserStations();
  }
}

function clearAllUserStations() {
  STATE.userStations = [];
  evaluateUserStations();
}

// ========================================================================= //
// 8. RENDER STATIONS, CIRCLES & VIOLATING LINES ON MAP                      //
// ========================================================================= //
function renderStationsOnMap(stations, mode) {
  const targetLayer = mode === 'haro' ? layers.haroStations : (mode === 'random' ? layers.randomStations : layers.userStations);
  targetLayer.clearLayers();
  if (!stations || !stations.length) return;

  STATE.currentStations = stations;

  const colorConfig = {
    haro: { pin: '#10b981', circle: '#10b981', label: 'Trạm HARO Tối Ưu' },
    random: { pin: '#3b82f6', circle: '#3b82f6', label: 'Trạm Ngẫu Nhiên' },
    user: { pin: '#a855f7', circle: '#a855f7', label: 'Trạm Thủ Công (User)' },
  };
  const baseConf = colorConfig[mode] || colorConfig.haro;
  const radiusMeters = cellsToMeters(STATE.rCover);

  stations.forEach((st, idx) => {
    const isOffline = STATE.disabledStationIds.has(idx);
    const pinColor = isOffline ? '#ef4444' : baseConf.pin;
    const circleColor = isOffline ? '#ef4444' : baseConf.circle;

    // 1. Coverage Circle Buffer
    if (document.getElementById('chkStationCircles').checked) {
      L.circle([st.lat, st.lon], {
        radius: radiusMeters,
        color: circleColor,
        weight: isOffline ? 1.5 : 1.5,
        fillColor: circleColor,
        fillOpacity: isOffline ? 0.04 : 0.12,
        dashArray: isOffline ? '5, 5' : (mode === 'user' ? '4, 4' : null),
      }).addTo(targetLayer);
    }

    // 2. Custom Station Marker
    const iconHtml = `
      <div class="relative flex items-center justify-center">
        <div class="w-6 h-6 rounded-full border-2 ${isOffline ? 'border-rose-300 pulse-offline' : 'border-white ' + (mode === 'haro' ? 'pulse-haro' : '')} flex items-center justify-center text-[10px] font-bold text-white shadow-lg transition-all" style="background-color: ${pinColor}">
          ${isOffline ? '⚠️' : (idx + 1)}
        </div>
      </div>
    `;

    const customIcon = L.divIcon({
      html: iconHtml,
      className: 'custom-station-icon',
      iconSize: [24, 24],
      iconAnchor: [12, 12],
    });

    const marker = L.marker([st.lat, st.lon], { icon: customIcon }).addTo(targetLayer);

    // Click marker to toggle failure in non-user mode
    marker.on('click', () => {
      if (!STATE.isUserClickMode && mode !== 'user') {
        toggleStationFailure(idx);
      }
    });

    const statusBadge = isOffline
      ? `<span class="px-1.5 py-0.5 rounded text-[10px] bg-rose-500/20 text-rose-300 font-bold border border-rose-500/30">MẤT KẾT NỐI (OFFLINE)</span>`
      : `<span class="px-1.5 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-300 font-bold border border-emerald-500/30">HOẠT ĐỘNG (ONLINE)</span>`;

    marker.bindPopup(`
      <div class="text-xs space-y-1.5 min-w-[200px]">
        <div class="font-bold text-white flex items-center justify-between">
          <span class="flex items-center gap-1.5">
            <span class="w-2.5 h-2.5 rounded-full" style="background:${pinColor}"></span>
            ${baseConf.label} #${idx + 1}
          </span>
          ${statusBadge}
        </div>
        <div class="text-gray-300">Tọa độ WGS84: <span class="font-mono text-cyan-300">${st.lat.toFixed(4)}°N, ${st.lon.toFixed(4)}°E</span></div>
        <div class="text-gray-300">Ô Lưới: <span class="font-mono text-emerald-400">[${st.grid_x}, ${st.grid_y}]</span></div>
        <div class="text-gray-300">Bán kính R<sub class="text-[9px]">sens</sub>: <span class="font-mono text-amber-400">${STATE.rCover} ô (~${(radiusMeters/1000).toFixed(1)} km)</span></div>
        ${mode === 'user' ? `
          <button onclick="removeUserStationAtIndex(${idx})" class="mt-1 w-full py-1 px-2 rounded bg-rose-900/60 hover:bg-rose-800 text-rose-200 text-[10px] font-semibold">Xóa trạm này</button>
        ` : `
          <div class="pt-1 border-t border-gray-700">
            <button onclick="toggleStationFailure(${idx})" class="w-full py-1 px-2 rounded ${isOffline ? 'bg-emerald-800 hover:bg-emerald-700 text-emerald-100' : 'bg-rose-900/80 hover:bg-rose-800 text-rose-100'} text-[10px] font-semibold transition-colors flex items-center justify-center gap-1">
              <span>${isOffline ? '🔄 Khôi phục trạm (Online)' : '💥 Ngắt trạm này (Giả lập sự cố)'}</span>
            </button>
          </div>
        `}
      </div>
    `);
  });
}

function removeUserStationAtIndex(index) {
  if (index >= 0 && index < STATE.userStations.length) {
    STATE.userStations.splice(index, 1);
    evaluateUserStations();
  }
}

function renderViolatingPairsLines(violatingPairs, stations) {
  layers.violatingLines.clearLayers();
  if (!violatingPairs || !violatingPairs.length || !stations) return;

  violatingPairs.forEach((pair) => {
    const stA = stations[pair.station_i];
    const stB = stations[pair.station_j];
    if (stA && stB) {
      const line = L.polyline([[stA.lat, stA.lon], [stB.lat, stB.lon]], {
        color: '#f43f5e',
        weight: 3,
        dashArray: '6, 6',
      }).addTo(layers.violatingLines);

      line.bindTooltip(`⚠️ Vi phạm khoảng cách tối thiểu: ${pair.distance_km} km < ${ (cellsToMeters(STATE.dMin)/1000).toFixed(1) } km d_min!`, {
        sticky: true,
        className: 'bg-rose-950 text-rose-200 text-xs px-2 py-1 rounded border border-rose-500/40',
      });
    }
  });
}

async function loadOfficialBaseline() {
  try {
    const resp = await fetch('/api/existing_stations');
    const data = await resp.json();
    STATE.results.official = data;

    // Render baseline row in table
    document.getElementById('offCov').textContent = `${data.coverage_percent.toFixed(2)}%`;
    document.getElementById('offFit').textContent = data.penalized_fitness.toFixed(4);
    document.getElementById('offOverlap').textContent = `${data.overlap_percent.toFixed(1)}%`;
    document.getElementById('offMinDist').textContent = `${data.min_pairwise_distance_cells} ô (${data.min_pairwise_distance_km} km)`;

    // Render official station markers
    renderOfficialStationsLayer(data.stations);
  } catch (err) {
    console.error('Failed to load official baseline:', err);
  }
}

function renderOfficialStationsLayer(stations) {
  layers.officialStations.clearLayers();
  if (!stations) return;

  stations.forEach((st, i) => {
    const iconHtml = `
      <div class="w-5 h-5 rounded-full bg-amber-500 border border-white flex items-center justify-center text-[9px] font-bold text-gray-950 shadow-md">
        ${i + 1}
      </div>
    `;
    const icon = L.divIcon({ html: iconHtml, className: 'official-icon', iconSize: [20, 20] });
    const marker = L.marker([st.lat, st.lon], { icon }).addTo(layers.officialStations);

    marker.bindPopup(`
      <div class="text-xs space-y-1">
        <div class="font-bold text-amber-400 flex items-center gap-1.5">
          <span class="w-2 h-2 rounded-full bg-amber-400"></span>
          Trạm Hiện Hữu TP.HCM #${i + 1}
        </div>
        <div class="text-gray-200 font-semibold">${st.original_click?.name || 'Trạm quan trắc đô thị'}</div>
        <div class="text-gray-300">Tọa độ: ${st.lat.toFixed(4)}°N, ${st.lon.toFixed(4)}°E</div>
        <div class="text-gray-400 text-[10px]">Cụm trạm trung tâm bị chồng lấn cao do vị trí lịch sử.</div>
      </div>
    `);
  });

  // Layer toggle listener
  if (!document.getElementById('chkOfficialStations').checked) {
    map.removeLayer(layers.officialStations);
  }
}

// ========================================================================= //
// 9. CHART.JS INITIALIZATION & UPDATES                                      //
// ========================================================================= //
function initCharts() {
  // Chart 1: Convergence Line Chart
  const ctxConv = document.getElementById('chartConvergence').getContext('2d');
  STATE.charts.convergence = new Chart(ctxConv, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Tỷ Lệ Bao Phủ (% Coverage)',
        data: [],
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        fill: true,
        tension: 0.25,
        borderWidth: 2,
        pointRadius: 2,
        pointBackgroundColor: '#10b981',
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: 'index',
          intersect: false,
          callbacks: {
            label: (ctx) => `Độ bao phủ: ${ctx.parsed.y.toFixed(2)}%`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#9ca3af', font: { family: 'monospace', size: 10 }, maxTicksLimit: 12 },
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#9ca3af', font: { family: 'monospace', size: 10 } },
          title: { display: true, text: '% Bao phủ', color: '#6b7280', font: { size: 10 } }
        }
      }
    }
  });

  // Chart 2: Method Comparison Bar Chart
  const ctxComp = document.getElementById('chartComparison').getContext('2d');
  STATE.charts.comparison = new Chart(ctxComp, {
    type: 'bar',
    data: {
      labels: ['1. Tối Ưu HARO', '2. Random', '3. User Click', '4. Trạm Hiện Hữu'],
      datasets: [
        {
          label: 'Tỷ Lệ Bao Phủ (%)',
          data: [0, 0, 0, 47.79],
          backgroundColor: '#10b981',
          borderRadius: 4,
        },
        {
          label: 'Tỷ Lệ Chồng Lấn (%)',
          data: [0, 0, 0, 28.40],
          backgroundColor: '#f59e0b',
          borderRadius: 4,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: { color: '#d1d5db', font: { size: 10 }, boxWidth: 12 }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#9ca3af', font: { size: 10 } },
        },
        y: {
          beginAtZero: true,
          max: 100,
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#9ca3af', font: { family: 'monospace', size: 10 } },
        }
      }
    }
  });

  // Chart 3: Tournament Multi-Trace Line Chart
  const elTour = document.getElementById('chartTournament');
  if (elTour) {
    const ctxTour = elTour.getContext('2d');
    STATE.charts.tournament = new Chart(ctxTour, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'HARO (Proposed)', borderColor: '#10b981', backgroundColor: 'transparent', data: [], tension: 0.25, borderWidth: 2.4, pointRadius: 1 },
          { label: 'ARO (Wang 2022)', borderColor: '#00e5ff', backgroundColor: 'transparent', data: [], tension: 0.25, borderWidth: 1.8, pointRadius: 1 },
          { label: 'PSO (Kennedy 1995)', borderColor: '#ff007f', backgroundColor: 'transparent', data: [], tension: 0.25, borderWidth: 1.8, pointRadius: 1 },
          { label: 'MVO (Mirjalili 2016)', borderColor: '#ffd600', backgroundColor: 'transparent', data: [], tension: 0.25, borderWidth: 1.8, pointRadius: 1 },
          { label: 'GA (Genetic Algo)', borderColor: '#b388ff', backgroundColor: 'transparent', data: [], tension: 0.25, borderWidth: 1.8, pointRadius: 1 },
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: { color: '#d1d5db', font: { size: 9 }, boxWidth: 10, padding: 6 }
          },
          tooltip: {
            mode: 'index',
            intersect: false,
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}%`
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#9ca3af', font: { family: 'monospace', size: 9 }, maxTicksLimit: 10 },
          },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#9ca3af', font: { family: 'monospace', size: 9 } },
            title: { display: true, text: '% Độ Bao Phủ', color: '#6b7280', font: { size: 9 } }
          }
        }
      }
    });
  }
}

function updateComparisonBarChart() {
  const covData = [
    STATE.results.haro ? STATE.results.haro.coverage_percent : 0,
    STATE.results.random ? STATE.results.random.coverage_percent : 0,
    STATE.results.user ? STATE.results.user.coverage_percent : 0,
    STATE.results.official ? STATE.results.official.coverage_percent : 47.79,
  ];

  const overlapData = [
    STATE.results.haro ? STATE.results.haro.overlap_percent : 0,
    STATE.results.random ? STATE.results.random.overlap_percent : 0,
    STATE.results.user ? STATE.results.user.overlap_percent : 0,
    STATE.results.official ? STATE.results.official.overlap_percent : 28.40,
  ];

  STATE.charts.comparison.data.datasets[0].data = covData;
  STATE.charts.comparison.data.datasets[1].data = overlapData;
  STATE.charts.comparison.update();
}

// ========================================================================= //
// 10. THESIS CSV DATABASE VIEWER MODAL                                      //
// ========================================================================= //
async function openThesisDataModal(datasetKey = 'tong_hop') {
  const modal = document.getElementById('modalThesisData');
  modal.classList.remove('hidden');

  // Activate tab button
  document.querySelectorAll('.dataset-tab-btn').forEach(btn => {
    if (btn.dataset.dataset === datasetKey) {
      btn.className = 'dataset-tab-btn active px-3 py-1.5 rounded-lg text-xs font-medium bg-indigo-600 text-white transition-colors';
    } else {
      btn.className = 'dataset-tab-btn px-3 py-1.5 rounded-lg text-xs font-medium bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white transition-colors';
    }
  });

  try {
    const resp = await fetch(`/api/thesis_data?dataset=${datasetKey}`);
    const data = await resp.json();

    STATE.thesisCsvData.currentKey = datasetKey;
    STATE.thesisCsvData.headers = data.headers;
    STATE.thesisCsvData.rows = data.rows;

    document.getElementById('modalDatasetBadge').textContent = data.metadata.badge;
    document.getElementById('modalDatasetSubtitle').textContent = data.metadata.subtitle;
    document.getElementById('modalKeyInsight').textContent = data.metadata.key_insight;
    document.getElementById('modalFileName').textContent = data.file_name;
    document.getElementById('modalRowCount').textContent = `${data.row_count} dòng dữ liệu`;

    renderThesisTable(data.headers, data.rows);
  } catch (err) {
    console.error('Failed to load thesis CSV:', err);
  }
}

function renderThesisTable(headers, rows) {
  const thead = document.getElementById('tableThesisThead');
  const tbody = document.getElementById('tableThesisTbody');

  thead.innerHTML = '';
  tbody.innerHTML = '';

  if (!headers || !headers.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="py-8 text-center text-gray-400">Không có dữ liệu bảng.</td></tr>';
    return;
  }

  // Header row
  const trHead = document.createElement('tr');
  headers.forEach(h => {
    const th = document.createElement('th');
    th.className = 'py-2.5 px-3 whitespace-nowrap font-semibold';
    th.textContent = h;
    trHead.appendChild(th);
  });
  thead.appendChild(trHead);

  // Body rows
  rows.forEach((r, idx) => {
    const tr = document.createElement('tr');
    tr.className = idx % 2 === 0 ? 'bg-[#0e1628]' : 'bg-[#111b32]';
    tr.classList.add('hover:bg-indigo-950/40', 'transition-colors');

    r.forEach((cell, cellIdx) => {
      const td = document.createElement('td');
      td.className = 'py-2 px-3 whitespace-nowrap text-gray-300';
      
      // Highlight HARO algorithm name
      if (cell === 'HARO') {
        td.innerHTML = `<span class="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-bold">HARO (Proposed)</span>`;
      } else if (cell === 'Win') {
        td.innerHTML = `<span class="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-bold">Win</span>`;
      } else if (cell === 'Loss') {
        td.innerHTML = `<span class="px-2 py-0.5 rounded bg-rose-500/20 text-rose-400 font-bold">Loss</span>`;
      } else {
        td.textContent = cell;
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

function filterThesisTable(keyword) {
  const q = keyword.toLowerCase().trim();
  const filtered = STATE.thesisCsvData.rows.filter(row => {
    return row.some(cell => String(cell).toLowerCase().includes(q));
  });
  renderThesisTable(STATE.thesisCsvData.headers, filtered);
  document.getElementById('modalRowCount').textContent = `${filtered.length}/${STATE.thesisCsvData.rows.length} dòng dữ liệu`;
}

function exportCurrentTableToCsv() {
  const headers = STATE.thesisCsvData.headers;
  const rows = STATE.thesisCsvData.rows;
  if (!headers.length) return;

  let csvContent = "data:text/csv;charset=utf-8," + headers.join(",") + "\n";
  rows.forEach(r => {
    csvContent += r.map(c => `"${c}"`).join(",") + "\n";
  });

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `${STATE.thesisCsvData.currentKey}_results.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// ========================================================================= //
// 11. ACADEMIC REPORT FIGURE EXPORT MODAL                                   //
// ========================================================================= //
async function openReportFigureModal() {
  const modal = document.getElementById('modalReportFigure');
  const spinner = document.getElementById('reportLoadingSpinner');
  const imgPreview = document.getElementById('imgReportPreview');
  const btnDownload = document.getElementById('btnDownloadReportPng');

  modal.classList.remove('hidden');
  spinner.classList.remove('hidden');
  imgPreview.classList.add('hidden');

  // Lấy các trạm và metrics theo mode đang active hoặc fallback sang HARO/Random
  let stations = [];
  let metrics = {};

  if (STATE.activeMode === 'haro' && STATE.results.haro) {
    stations = STATE.results.haro.stations;
    metrics = {
      algorithm: 'HARO (Halo-enhanced Artificial Rabbits Optimization)',
      coverage_percent: STATE.results.haro.coverage_percent,
      fitness: STATE.results.haro.fitness || STATE.results.haro.penalized_fitness,
      overlap_percent: STATE.results.haro.overlap_percent,
      min_pairwise_distance_km: STATE.results.haro.min_pairwise_distance_km,
      min_pairwise_distance_cells: STATE.results.haro.min_pairwise_distance_cells,
      feasible: STATE.results.haro.feasible,
      fe: STATE.results.haro.actual_fe || 1530,
      runtime: STATE.results.haro.runtime_seconds || 5.59,
    };
  } else if (STATE.activeMode === 'random' && STATE.results.random) {
    stations = STATE.results.random.stations;
    metrics = {
      algorithm: 'Random Uniform Baseline (Sinh Ngẫu Nhiên)',
      coverage_percent: STATE.results.random.coverage_percent,
      fitness: STATE.results.random.penalized_fitness,
      overlap_percent: STATE.results.random.overlap_percent,
      min_pairwise_distance_km: STATE.results.random.min_pairwise_distance_km,
      min_pairwise_distance_cells: STATE.results.random.min_pairwise_distance_cells,
      feasible: STATE.results.random.feasible,
      fe: STATE.k,
      runtime: STATE.results.random.runtime_seconds,
    };
  } else if (STATE.activeMode === 'user' && STATE.results.user) {
    stations = STATE.results.user.stations;
    metrics = {
      algorithm: 'User Click Placement (Người Dùng Tự Chọn Thủ Công)',
      coverage_percent: STATE.results.user.coverage_percent,
      fitness: STATE.results.user.penalized_fitness,
      overlap_percent: STATE.results.user.overlap_percent,
      min_pairwise_distance_km: STATE.results.user.min_pairwise_distance_km,
      min_pairwise_distance_cells: STATE.results.user.min_pairwise_distance_cells,
      feasible: STATE.results.user.feasible,
      fe: STATE.results.user.placed_count,
      runtime: STATE.results.user.runtime_seconds,
    };
  } else {
    // Default metrics nếu chưa chạy lượt nào
    metrics = {
      algorithm: 'HARO (Halo-enhanced Artificial Rabbits Optimization)',
      coverage_percent: 28.35,
      fitness: -0.28352,
      overlap_percent: 18.0,
      min_pairwise_distance_km: 5.44,
      min_pairwise_distance_cells: 10.3,
      feasible: true,
      fe: 1530,
      runtime: 5.59,
    };
  }

  const convTrace = STATE.charts.convergence ? STATE.charts.convergence.data.datasets[0].data : [];

  try {
    const resp = await fetch('/api/export_report_figure', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        scenario: STATE.scenario,
        grid_m: STATE.gridM,
        grid_n: STATE.gridN,
        k: STATE.k,
        coverage_radius: STATE.rCover,
        min_station_distance: STATE.dMin,
        stations: stations,
        metrics: metrics,
        convergence: convTrace,
      })
    });

    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }

    const blob = await resp.blob();
    const imgUrl = URL.createObjectURL(blob);

    imgPreview.src = imgUrl;
    btnDownload.href = imgUrl;

    spinner.classList.add('hidden');
    imgPreview.classList.remove('hidden');
  } catch (err) {
    console.error('Failed to generate report figure:', err);
    spinner.innerHTML = `<span class="text-xs text-rose-400">Lỗi khi xuất ảnh báo cáo: ${err.message}</span>`;
  }
}

// ========================================================================= //
// 12. EVENT LISTENERS & UI WIRING                                           //
// ========================================================================= //
function initEventListeners() {
  // Scenario select
  document.getElementById('selectScenario').addEventListener('change', async (e) => {
    STATE.scenario = e.target.value;
    const desc = document.getElementById('scenarioDesc');
    if (STATE.scenario === 'standard_50x50') {
      desc.textContent = 'Lưới không gian 50x50 với 12 nguồn phát thải công nghiệp/giao thông và 2 vùng cấm trắc địa chuẩn hóa.';
    } else {
      desc.textContent = 'Địa bàn thực tế TP.HCM: 12 Điểm nóng ô nhiễm trọng điểm, vùng cấm hành lang Sông Sài Gòn & sân bay Tân Sơn Nhất.';
    }
    await loadEnvironmentData();
  });

  // Sliders
  document.getElementById('sliderK').addEventListener('input', (e) => {
    STATE.k = parseInt(e.target.value);
    document.getElementById('labelKVal').textContent = `${STATE.k} trạm`;
  });

  document.getElementById('sliderR').addEventListener('input', (e) => {
    STATE.rCover = parseFloat(e.target.value);
    const m = cellsToMeters(STATE.rCover);
    document.getElementById('labelRVal').textContent = `${STATE.rCover.toFixed(1)} ô (~${(m/1000).toFixed(1)} km)`;
    // Re-render station circles
    if (STATE.activeMode === 'haro' && STATE.results.haro) {
      renderStationsOnMap(STATE.results.haro.stations, 'haro');
    } else if (STATE.activeMode === 'random' && STATE.results.random) {
      renderStationsOnMap(STATE.results.random.stations, 'random');
    } else if (STATE.activeMode === 'user') {
      evaluateUserStations();
    }
  });

  document.getElementById('sliderDMin').addEventListener('input', (e) => {
    STATE.dMin = parseFloat(e.target.value);
    const m = cellsToMeters(STATE.dMin);
    document.getElementById('labelDMinVal').textContent = `${STATE.dMin.toFixed(1)} ô (~${(m/1000).toFixed(1)} km)`;
    if (STATE.activeMode === 'user') {
      evaluateUserStations();
    }
  });

  document.getElementById('sliderMaxIter').addEventListener('input', (e) => {
    STATE.maxIterations = parseInt(e.target.value);
    document.getElementById('labelMaxIterVal').textContent = `${STATE.maxIterations} vòng`;
  });

  document.getElementById('sliderHeatOpacity').addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    if (layers.pollutionHeat) {
      layers.pollutionHeat.setOptions({ opacity: val });
    }
  });

  // Action Buttons
  document.getElementById('btnRunHaro').addEventListener('click', startHaroOptimization);
  document.getElementById('btnStopHaro').addEventListener('click', stopHaroOptimization);
  document.getElementById('btnRunRandom').addEventListener('click', runRandomPlacement);
  document.getElementById('btnToggleUserClick').addEventListener('click', toggleUserClickMode);
  document.getElementById('btnUndoUserStation').addEventListener('click', undoLastUserStation);
  document.getElementById('btnClearUserStations').addEventListener('click', clearAllUserStations);

  // Combo 1 & 2 Action Buttons
  const btnSingle = document.getElementById('btnRunSingleAlgo');
  if (btnSingle) btnSingle.addEventListener('click', runSingleAlgorithm);

  const btnTour = document.getElementById('btnRunTournament');
  if (btnTour) btnTour.addEventListener('click', runAlgorithmTournament);

  const btnRes1 = document.getElementById('btnResilienceDrop1');
  if (btnRes1) btnRes1.addEventListener('click', () => runResilienceSimulation('random_1'));

  const btnRes2 = document.getElementById('btnResilienceDrop2');
  if (btnRes2) btnRes2.addEventListener('click', () => runResilienceSimulation('random_2'));

  const btnResWorst = document.getElementById('btnResilienceWorst');
  if (btnResWorst) btnResWorst.addEventListener('click', () => runResilienceSimulation('worst_case'));

  const btnResReset = document.getElementById('btnResilienceReset');
  if (btnResReset) btnResReset.addEventListener('click', () => {
    STATE.disabledStationIds.clear();
    runResilienceSimulation('custom', []);
  });

  const btnRecon = document.getElementById('btnRunReconstruction');
  if (btnRecon) btnRecon.addEventListener('click', runSpatialReconstruction);

  // Layer Checkboxes
  document.getElementById('chkHeatmap').addEventListener('change', (e) => {
    if (e.target.checked && layers.pollutionHeat) {
      layers.pollutionHeat.addTo(map);
    } else if (layers.pollutionHeat) {
      map.removeLayer(layers.pollutionHeat);
    }
  });

  const chkRecon = document.getElementById('chkReconHeatmap');
  if (chkRecon) {
    chkRecon.addEventListener('change', (e) => {
      if (layers.reconHeat) {
        e.target.checked ? layers.reconHeat.addTo(map) : map.removeLayer(layers.reconHeat);
      }
    });
  }

  const chkHalo = document.getElementById('chkHaloRings');
  if (chkHalo) {
    chkHalo.addEventListener('change', (e) => {
      if (layers.haloRings) {
        e.target.checked ? layers.haloRings.addTo(map) : map.removeLayer(layers.haloRings);
      }
    });
  }

  document.getElementById('chkBoundingBox').addEventListener('change', (e) => {
    e.target.checked ? layers.boundingBox.addTo(map) : map.removeLayer(layers.boundingBox);
  });

  document.getElementById('chkForbiddenZones').addEventListener('change', (e) => {
    e.target.checked ? layers.forbiddenZones.addTo(map) : map.removeLayer(layers.forbiddenZones);
  });

  document.getElementById('chkSources').addEventListener('change', (e) => {
    e.target.checked ? layers.sources.addTo(map) : map.removeLayer(layers.sources);
  });

  document.getElementById('chkStationCircles').addEventListener('change', () => {
    if (STATE.activeMode === 'haro' && STATE.results.haro) {
      renderStationsOnMap(STATE.results.haro.stations, 'haro');
    } else if (STATE.activeMode === 'random' && STATE.results.random) {
      renderStationsOnMap(STATE.results.random.stations, 'random');
    } else if (STATE.activeMode === 'user' && STATE.results.user) {
      renderStationsOnMap(STATE.results.user.stations, 'user');
    }
  });

  document.getElementById('chkOfficialStations').addEventListener('change', (e) => {
    e.target.checked ? layers.officialStations.addTo(map) : map.removeLayer(layers.officialStations);
  });

  // Tile Switchers
  document.getElementById('btnTileDark').addEventListener('click', () => setTileLayer('dark'));
  document.getElementById('btnTileLight').addEventListener('click', () => setTileLayer('light'));
  document.getElementById('btnTileOSM').addEventListener('click', () => setTileLayer('osm'));

  // Dashboard Tabs
  document.getElementById('tabBtnComparison').addEventListener('click', () => activateTab('comparison'));
  document.getElementById('tabBtnCharts').addEventListener('click', () => activateTab('charts'));
  const tabTour = document.getElementById('tabBtnTournament');
  if (tabTour) tabTour.addEventListener('click', () => activateTab('tournament'));

  // Thesis Modal Events
  document.getElementById('btnOpenThesisData').addEventListener('click', () => openThesisDataModal('tong_hop'));
  document.getElementById('btnCloseThesisData').addEventListener('click', () => {
    document.getElementById('modalThesisData').classList.add('hidden');
  });

  document.querySelectorAll('.dataset-tab-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      openThesisDataModal(e.target.dataset.dataset);
    });
  });

  document.getElementById('inputSearchTable').addEventListener('input', (e) => {
    filterThesisTable(e.target.value);
  });

  document.getElementById('btnExportCsv').addEventListener('click', exportCurrentTableToCsv);

  // Report Figure Modal Events
  document.getElementById('btnExportFigure').addEventListener('click', openReportFigureModal);
  document.getElementById('btnSideExportFigure').addEventListener('click', openReportFigureModal);
  document.getElementById('btnCloseReportFigure').addEventListener('click', () => {
    document.getElementById('modalReportFigure').classList.add('hidden');
  });
  document.getElementById('btnFooterCloseReport').addEventListener('click', () => {
    document.getElementById('modalReportFigure').classList.add('hidden');
  });

  // Fullscreen button
  document.getElementById('btnToggleFullscreen').addEventListener('click', () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(err => console.log(err));
    } else {
      document.exitFullscreen().catch(err => console.log(err));
    }
  });
}

function updateSliderDisplays() {
  document.getElementById('sliderK').value = STATE.k;
  document.getElementById('labelKVal').textContent = `${STATE.k} trạm`;

  document.getElementById('sliderR').value = STATE.rCover;
  const mR = cellsToMeters(STATE.rCover);
  document.getElementById('labelRVal').textContent = `${STATE.rCover.toFixed(1)} ô (~${(mR/1000).toFixed(1)} km)`;

  document.getElementById('sliderDMin').value = STATE.dMin;
  const mD = cellsToMeters(STATE.dMin);
  document.getElementById('labelDMinVal').textContent = `${STATE.dMin.toFixed(1)} ô (~${(mD/1000).toFixed(1)} km)`;
}

function updateStatus(text, type = 'ready') {
  const statusText = document.getElementById('statusText');
  statusText.textContent = text;
}

function updateModeBadge(text, mode) {
  const badge = document.getElementById('activeModeBadge');
  const txt = document.getElementById('activeModeText');
  txt.textContent = text;

  if (mode === 'haro') {
    badge.className = 'glass-panel px-3.5 py-2 rounded-lg shadow-xl text-xs flex items-center space-x-2 font-medium text-emerald-400 border-emerald-500/30';
  } else if (mode === 'random') {
    badge.className = 'glass-panel px-3.5 py-2 rounded-lg shadow-xl text-xs flex items-center space-x-2 font-medium text-blue-400 border-blue-500/30';
  } else if (mode === 'user') {
    badge.className = 'glass-panel px-3.5 py-2 rounded-lg shadow-xl text-xs flex items-center space-x-2 font-medium text-purple-400 border-purple-500/30';
  }
}

// ========================================================================= //
// 13. COMBO 1: TOURNAMENT RACE, SINGLE ALGO & HALO HUD                      //
// ========================================================================= //
function updateHaloHUD(haloLayer, energy) {
  const layerEl = document.getElementById('hudHaloLayer');
  const valEl = document.getElementById('hudEnergyVal');
  const barEl = document.getElementById('hudEnergyBar');
  const phaseEl = document.getElementById('hudSearchPhase');

  if (!layerEl) return;

  const lUpper = (haloLayer || 'outer').toUpperCase();
  layerEl.textContent = `${lUpper} LAYER`;

  if (haloLayer === 'outer') {
    layerEl.className = 'font-mono text-[10px] font-bold px-2 py-0.5 rounded bg-purple-500/20 text-purple-300 uppercase border border-purple-500/30';
  } else if (haloLayer === 'middle') {
    layerEl.className = 'font-mono text-[10px] font-bold px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 uppercase border border-cyan-500/30';
  } else {
    layerEl.className = 'font-mono text-[10px] font-bold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 uppercase border border-emerald-500/30';
  }

  valEl.textContent = `A = ${Number(energy).toFixed(2)}`;
  const pct = Math.max(5, Math.min(100, (Number(energy) / 2.0) * 100));
  barEl.style.width = `${pct}%`;

  if (Number(energy) > 1.0) {
    phaseEl.textContent = 'Giai đoạn: Thám hiểm diện rộng (Detour Foraging)';
    phaseEl.className = 'text-[10px] text-amber-400 font-medium truncate';
  } else {
    phaseEl.textContent = 'Giai đoạn: Khai phá cục bộ (Random Hiding)';
    phaseEl.className = 'text-[10px] text-emerald-400 font-medium truncate';
  }
}

function renderHaloRings(stations, haloLayer) {
  if (!layers.haloRings) return;
  layers.haloRings.clearLayers();
  const chk = document.getElementById('chkHaloRings');
  if (!chk || !chk.checked || !stations || !stations.length) return;

  let avgLat = 0, avgLon = 0;
  stations.forEach(s => { avgLat += s.lat; avgLon += s.lon; });
  avgLat /= stations.length;
  avgLon /= stations.length;

  const rInnerMeters = cellsToMeters(4.0);
  const rMidMeters = cellsToMeters(8.0);
  const rOuterMeters = cellsToMeters(14.0);

  L.circle([avgLat, avgLon], {
    radius: rOuterMeters,
    color: haloLayer === 'outer' ? '#a855f7' : '#475569',
    weight: haloLayer === 'outer' ? 2 : 1,
    dashArray: '6, 6',
    fillColor: '#a855f7',
    fillOpacity: haloLayer === 'outer' ? 0.08 : 0.02,
  }).addTo(layers.haloRings);

  L.circle([avgLat, avgLon], {
    radius: rMidMeters,
    color: haloLayer === 'middle' ? '#06b6d4' : '#334155',
    weight: haloLayer === 'middle' ? 2 : 1,
    dashArray: '4, 4',
    fillColor: '#06b6d4',
    fillOpacity: haloLayer === 'middle' ? 0.10 : 0.02,
  }).addTo(layers.haloRings);

  L.circle([avgLat, avgLon], {
    radius: rInnerMeters,
    color: haloLayer === 'inner' ? '#10b981' : '#1e293b',
    weight: haloLayer === 'inner' ? 2.5 : 1,
    fillColor: '#10b981',
    fillOpacity: haloLayer === 'inner' ? 0.15 : 0.03,
  }).addTo(layers.haloRings);
}

async function runAlgorithmTournament() {
  try {
    STATE.activeMode = 'tournament';
    disableUserClickMode();
    updateStatus('Đấu trường 5 giải thuật đang chạy đua (HARO vs ARO vs PSO vs MVO vs GA)...', 'loading');
    updateModeBadge('Đấu Trường: Đang chạy đua 5 giải thuật...', 'haro');

    activateTab('tournament');

    const resp = await fetch('/api/run_tournament', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        scenario: STATE.scenario,
        grid_m: STATE.gridM,
        grid_n: STATE.gridN,
        k: STATE.k,
        coverage_radius: STATE.rCover,
        min_station_distance: STATE.dMin,
        max_iterations: STATE.maxIterations,
        pop_size: 30,
        seed: 42,
      })
    });
    const res = await resp.json();
    STATE.tournamentResults = res;

    renderTournamentLeaderboard(res.leaderboard);
    updateTournamentChart(res.traces, res.iteration_labels);

    const winner = res.leaderboard[0];
    STATE.currentStations = winner.stations;
    renderStationsOnMap(winner.stations, 'haro');

    updateStatus(`Đấu trường hoàn tất! Quán quân: ${winner.code} với độ bao phủ ${winner.coverage_percent.toFixed(2)}%`, 'ready');
    updateModeBadge(`Đấu Trường: ${winner.code} Vô Địch (${winner.coverage_percent.toFixed(2)}%)`, 'haro');
  } catch (err) {
    console.error('Lỗi chạy đấu trường:', err);
    updateStatus('Lỗi khi chạy đấu trường giải thuật', 'error');
  }
}

function renderTournamentLeaderboard(leaderboard) {
  const tbody = document.getElementById('tournamentTbody');
  if (!tbody || !leaderboard) return;
  tbody.innerHTML = '';

  leaderboard.forEach((item) => {
    const tr = document.createElement('tr');
    tr.className = 'hover:bg-gray-800/50 transition-colors';

    let rankBadge = `<span class="px-1.5 py-0.5 rounded text-[10px] font-bold bg-gray-800 text-gray-400">#${item.rank}</span>`;
    if (item.rank === 1) {
      rankBadge = `<span class="px-1.5 py-0.5 rounded text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40">🥇 #1</span>`;
    } else if (item.rank === 2) {
      rankBadge = `<span class="px-1.5 py-0.5 rounded text-[10px] font-bold bg-gray-400/20 text-gray-200">🥈 #2</span>`;
    } else if (item.rank === 3) {
      rankBadge = `<span class="px-1.5 py-0.5 rounded text-[10px] font-bold bg-amber-700/20 text-amber-500">🥉 #3</span>`;
    }

    const algoName = item.code === 'HARO'
      ? `<span class="font-bold text-emerald-400">${item.code} (Proposed)</span>`
      : `<span class="font-medium" style="color:${item.color}">${item.code}</span>`;

    tr.innerHTML = `
      <td class="py-2 px-2.5 font-sans">${rankBadge}</td>
      <td class="py-2 px-2.5 font-sans">${algoName}</td>
      <td class="py-2 px-2.5 font-bold text-emerald-400">${item.coverage_percent.toFixed(2)}%</td>
      <td class="py-2 px-2.5 text-gray-300">${item.fitness.toFixed(4)}</td>
      <td class="py-2 px-2.5 text-amber-400">${item.overlap_percent.toFixed(1)}%</td>
      <td class="py-2 px-2.5 text-gray-400">${item.min_pairwise_distance_km.toFixed(2)} km</td>
      <td class="py-2 px-2.5 text-right text-gray-400">${item.actual_fe.toLocaleString()} FE / ${item.runtime_seconds}s</td>
      <td class="py-2 px-2 text-center">
        <button onclick="displayTournamentStations('${item.code}')" class="px-2 py-0.5 rounded bg-gray-800 hover:bg-gray-700 text-[10px] text-cyan-300 border border-gray-700 font-sans transition-colors" title="Xem vị trí trạm trên bản đồ">
          Xem
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function displayTournamentStations(algoCode) {
  if (!STATE.tournamentResults) return;
  const item = STATE.tournamentResults.leaderboard.find(x => x.code === algoCode);
  if (!item) return;

  STATE.currentStations = item.stations;
  STATE.disabledStationIds.clear();

  document.getElementById('resilienceRetention').textContent = '100.0%';
  document.getElementById('resilienceRetention').style.color = '#10b981';
  document.getElementById('resilienceDrop').textContent = '0.00%';
  document.getElementById('resilienceStatusBadge').textContent = `Hiển thị mạng lưới trạm của ${item.code}`;

  renderStationsOnMap(item.stations, 'haro');
  updateModeBadge(`${item.code}: Độ bao phủ ${item.coverage_percent.toFixed(2)}% | Chồng lấn ${item.overlap_percent.toFixed(1)}%`, 'haro');
  updateStatus(`Đang hiển thị vị trí K=${STATE.k} trạm của giải thuật ${item.code} trên bản đồ.`, 'ready');
}

function updateTournamentChart(traces, labels) {
  if (!STATE.charts.tournament) return;
  STATE.charts.tournament.data.labels = labels;

  const datasetMap = {
    'HARO': 0,
    'ARO': 1,
    'PSO': 2,
    'MVO': 3,
    'GA': 4,
  };

  Object.entries(traces).forEach(([code, traceInfo]) => {
    const dsIdx = datasetMap[code];
    if (dsIdx !== undefined) {
      STATE.charts.tournament.data.datasets[dsIdx].data = traceInfo.data;
    }
  });

  STATE.charts.tournament.update();
}

async function runSingleAlgorithm() {
  const sel = document.getElementById('selectSingleAlgo');
  const code = sel ? sel.value : 'HARO';
  try {
    updateStatus(`Đang chạy giải thuật ${code}...`, 'loading');
    updateModeBadge(`${code}: Đang tối ưu hóa...`, 'haro');

    const resp = await fetch('/api/run_single_algorithm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        algorithm: code,
        scenario: STATE.scenario,
        grid_m: STATE.gridM,
        grid_n: STATE.gridN,
        k: STATE.k,
        coverage_radius: STATE.rCover,
        min_station_distance: STATE.dMin,
        max_iterations: STATE.maxIterations,
        pop_size: 30,
        seed: 42,
      })
    });
    const res = await resp.json();

    STATE.currentStations = res.stations;
    STATE.disabledStationIds.clear();
    renderStationsOnMap(res.stations, 'haro');

    if (res.convergence) {
      STATE.charts.convergence.data.labels = res.convergence.map((_, i) => `G${i}`);
      STATE.charts.convergence.data.datasets[0].data = res.convergence;
      STATE.charts.convergence.data.datasets[0].borderColor = res.color || '#10b981';
      STATE.charts.convergence.update();
    }

    updateStatus(`Giải thuật ${code} hoàn tất: Độ bao phủ ${res.coverage_percent.toFixed(2)}% trong ${res.runtime_seconds}s`, 'ready');
    updateModeBadge(`${code}: Độ bao phủ ${res.coverage_percent.toFixed(2)}%`, 'haro');
  } catch (err) {
    console.error(`Lỗi khi chạy ${code}:`, err);
    updateStatus(`Lỗi khi chạy giải thuật ${code}`, 'error');
  }
}

// ========================================================================= //
// 14. COMBO 2: RESILIENCE SIMULATOR & SPATIAL RECONSTRUCTION (IDW)          //
// ========================================================================= //
async function toggleStationFailure(idx) {
  if (STATE.disabledStationIds.has(idx)) {
    STATE.disabledStationIds.delete(idx);
  } else {
    STATE.disabledStationIds.add(idx);
  }
  await runResilienceSimulation('custom', Array.from(STATE.disabledStationIds));
}

async function runResilienceSimulation(failureType = 'random_1', disabledIds = null) {
  try {
    updateStatus('Đang chạy kiểm định độ bền vững (Resilience Test)...', 'loading');
    const stations = STATE.currentStations.length ? STATE.currentStations : (STATE.results.haro?.stations || []);
    if (!stations.length) {
      alert('Vui lòng chạy Tối ưu HARO hoặc Sinh ngẫu nhiên trước khi kiểm định độ bền vững.');
      return;
    }

    const payload = {
      scenario: STATE.scenario,
      grid_m: STATE.gridM,
      grid_n: STATE.gridN,
      k: STATE.k,
      coverage_radius: STATE.rCover,
      min_station_distance: STATE.dMin,
      stations: stations,
      failure_type: failureType,
      disabled_station_ids: disabledIds !== null ? disabledIds : (failureType === 'custom' ? Array.from(STATE.disabledStationIds) : null),
    };

    const resp = await fetch('/api/resilience_simulation', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const res = await resp.json();

    if (res.status === 'error') {
      alert(res.message);
      return;
    }

    STATE.disabledStationIds = new Set(res.disabled_indices || []);

    document.getElementById('resilienceRetention').textContent = `${res.retention_ratio_percent.toFixed(1)}%`;
    document.getElementById('resilienceRetention').style.color = res.status_color || '#10b981';
    document.getElementById('resilienceDrop').textContent = `${res.coverage_drop_percent.toFixed(2)}%`;
    
    const statusBadge = document.getElementById('resilienceStatusBadge');
    statusBadge.textContent = `${res.operational_status_vn} (${res.operational_status})`;
    statusBadge.style.color = res.status_color || '#10b981';
    statusBadge.style.borderColor = `${res.status_color}40`;
    statusBadge.style.backgroundColor = `${res.status_color}18`;

    renderStationsOnMap(stations, STATE.activeMode || 'haro');
    updateStatus(`Mô phỏng sự cố: ${res.scenario_name} - Giữ vững ${res.retention_ratio_percent.toFixed(1)}%`, 'ready');
  } catch (err) {
    console.error('Lỗi kiểm định resilience:', err);
    updateStatus('Lỗi khi mô phỏng hỏng hóc mạng trạm', 'error');
  }
}

async function runSpatialReconstruction() {
  try {
    updateStatus('Đang nội suy tái tạo trường ô nhiễm bằng IDW...', 'loading');
    const stations = STATE.currentStations.length ? STATE.currentStations : (STATE.results.haro?.stations || []);
    if (!stations.length) {
      alert('Vui lòng chạy Tối ưu HARO hoặc Sinh ngẫu nhiên trước khi tái tạo không gian.');
      return;
    }

    const payload = {
      scenario: STATE.scenario,
      grid_m: STATE.gridM,
      grid_n: STATE.gridN,
      stations: stations,
      power: 2.0,
    };

    const resp = await fetch('/api/spatial_reconstruction', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const res = await resp.json();

    if (res.status === 'error') {
      alert(res.message);
      return;
    }

    document.getElementById('reconRmse').textContent = `${res.rmse} µg/m³`;
    document.getElementById('reconR2').textContent = `${res.r2}`;
    document.getElementById('reconMae').textContent = `${res.mae} µg/m³`;
    document.getElementById('reconHotspotRmse').textContent = `${res.hotspot_rmse} µg/m³`;

    if (layers.reconHeat) {
      map.removeLayer(layers.reconHeat);
    }

    layers.reconHeat = L.heatLayer(res.heat_points, {
      radius: 25,
      blur: 16,
      maxZoom: 16,
      max: 1.0,
      gradient: {
        0.2: '#00bcd4', // Cyan
        0.4: '#3b82f6', // Blue
        0.6: '#eab308', // Yellow
        0.8: '#f97316', // Orange
        1.0: '#ef4444', // Red
      }
    });

    const chk = document.getElementById('chkReconHeatmap');
    if (chk) chk.checked = true;
    layers.reconHeat.addTo(map);

    updateStatus(`Tái tạo không gian hoàn tất: RMSE = ${res.rmse} µg/m³, R² = ${res.r2}`, 'ready');
  } catch (err) {
    console.error('Lỗi tái tạo IDW:', err);
    updateStatus('Lỗi khi tái tạo trường ô nhiễm', 'error');
  }
}

function activateTab(tabName) {
  const btnComp = document.getElementById('tabBtnComparison');
  const btnCharts = document.getElementById('tabBtnCharts');
  const btnTour = document.getElementById('tabBtnTournament');

  const contentComp = document.getElementById('tabContentComparison');
  const contentCharts = document.getElementById('tabContentCharts');
  const contentTour = document.getElementById('tabContentTournament');

  const inactiveClass = 'text-gray-400 hover:text-gray-200 pb-2.5 pt-1.5 flex items-center space-x-1.5 transition-colors';
  const activeClass = 'text-emerald-400 border-b-2 border-emerald-400 pb-2.5 pt-1.5 flex items-center space-x-1.5 font-bold transition-colors';

  btnComp.className = inactiveClass;
  btnCharts.className = inactiveClass;
  if (btnTour) btnTour.className = inactiveClass;

  contentComp.classList.add('hidden');
  contentCharts.classList.add('hidden');
  if (contentTour) contentTour.classList.add('hidden');

  if (tabName === 'comparison') {
    btnComp.className = activeClass;
    contentComp.classList.remove('hidden');
  } else if (tabName === 'charts') {
    btnCharts.className = activeClass;
    contentCharts.classList.remove('hidden');
    if (STATE.charts.convergence) STATE.charts.convergence.resize();
    if (STATE.charts.comparison) STATE.charts.comparison.resize();
  } else if (tabName === 'tournament') {
    if (btnTour) btnTour.className = 'text-amber-400 border-b-2 border-amber-400 pb-2.5 pt-1.5 flex items-center space-x-1.5 font-bold transition-colors';
    if (contentTour) contentTour.classList.remove('hidden');
    if (STATE.charts.tournament) STATE.charts.tournament.resize();
  }
}

