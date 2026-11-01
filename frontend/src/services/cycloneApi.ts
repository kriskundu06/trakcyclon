
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

// ==========================================
// TYPES: API REQUEST
// ==========================================

export interface CycloneObservation {
  latitude: number;
  longitude: number;
  timestamp: string;
}

export interface PredictionRequest {
  previous: CycloneObservation;
  current: CycloneObservation;
}

// ==========================================
// TYPES: API RESPONSE
// ==========================================

export interface ForecastPoint {
  horizon_hours: number;
  latitude: number;
  longitude: number;
}

export interface PredictionResponse {
  model_name: string;
  input_interval_hours: number;
  movement_speed_kmh: number;
  movement_heading_degrees: number;
  forecasts: ForecastPoint[];
}

// ==========================================
// PREDICTION API
// ==========================================

export async function predictCyclone(
  request: PredictionRequest
): Promise<PredictionResponse> {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => null);

    throw new Error(
      error?.detail || `Prediction failed: HTTP ${response.status}`
    );
  }

  return (await response.json()) as PredictionResponse;
}

// ==========================================
// BACKEND HEALTH CHECK
// ==========================================

export interface BackendHealth {
  status: string;
  models_loaded: number[];
  required_horizons: number[];
}

export async function checkBackendHealth(): Promise<BackendHealth> {
  const response = await fetch(`${API_BASE_URL}/health`);

  if (!response.ok) {
    throw new Error("Unable to connect to CycloneTrack API.");
  }

  return (await response.json()) as BackendHealth;
}

export interface RegionStats {
  code: string;
  region: string;
  observations: number;
  cyclones: number;
}

export interface DatasetStats {
  total_observations: number;
  total_cyclones: number;
  monitored_regions: number;
  max_wind_kmh: number | null;
  dataset_start: string | null;
  dataset_end: string | null;
  regions: RegionStats[];
}

export async function getDatasetStats(): Promise<DatasetStats> {
  const response = await fetch(`${API_BASE_URL}/dataset-stats`);

  if (!response.ok) {
    throw new Error("Unable to fetch historical dataset statistics.");
  }

  return (await response.json()) as DatasetStats;
}
