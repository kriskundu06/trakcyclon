
import { useMemo, useState, useEffect, type FormEvent } from "react";
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Polyline,
  Popup,
  useMap,
} from "react-leaflet";

import "leaflet/dist/leaflet.css";
import "./CycloneMap.css";

import {
  predictCyclone,
  type PredictionResponse,
} from "../services/cycloneApi";


function FitMapToPrediction({
  positions,
}: {
  positions: [number, number][];
}) {
  const map = useMap();

  useEffect(() => {
    if (positions.length > 0) {
      map.fitBounds(positions, {
        padding: [40, 40],
        maxZoom: 7,
      });
    }
  }, [map, positions]);

  return null;
}

type CycloneMapProps = {
  showHistoricalPath?: boolean;
  showPredictedPath?: boolean;
};

function CycloneMap({
  showHistoricalPath = true,
  showPredictedPath = true,
}: CycloneMapProps) {
  // Input observations
  const [previousLat, setPreviousLat] = useState("15.2");
  const [previousLon, setPreviousLon] = useState("85.1");
  const [previousTime, setPreviousTime] = useState("2026-10-31T00:00");

  const [currentLat, setCurrentLat] = useState("15.6");
  const [currentLon, setCurrentLon] = useState("85.5");
  const [currentTime, setCurrentTime] = useState("2026-10-31T06:00");

  // Prediction state
  const [prediction, setPrediction] =
    useState<PredictionResponse | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Generate prediction using FastAPI
  const handlePrediction = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    setLoading(true);
    setError("");
    setPrediction(null);

    try {
      const values = [
        previousLat,
        previousLon,
        currentLat,
        currentLon,
      ].map(Number);

      if (!values.every(Number.isFinite)) {
        throw new Error("Please enter valid coordinates.");
      }

      if (
        values[0] < -90 || values[0] > 90 ||
        values[2] < -90 || values[2] > 90 ||
        values[1] < -180 || values[1] > 180 ||
        values[3] < -180 || values[3] > 180
      ) {
        throw new Error("Please enter valid latitude and longitude values.");
      }

      const previousDate = new Date(previousTime);
      const currentDate = new Date(currentTime);

      if (
        Number.isNaN(previousDate.getTime()) ||
        Number.isNaN(currentDate.getTime()) ||
        currentDate <= previousDate
      ) {
        throw new Error("Current observation time must be later than previous time.");
      }

      const result = await predictCyclone({
        previous: {
          latitude: Number(previousLat),
          longitude: Number(previousLon),
          timestamp: previousDate.toISOString(),
        },
        current: {
          latitude: Number(currentLat),
          longitude: Number(currentLon),
          timestamp: currentDate.toISOString(),
        },
      });

      setPrediction(result);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong while generating the prediction."
      );
    } finally {
      setLoading(false);
    }
  };

  // Coordinates for the historical and predicted paths
  const historicalPath: [number, number][] = prediction
    ? [
        [Number(previousLat), Number(previousLon)],
        [Number(currentLat), Number(currentLon)],
      ]
    : [];

  const predictedPath: [number, number][] = prediction
    ? [
        [Number(currentLat), Number(currentLon)],
        ...prediction.forecasts.map(
          (point) => [point.latitude, point.longitude] as [number, number]
        ),
      ]
    : [];

  const fitPositions = useMemo(
    () => (prediction ? [...historicalPath, ...predictedPath] : []),
    [prediction, previousLat, previousLon, currentLat, currentLon]
  );

  const currentPosition: [number, number] = [
    Number(currentLat),
    Number(currentLon),
  ];

  return (
    <div className="cyclone-map-widget">
      {/* Prediction input panel */}
      <section className="map-console">
        <div className="map-console-header">
          <div>
            <span className="map-console-kicker">
              GEOSPATIAL MONITORING
            </span>
            <h2>North Indian Ocean</h2>
            <p>ML-powered cyclone trajectory prediction</p>
          </div>

          <div className="map-console-status">
            <span className="map-status-light" />
            XGBOOST PREDICTION
          </div>
        </div>

        <form
          onSubmit={handlePrediction}
          style={{
            padding: "20px",
            marginBottom: "20px",
            background: "rgba(20, 40, 60, 0.55)",
            border: "1px solid rgba(53, 213, 239, 0.25)",
            borderRadius: "12px",
          }}
        >
          <h3 style={{ marginTop: 0 }}>Cyclone Observation Input</h3>

          <p style={{ color: "#94a3b8", fontSize: "13px" }}>
            Enter two consecutive observations of the cyclone.
          </p>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
              gap: "20px",
            }}
          >
            {/* Previous observation */}
            <div>
              <h4>Previous Observation</h4>

              <label>Latitude</label>
              <input
                type="number"
                step="any"
                value={previousLat}
                onChange={(e) => setPreviousLat(e.target.value)}
                required
                style={inputStyle}
              />

              <label>Longitude</label>
              <input
                type="number"
                step="any"
                value={previousLon}
                onChange={(e) => setPreviousLon(e.target.value)}
                required
                style={inputStyle}
              />

              <label>Timestamp</label>
              <input
                type="datetime-local"
                value={previousTime}
                onChange={(e) => setPreviousTime(e.target.value)}
                required
                style={inputStyle}
              />
            </div>

            {/* Current observation */}
            <div>
              <h4>Current Observation</h4>

              <label>Latitude</label>
              <input
                type="number"
                step="any"
                value={currentLat}
                onChange={(e) => setCurrentLat(e.target.value)}
                required
                style={inputStyle}
              />

              <label>Longitude</label>
              <input
                type="number"
                step="any"
                value={currentLon}
                onChange={(e) => setCurrentLon(e.target.value)}
                required
                style={inputStyle}
              />

              <label>Timestamp</label>
              <input
                type="datetime-local"
                value={currentTime}
                onChange={(e) => setCurrentTime(e.target.value)}
                required
                style={inputStyle}
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            style={{
              marginTop: "20px",
              padding: "12px 24px",
              borderRadius: "8px",
              border: "none",
              background: loading ? "#64748b" : "#35d5ef",
              color: "#071522",
              fontWeight: 700,
              cursor: loading ? "wait" : "pointer",
            }}
          >
            {loading ? "Generating Prediction..." : "Generate Prediction"}
          </button>

          {error && (
            <p style={{ color: "#ff647c", marginTop: "12px" }}>
              {error}
            </p>
          )}
        </form>

        {/* Prediction results */}
        {prediction && (
          <div
            style={{
              padding: "18px",
              marginBottom: "20px",
              background: "rgba(20, 40, 60, 0.55)",
              border: "1px solid rgba(53, 213, 239, 0.25)",
              borderRadius: "12px",
            }}
          >
            <h3>Prediction Results</h3>

            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: "24px",
                marginBottom: "16px",
              }}
            >
              <span>
                <strong>Model:</strong> {prediction.model_name}
              </span>

              <span>
                <strong>Input interval:</strong>{" "}
                {prediction.input_interval_hours.toFixed(1)} hours
              </span>

              <span>
                <strong>Movement speed:</strong>{" "}
                {prediction.movement_speed_kmh.toFixed(2)} km/h
              </span>

              <span>
                <strong>Heading:</strong>{" "}
                {prediction.movement_heading_degrees.toFixed(1)}°
              </span>
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
                gap: "12px",
              }}
            >
              {prediction.forecasts.map((point) => (
                <div
                  key={point.horizon_hours}
                  style={{
                    padding: "14px",
                    borderRadius: "8px",
                    background: "rgba(53, 213, 239, 0.08)",
                  }}
                >
                  <strong>+{point.horizon_hours} Hours</strong>
                  <p>
                    Lat: {point.latitude.toFixed(4)}°
                  </p>
                  <p>
                    Lon: {point.longitude.toFixed(4)}°
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Map header */}
        <div className="map-coordinate-bar">
          <span>REGION: NORTH INDIAN OCEAN</span>
          <span>
            {prediction
              ? `LAT ${currentPosition[0].toFixed(2)}° | LON ${currentPosition[1].toFixed(2)}°`
              : "AWAITING OBSERVATIONS"}
          </span>
        </div>

        {/* Interactive map */}
        <div className="map-canvas">
          <MapContainer
            center={[15, 82]}
            zoom={5}
            scrollWheelZoom={true}
            zoomControl={true}
            style={{ height: "100%", width: "100%" }}
          >
            <FitMapToPrediction positions={fitPositions} />
            <TileLayer
              attribution="&copy; OpenStreetMap contributors"
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />

            {/* Historical path */}
            {prediction && showHistoricalPath && (
              <Polyline
                positions={historicalPath}
                pathOptions={{
                  color: "#35d5ef",
                  weight: 4,
                  opacity: 0.95,
                }}
              />
            )}

            {/* Predicted path */}
            {prediction && showPredictedPath && (
              <Polyline
                positions={predictedPath}
                pathOptions={{
                  color: "#f4ad65",
                  weight: 4,
                  opacity: 0.95,
                  dashArray: "8 8",
                }}
              />
            )}

            {/* Previous observation */}
            {prediction && showHistoricalPath && (
              <CircleMarker
                center={[Number(previousLat), Number(previousLon)]}
                radius={7}
                pathOptions={{
                  color: "#eafaff",
                  fillColor: "#35d5ef",
                  fillOpacity: 1,
                  weight: 2,
                }}
              >
                <Popup>Previous Observation</Popup>
              </CircleMarker>
            )}

            {/* Current cyclone position */}
            {prediction && (
              <CircleMarker
                center={currentPosition}
                radius={10}
                pathOptions={{
                  color: "#ffffff",
                  fillColor: "#ff647c",
                  fillOpacity: 1,
                  weight: 2,
                }}
              >
                <Popup>
                  <strong>Current Cyclone Position</strong>
                  <br />
                  Latitude: {currentPosition[0]}
                  <br />
                  Longitude: {currentPosition[1]}
                </Popup>
              </CircleMarker>
            )}

            {/* Forecast positions */}
            {showPredictedPath && prediction?.forecasts.map((point) => (
              <CircleMarker
                key={point.horizon_hours}
                center={[point.latitude, point.longitude]}
                radius={8}
                pathOptions={{
                  color: "#ffffff",
                  fillColor: "#f4ad65",
                  fillOpacity: 1,
                  weight: 2,
                }}
              >
                <Popup>
                  <strong>+{point.horizon_hours} Hour Forecast</strong>
                  <br />
                  Latitude: {point.latitude.toFixed(4)}
                  <br />
                  Longitude: {point.longitude.toFixed(4)}
                </Popup>
              </CircleMarker>
            ))}
          </MapContainer>

          <div className="map-corner-label">
            <span>◉</span> TRACKING VIEW
          </div>
        </div>

        {/* Legend */}
        <div className="map-legend">
          <div className="map-legend-title">MAP LEGEND</div>

          <div className="map-legend-items">
            {showHistoricalPath && (
              <span>
                <i className="legend-blue legend-line" />
                Historical path
              </span>
            )}

            {showPredictedPath && (
              <span>
                <i className="legend-orange legend-line" />
                Predicted path
              </span>
            )}
          </div>

          <p className="map-legend-hint">
            Generate a prediction to display the cyclone's forecast trajectory.
          </p>
        </div>
      </section>
    </div>
  );
}

// Input styling
const inputStyle: React.CSSProperties = {
  display: "block",
  width: "100%",
  boxSizing: "border-box",
  padding: "10px",
  marginTop: "6px",
  marginBottom: "12px",
  borderRadius: "6px",
  border: "1px solid rgba(53, 213, 239, 0.3)",
  background: "#0b1726",
  color: "#ffffff",
};

export default CycloneMap;
