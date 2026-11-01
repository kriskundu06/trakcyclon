import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";

interface RegionStats {
  code: string;
  region: string;
  observations: number;
  cyclones: number;
}

interface DatasetStats {
  total_observations: number;
  total_cyclones: number;
  monitored_regions: number;
  max_wind_kmh: number;
  dataset_start: string;
  dataset_end: string;
  regions: RegionStats[];
}

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const COLORS = ["#35d5ef", "#a78bfa", "#f4a85e"];

const chartText = {
  fill: "#91a8bd",
  fontSize: 10,
};

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload || !payload.length) return null;

  return (
    <div
      style={{
        background: "#0b1725",
        border: "1px solid #29465c",
        borderRadius: "8px",
        padding: "11px 14px",
        color: "#e7f0fa",
      }}
    >
      <p style={{ color: "#91a8bd", fontSize: "10px", marginBottom: "6px" }}>
        {label || payload[0].name}
      </p>
      <strong style={{ color: "#35d5ef", fontSize: "14px" }}>
        {Number(payload[0].value).toLocaleString("en-IN")}
      </strong>
    </div>
  );
}

function formatDate(value: string) {
  if (!value) return "—";
  const date = new Date(`${value}T00:00:00`);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleDateString("en-IN", {
        day: "2-digit",
        month: "short",
        year: "numeric",
      });
}

function Analytics() {
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadDatasetStats() {
      try {
        setLoading(true);
        setError("");

        const response = await fetch(`${API_BASE_URL}/dataset-stats`);
        if (!response.ok) {
          throw new Error(`Dataset statistics request failed (${response.status})`);
        }

        const data: DatasetStats = await response.json();
        if (!cancelled) setStats(data);
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof Error
              ? err.message
              : "Unable to load dataset statistics."
          );
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadDatasetStats();

    return () => {
      cancelled = true;
    };
  }, []);

  const regionData = stats?.regions ?? [];
  const totalObservations = stats?.total_observations ?? 0;
  const dateRange =
    stats?.dataset_start && stats?.dataset_end
      ? `${formatDate(stats.dataset_start)} – ${formatDate(stats.dataset_end)}`
      : "—";

  return (
    <div className="analytics-page">
      <style>{`

        .analytics-page {
          color: #e7f0fa;
          width: 100%;
        }

        .analytics-hero {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 15px;
          margin-bottom: 27px;
        }

        .analytics-kicker {
          color: #35d5ef;
          font-family: 'DM Mono', monospace;
          font-size: 9px;
          letter-spacing: 1.6px;
          margin-bottom: 9px;
        }

        .analytics-hero h1 {
          color: #f0f6fc;
          font-size: 30px;
          letter-spacing: -1px;
          margin: 0;
        }

        .analytics-hero p {
          color: #849aaf;
          font-size: 11px;
          margin-top: 8px;
          line-height: 1.6;
        }

        .analytics-demo {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 9px 12px;
          color: #71d8ed;
          background: rgba(53,213,239,0.07);
          border: 1px solid #285267;
          border-radius: 6px;
          font-family: 'DM Mono', monospace;
          font-size: 9px;
          white-space: nowrap;
        }

        .analytics-demo span {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: #35d5ef;
          box-shadow: 0 0 8px #35d5ef;
        }

        .analytics-section-label {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 10px;
          margin-bottom: 13px;
          color: #a9bdcf;
          font-family: 'DM Mono', monospace;
          font-size: 9px;
          letter-spacing: 1.2px;
        }

        .analytics-section-label small {
          color: #627b91;
          font-size: 8px;
        }

        .analytics-cards {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 13px;
          margin-bottom: 26px;
        }

        .analytics-card {
          min-width: 0;
          min-height: 125px;
          padding: 17px;
          position: relative;
          overflow: hidden;
          background: linear-gradient(145deg, #132438, #0f1c2c);
          border: 1px solid #20364a;
          border-radius: 10px;
        }

        .analytics-card-top {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 7px;
        }

        .analytics-card-label {
          color: #91a8bc;
          font-family: 'DM Mono', monospace;
          font-size: 8px;
          letter-spacing: .7px;
        }

        .analytics-card-icon {
          width: 27px;
          height: 27px;
          display: grid;
          place-items: center;
          border-radius: 7px;
          font-size: 15px;
          background: rgba(53,213,239,.1);
          color: #35d5ef;
        }

        .analytics-card h2 {
          margin-top: 14px;
          color: #edf8ff;
          font-size: 25px;
          letter-spacing: -.7px;
          overflow-wrap: anywhere;
        }

        .analytics-card h2 span {
          color: #8da6bb;
          font-size: 11px;
          letter-spacing: 0;
          font-weight: 500;
        }

        .analytics-card small {
          display: block;
          margin-top: 6px;
          color: #71899e;
          font-size: 9px;
        }

        .analytics-card::after {
          content: "";
          position: absolute;
          height: 2px;
          bottom: 0;
          left: 0;
          right: 0;
          background: #35d5ef;
          opacity: .65;
        }

        .analytics-card.orange::after { background: #f4a85e; }
        .analytics-card.purple::after { background: #a78bfa; }
        .analytics-card.green::after { background: #48d6a0; }

        .analytics-card.orange .analytics-card-icon {
          color: #f4a85e;
          background: rgba(244,168,94,.1);
        }

        .analytics-card.purple .analytics-card-icon {
          color: #a78bfa;
          background: rgba(167,139,250,.1);
        }

        .analytics-card.green .analytics-card-icon {
          color: #48d6a0;
          background: rgba(72,214,160,.1);
        }

        .analytics-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 16px;
        }

        .analytics-panel {
          min-width: 0;
          padding: 19px;
          background: #101d2d;
          border: 1px solid #20364a;
          border-radius: 10px;
          overflow: hidden;
        }

        .analytics-panel-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 10px;
          margin-bottom: 17px;
        }

        .analytics-panel h2 {
          color: #e6f1fa;
          font-size: 14px;
          margin: 0;
        }

        .analytics-panel p {
          margin-top: 6px;
          color: #8197ab;
          font-size: 10px;
          line-height: 1.5;
        }

        .analytics-panel-tag {
          padding: 5px 7px;
          border-radius: 4px;
          background: #1b3042;
          color: #91a9bc;
          font-family: 'DM Mono', monospace;
          font-size: 8px;
          white-space: nowrap;
        }

        .analytics-chart {
          width: 100%;
          height: 255px;
        }

        .analytics-chart.large {
          height: 285px;
        }

        .analytics-chart-legend {
          display: flex;
          flex-wrap: wrap;
          gap: 15px;
          margin-top: 7px;
          color: #91a8bc;
          font-size: 9px;
        }

        .analytics-chart-legend span {
          display: flex;
          align-items: center;
          gap: 6px;
        }

        .analytics-chart-legend i {
          width: 8px;
          height: 8px;
          border-radius: 50%;
        }

        .analytics-trend-summary {
          display: flex;
          gap: 20px;
          flex-wrap: wrap;
          margin-top: 10px;
          padding-top: 13px;
          border-top: 1px solid #203247;
        }

        .analytics-trend-summary span {
          display: block;
          color: #71899e;
          font-size: 9px;
          margin-bottom: 5px;
        }

        .analytics-trend-summary strong {
          color: #dceaf5;
          font-size: 13px;
        }

        .analytics-full {
          grid-column: 1 / -1;
        }

        .analytics-disclaimer {
          display: flex;
          gap: 8px;
          margin-top: 18px;
          color: #71879b;
          font-size: 9px;
          line-height: 1.6;
        }

        @media (max-width: 1050px) {
          .analytics-cards {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
        }

        @media (max-width: 700px) {
          .analytics-grid {
            grid-template-columns: 1fr;
          }

          .analytics-full {
            grid-column: auto;
          }

          .analytics-hero h1 {
            font-size: 25px;
          }
        }

        @media (max-width: 430px) {
          .analytics-hero {
            flex-direction: column;
          }

          .analytics-cards {
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 9px;
          }

          .analytics-card {
            padding: 12px;
          }

          .analytics-card h2 {
            font-size: 21px;
          }

          .analytics-panel {
            padding: 13px;
          }

          .analytics-chart {
            height: 230px;
          }

          .analytics-chart.large {
            height: 245px;
          }
        }
      
      `}</style>

      <header className="analytics-hero">
        <div>
          <div className="analytics-kicker">INTELLIGENCE / DATA ANALYSIS</div>
          <h1>Cyclone Analytics</h1>
          <p>North Indian Ocean historical dataset overview</p>
        </div>

        <span className="analytics-demo">
          <span /> {loading ? "LOADING DATA" : error ? "API UNAVAILABLE" : "LIVE DATASET"}
        </span>
      </header>

      {error && (
        <div
          role="alert"
          style={{
            marginBottom: 18,
            padding: "12px 15px",
            border: "1px solid #8b4b4b",
            borderRadius: 8,
            background: "rgba(180, 65, 65, 0.10)",
            color: "#ffb4b4",
            fontSize: 12,
          }}
        >
          Could not load analytics: {error}. Check that the FastAPI backend is
          running and that <code>/dataset-stats</code> is available.
        </div>
      )}

      <div className="analytics-section-label">
        <span>DATASET OVERVIEW</span>
        <small>{loading ? "FETCHING FROM API" : "BACKEND DATA"}</small>
      </div>

      <div className="analytics-cards">
        <div className="analytics-card">
          <div className="analytics-card-top">
            <span className="analytics-card-label">TOTAL OBSERVATIONS</span>
            <span className="analytics-card-icon">▤</span>
          </div>
          <h2>{loading ? "—" : totalObservations.toLocaleString("en-IN")}</h2>
          <small>Historical cyclone records</small>
        </div>

        <div className="analytics-card orange">
          <div className="analytics-card-top">
            <span className="analytics-card-label">UNIQUE CYCLONES</span>
            <span className="analytics-card-icon">◎</span>
          </div>
          <h2>{loading ? "—" : (stats?.total_cyclones ?? 0).toLocaleString("en-IN")}</h2>
          <small>Across the available dataset</small>
        </div>

        <div className="analytics-card purple">
          <div className="analytics-card-top">
            <span className="analytics-card-label">MONITORED REGIONS</span>
            <span className="analytics-card-icon">⌖</span>
          </div>
          <h2>{loading ? "—" : stats?.monitored_regions ?? "—"}</h2>
          <small>Arabian Sea and Bay of Bengal</small>
        </div>

        <div className="analytics-card green">
          <div className="analytics-card-top">
            <span className="analytics-card-label">MAXIMUM WIND SPEED</span>
            <span className="analytics-card-icon">↗</span>
          </div>
          <h2>
            {loading ? "—" : (stats?.max_wind_kmh ?? 0).toLocaleString("en-IN", { maximumFractionDigits: 1 })}
            <span> km/h</span>
          </h2>
          <small>Maximum recorded dataset value</small>
        </div>
      </div>

      <div className="analytics-section-label">
        <span>REGIONAL BREAKDOWN</span>
        <small>NORTH INDIAN OCEAN</small>
      </div>

      <div className="analytics-grid">
        <section className="analytics-panel">
          <div className="analytics-panel-header">
            <div>
              <h2>Observations by Region</h2>
              <p>Historical records available in each sub-basin</p>
            </div>
            <span className="analytics-panel-tag">RECORDS</span>
          </div>

          <div className="analytics-chart">
            {regionData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={regionData} margin={{ top: 10, right: 5, left: -18, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 5" stroke="#24384b" vertical={false} />
                  <XAxis dataKey="region" tick={{ ...chartText, fontSize: 9 }} axisLine={false} tickLine={false} interval={0} />
                  <YAxis tick={chartText} axisLine={false} tickLine={false} />
                  <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(53,213,239,.06)" }} />
                  <Bar dataKey="observations" name="Observations" fill="#35d5ef" radius={[6, 6, 0, 0]} maxBarSize={55} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ color: "#8197ab", fontSize: 12, padding: 20 }}>
                {loading ? "Loading regional data…" : "No regional data available."}
              </div>
            )}
          </div>
          <div className="analytics-chart-legend">
            <span><i style={{ background: "#35d5ef" }} /> Historical observations</span>
          </div>
        </section>

        <section className="analytics-panel">
          <div className="analytics-panel-header">
            <div>
              <h2>Observation Share</h2>
              <p>Proportion of records contributed by each region</p>
            </div>
            <span className="analytics-panel-tag">SHARE</span>
          </div>

          <div className="analytics-chart">
            {regionData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={regionData}
                    dataKey="observations"
                    nameKey="region"
                    cx="50%"
                    cy="48%"
                    innerRadius={53}
                    outerRadius={83}
                    paddingAngle={4}
                    stroke="none"
                  >
                    {regionData.map((entry, index) => (
                      <Cell key={entry.code} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                  <Legend
                    verticalAlign="bottom"
                    iconType="circle"
                    iconSize={7}
                    formatter={(value) => (
                      <span style={{ color: "#9db1c3", fontSize: "9px" }}>{value}</span>
                    )}
                  />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ color: "#8197ab", fontSize: 12, padding: 20 }}>
                {loading ? "Loading regional data…" : "No regional data available."}
              </div>
            )}
          </div>
        </section>

        <section className="analytics-panel analytics-full">
          <div className="analytics-panel-header">
            <div>
              <h2>Cyclones by Region</h2>
              <p>Number of distinct cyclone tracks represented in each sub-basin</p>
            </div>
            <span className="analytics-panel-tag">CYCLONES</span>
          </div>

          <div className="analytics-chart large">
            {regionData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={regionData} margin={{ top: 12, right: 12, left: -15, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 5" stroke="#24384b" vertical={false} />
                  <XAxis dataKey="region" tick={chartText} axisLine={false} tickLine={false} />
                  <YAxis tick={chartText} axisLine={false} tickLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="cyclones" name="Cyclones" fill="#a78bfa" radius={[6, 6, 0, 0]} maxBarSize={70} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ color: "#8197ab", fontSize: 12, padding: 20 }}>
                {loading ? "Loading regional data…" : "No regional data available."}
              </div>
            )}
          </div>

          <div className="analytics-trend-summary">
            <div>
              <span>DATASET START</span>
              <strong>{loading ? "—" : formatDate(stats?.dataset_start ?? "")}</strong>
            </div>
            <div>
              <span>DATASET END</span>
              <strong>{loading ? "—" : formatDate(stats?.dataset_end ?? "")}</strong>
            </div>
            <div>
              <span>DATE COVERAGE</span>
              <strong>{loading ? "—" : dateRange}</strong>
            </div>
          </div>
        </section>
      </div>

      <div className="analytics-disclaimer">
        <span>ⓘ</span>
        These charts summarize historical dataset records returned by the backend.
        Regional cyclone counts may overlap because a cyclone can pass through more
        than one sub-basin. This page does not show live cyclone activity or official forecasts.
      </div>
    </div>
  );
}

export default Analytics;
