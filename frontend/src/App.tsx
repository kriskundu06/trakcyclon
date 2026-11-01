import { useEffect, useState } from "react";
import { getDatasetStats, type DatasetStats } from "./services/cycloneApi";
import Analytics from "./pages/analytics";
import History from "./pages/History";
import Settings from "./pages/Settings";
import Help from "./pages/Help";
import CycloneMap from "./components/CycloneMap";
import "./App.css";

const navigation = [
  { name: "Dashboard", icon: "◫", group: "MONITOR" },
  { name: "Cyclone Map", icon: "◎", group: "MONITOR" },
  { name: "Analytics", icon: "▥", group: "INSIGHTS" },
  { name: "History", icon: "◷", group: "INSIGHTS" },
  { name: "Settings", icon: "⚙", group: "SYSTEM" },
  { name: "Help", icon: "?", group: "SYSTEM" },
];

function App() {
  const [activePage, setActivePage] = useState("Dashboard");
  const [datasetStats, setDatasetStats] = useState<DatasetStats | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);
  const [statsError, setStatsError] = useState("");

  const [historicalPath, setHistoricalPath] = useState<boolean>(() => {
    const stored = localStorage.getItem("trakcyclon-historical-path");
    return stored === null ? true : stored === "true";
  });

  const [predictedPath, setPredictedPath] = useState<boolean>(() => {
    const stored = localStorage.getItem("trakcyclon-predicted-path");
    return stored === null ? true : stored === "true";
  });

  useEffect(() => {
    localStorage.setItem("trakcyclon-historical-path", String(historicalPath));
    localStorage.setItem("trakcyclon-predicted-path", String(predictedPath));
  }, [historicalPath, predictedPath]);

  useEffect(() => {
    getDatasetStats()
      .then((data) => {
        setDatasetStats(data);
        setStatsError("");
      })
      .catch((error) => {
        setStatsError(
          error instanceof Error ? error.message : "Unable to load statistics."
        );
      })
      .finally(() => {
        setStatsLoading(false);
      });
  }, []);

  const renderPage = () => {
    switch (activePage) {
      case "Cyclone Map":
        return (
          <section className="page-content">
            <div className="page-heading">
              <div>
                <div className="eyebrow">MONITORING / MAP</div>
                <h1>Cyclone Tracking Map</h1>
                <p>North Indian Ocean cyclone visualization</p>
              </div>
            </div>
            <div className="map-page-container">
              <CycloneMap
                showHistoricalPath={historicalPath}
                showPredictedPath={predictedPath}
              />
            </div>
          </section>
        );

      case "Analytics":
        return (
          <div className="page-content">
            <Analytics />
          </div>
        );

      case "History":
        return (
          <div className="page-content">
            <History />
          </div>
        );

      case "Settings":
        return (
          <div className="page-content">
            <Settings
              historicalPath={historicalPath}
              predictedPath={predictedPath}
              onHistoricalPathChange={setHistoricalPath}
              onPredictedPathChange={setPredictedPath}
            />
          </div>
        );

      case "Help":
        return (
          <div className="page-content">
            <Help />
          </div>
        );

      default:
        return (
          <div className="page-content">
            <div className="page-heading">
              <div>
                <div className="eyebrow">COMMAND CENTER / OVERVIEW</div>
                <h1>Monitoring Dashboard</h1>
                <p>North Indian Ocean cyclone activity at a glance</p>
              </div>
            </div>

            <div className="welcome-strip">
              <div>
                <span className="welcome-label">TRAKCYCLON INTELLIGENCE</span>
                <h2>Track the storm. Understand the path.</h2>
                <p>
                  Explore cyclone movement across the Bay of Bengal and Arabian
                  Sea.
                </p>
              </div>
              <div className="welcome-symbol">〰</div>
            </div>

            <div className="section-label">
              <span>REGIONAL OVERVIEW</span>
              <span className="section-caption">
                HISTORICAL DATASET STATISTICS
              </span>
            </div>

            <section className="stats-grid">
              <div className="stat-card">
                <div className="stat-top">
                  <span className="stat-label">HISTORICAL OBSERVATIONS</span>
                  <span className="stat-icon cyan">◎</span>
                </div>
                <h2>
                  {statsLoading
                    ? "..."
                    : datasetStats?.total_observations.toLocaleString("en-IN") ??
                      "--"}
                </h2>
                <small>Recorded cyclone observations</small>
                <div className="stat-accent cyan-line" />
              </div>

              <div className="stat-card">
                <div className="stat-top">
                  <span className="stat-label">UNIQUE CYCLONES</span>
                  <span className="stat-icon orange">↗</span>
                </div>
                <h2>
                  {statsLoading
                    ? "..."
                    : datasetStats?.total_cyclones.toLocaleString("en-IN") ??
                      "--"}
                </h2>
                <small>Distinct historical cyclone IDs</small>
                <div className="stat-accent orange-line" />
              </div>

              <div className="stat-card">
                <div className="stat-top">
                  <span className="stat-label">MAX RECORDED WIND</span>
                  <span className="stat-icon purple">⌖</span>
                </div>
                <h2>
                  {statsLoading
                    ? "..."
                    : datasetStats?.max_wind_kmh != null
                      ? datasetStats.max_wind_kmh.toFixed(1)
                      : "--"}
                  <span> km/h</span>
                </h2>
                <small>Historical dataset maximum</small>
                <div className="stat-accent purple-line" />
              </div>

              <div className="stat-card">
                <div className="stat-top">
                  <span className="stat-label">FORECAST HORIZONS</span>
                  <span className="stat-icon green">◷</span>
                </div>
                <h2>
                  6 / 12 / 24 <span>HRS</span>
                </h2>
                <small>XGBoost prediction intervals</small>
                <div className="stat-accent green-line" />
              </div>
            </section>

            {statsError && (
              <p style={{ color: "#ff647c", marginTop: "10px" }}>
                Dataset statistics unavailable: {statsError}
              </p>
            )}

            <section className="dashboard-map-section">
              <div className="section-heading">
                <div>
                  <div className="eyebrow">GEOSPATIAL VIEW</div>
                  <h2>North Indian Ocean</h2>
                  <p>Generate and explore cyclone trajectory predictions</p>
                </div>
                <button
                  className="outline-action"
                  onClick={() => setActivePage("Cyclone Map")}
                >
                  Open full map <span>↗</span>
                </button>
              </div>
              <div className="dashboard-map">
                <CycloneMap
                  showHistoricalPath={historicalPath}
                  showPredictedPath={predictedPath}
                />
              </div>
            </section>

            <section className="dashboard-bottom">
              <div className="command-panel">
                <div className="panel-heading">
                  <div>
                    <div className="eyebrow">HISTORICAL DATA</div>
                    <h3>Regional activity</h3>
                  </div>
                  <span className="small-live-label">DATASET</span>
                </div>

                {datasetStats?.regions.map((region) => (
                  <div className="activity-item" key={region.code}>
                    <span
                      className={`activity-dot ${
                        region.code === "BB" ? "orange-dot" : "blue-dot"
                      }`}
                    />
                    <div className="activity-info">
                      <strong>{region.region}</strong>
                      <small>
                        {region.observations.toLocaleString("en-IN")}{" "}
                        observations · {region.cyclones.toLocaleString("en-IN")}{" "}
                        cyclone records
                      </small>
                    </div>
                    <span className="activity-tag">HISTORICAL</span>
                  </div>
                ))}

                {!datasetStats && !statsLoading && (
                  <p style={{ padding: "12px" }}>
                    Regional statistics are currently unavailable.
                  </p>
                )}
              </div>

              <div className="command-panel about-panel">
                <div className="eyebrow">ABOUT THE SYSTEM</div>
                <h3>TrakCyclon</h3>
                <p>
                  A cyclone path visualization and prediction project focused
                  on tropical cyclone movement in the North Indian Ocean.
                </p>
                <button
                  className="text-action"
                  onClick={() => setActivePage("Help")}
                >
                  Explore user guide <span>→</span>
                </button>
              </div>
            </section>

            <div className="dashboard-disclaimer">
              <span>ⓘ</span>
              Statistics are derived from historical cyclone observations.
              XGBoost predictions are experimental model outputs, not official
              meteorological forecasts or warnings.
            </div>
          </div>
        );
    }
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            T<span>✳</span>
          </div>
          <div className="brand-name">
            <strong>TrakCyclon</strong>
            <small>CYCLONE PATH PREDICTOR</small>
          </div>
        </div>

        <div className="sidebar-divider" />

        <div className="nav-group">
          <span className="nav-heading">MONITORING</span>
          {navigation
            .filter((item) => item.group === "MONITOR")
            .map((item) => (
              <button
                key={item.name}
                className={`nav-item ${
                  activePage === item.name ? "active" : ""
                }`}
                onClick={() => setActivePage(item.name)}
              >
                <span className="nav-icon">{item.icon}</span>
                <span>{item.name}</span>
                {activePage === item.name && (
                  <span className="nav-active-mark" />
                )}
              </button>
            ))}
        </div>

        <div className="nav-group">
          <span className="nav-heading">INTELLIGENCE</span>
          {navigation
            .filter((item) => item.group === "INSIGHTS")
            .map((item) => (
              <button
                key={item.name}
                className={`nav-item ${
                  activePage === item.name ? "active" : ""
                }`}
                onClick={() => setActivePage(item.name)}
              >
                <span className="nav-icon">{item.icon}</span>
                <span>{item.name}</span>
                {activePage === item.name && (
                  <span className="nav-active-mark" />
                )}
              </button>
            ))}
        </div>

        <div className="nav-group">
          <span className="nav-heading">PREFERENCES</span>
          {navigation
            .filter((item) => item.group === "SYSTEM")
            .map((item) => (
              <button
                key={item.name}
                className={`nav-item ${
                  activePage === item.name ? "active" : ""
                }`}
                onClick={() => setActivePage(item.name)}
              >
                <span className="nav-icon">{item.icon}</span>
                <span>{item.name}</span>
                {activePage === item.name && (
                  <span className="nav-active-mark" />
                )}
              </button>
            ))}
        </div>

        <div className="sidebar-spacer" />

        <div className="sidebar-system">
          <span className="system-indicator" />
          <div>
            <strong>
              {statsLoading
                ? "Connecting to backend"
                : statsError
                  ? "Backend unavailable"
                  : "Backend connected"}
            </strong>
          </div>
        </div>

        <div className="profile">
          <div className="avatar">TC</div>
          <div className="profile-info">
            <strong>TrakCyclon User</strong>
            <small>Weather Analyst</small>
          </div>
          <span className="profile-menu">···</span>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">
            <span>TRAKCYCLON</span>
            <span className="breadcrumb-separator">/</span>
            <strong>{activePage}</strong>
          </div>
          <div className="topbar-right">
            <span className="topbar-region">⌖ NORTH INDIAN OCEAN</span>
          </div>
        </header>

        {renderPage()}

        <footer className="app-footer">
          <span>TRAKCYCLON © 2026</span>
          <span>
            CYCLONE PATH PREDICTOR <b>·</b> HISTORICAL DATA + ML
          </span>
        </footer>
      </main>
    </div>
  );
}

export default App;
