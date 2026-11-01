import { useEffect, useState } from "react";
import "./History.css";

interface CycloneRecord {
  id: string;
  name: string;
  region: string;
  region_code: string;
  start_date: string;
  end_date: string;
  peak_wind_kmh: number | null;
  observations: number;
}

interface HistoryResponse {
  total: number;
  limit: number;
  offset: number;
  region_totals: {
    bay_of_bengal: number;
    arabian_sea: number;
  };
  records: CycloneRecord[];
}

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const PAGE_SIZE = 25;

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

function History() {
  const [search, setSearch] = useState("");
  const [region, setRegion] = useState("all");
  const [page, setPage] = useState(0);
  const [data, setData] = useState<HistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadHistory() {
      try {
        setLoading(true);
        setError("");

        const params = new URLSearchParams({
          search: search.trim(),
          region,
          limit: String(PAGE_SIZE),
          offset: String(page * PAGE_SIZE),
        });

        const response = await fetch(`${API_BASE_URL}/history?${params}`);
        if (!response.ok) {
          const body = await response.json().catch(() => null);
          throw new Error(
            body?.detail || `History request failed (${response.status})`
          );
        }

        const result: HistoryResponse = await response.json();
        if (!cancelled) setData(result);
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof Error
              ? err.message
              : "Unable to load historical cyclone records."
          );
          setData(null);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadHistory();
    return () => {
      cancelled = true;
    };
  }, [search, region, page]);

  const clearFilters = () => {
    setSearch("");
    setRegion("all");
    setPage(0);
  };

  const total = data?.total ?? 0;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const firstRecord = total === 0 ? 0 : page * PAGE_SIZE + 1;
  const lastRecord = Math.min((page + 1) * PAGE_SIZE, total);

  return (
    <div className="history-page">
      <header className="history-header">
        <div>
          <div className="history-kicker">INTELLIGENCE / ARCHIVE</div>
          <h1>Cyclone History</h1>
          <p>Explore historically recorded cyclone tracks across the North Indian Ocean.</p>
        </div>
        <span className="history-demo"><i /> HISTORICAL DATASET</span>
      </header>

      <div className="history-stats">
        <div className="history-stat">
          <span>TOTAL MATCHING RECORDS</span>
          <strong>{loading ? "—" : total.toLocaleString("en-IN")}</strong>
        </div>
        <div className="history-stat">
          <span>BAY OF BENGAL</span>
          <strong>
            {loading ? "—" : (data?.region_totals.bay_of_bengal ?? 0).toLocaleString("en-IN")}
          </strong>
        </div>
        <div className="history-stat">
          <span>ARABIAN SEA</span>
          <strong>
            {loading ? "—" : (data?.region_totals.arabian_sea ?? 0).toLocaleString("en-IN")}
          </strong>
        </div>
        <div className="history-stat">
          <span>RECORDS PER PAGE</span>
          <strong>{PAGE_SIZE}</strong>
        </div>
      </div>

      <section className="history-panel">
        <div className="history-panel-heading">
          <div>
            <div className="history-kicker">RECORD MANAGEMENT</div>
            <h2>Historical cyclone records</h2>
            <p>Search the archive and filter by North Indian Ocean sub-basin.</p>
          </div>
          <span className="history-count">
            {loading ? "LOADING…" : `${total.toLocaleString("en-IN")} RESULTS`}
          </span>
        </div>

        <div className="history-filters">
          <div className="history-search">
            <span>⌕</span>
            <input
              type="text"
              placeholder="Search cyclone name or SID..."
              value={search}
              onChange={(event) => {
                setSearch(event.target.value);
                setPage(0);
              }}
            />
          </div>

          <select
            value={region}
            onChange={(event) => {
              setRegion(event.target.value);
              setPage(0);
            }}
          >
            <option value="all">All Regions</option>
            <option value="BB">Bay of Bengal</option>
            <option value="AS">Arabian Sea</option>
          </select>

          <button className="history-clear" onClick={clearFilters}>
            Reset ↺
          </button>
        </div>

        {error && (
          <div
            role="alert"
            style={{
              margin: "0 0 15px",
              padding: "12px 14px",
              border: "1px solid #8b4b4b",
              borderRadius: 8,
              background: "rgba(180, 65, 65, 0.10)",
              color: "#ffb4b4",
              fontSize: 12,
            }}
          >
            Could not load history: {error}. Check that the backend is running
            and that the <code>/history</code> endpoint is available.
          </div>
        )}

        <div className="history-table-wrap">
          <table className="history-table">
            <thead>
              <tr>
                <th>CYCLONE ID (SID)</th>
                <th>CYCLONE</th>
                <th>REGION</th>
                <th>PERIOD</th>
                <th>PEAK WIND</th>
                <th>OBSERVATIONS</th>
              </tr>
            </thead>
            <tbody>
              {!loading && !error && data?.records.map((record) => (
                <tr key={record.id}>
                  <td><span className="history-id">{record.id}</span></td>
                  <td><strong>{record.name}</strong></td>
                  <td>{record.region}</td>
                  <td>
                    {formatDate(record.start_date)}
                    {record.start_date !== record.end_date && (
                      <> – {formatDate(record.end_date)}</>
                    )}
                  </td>
                  <td>
                    <strong className="history-wind">
                      {record.peak_wind_kmh === null
                        ? "—"
                        : `${record.peak_wind_kmh.toLocaleString("en-IN")} km/h`}
                    </strong>
                  </td>
                  <td>{record.observations.toLocaleString("en-IN")}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {loading && (
            <div className="history-empty">
              <h3>Loading historical records…</h3>
              <p>Fetching cyclone data from the backend.</p>
            </div>
          )}

          {!loading && !error && data?.records.length === 0 && (
            <div className="history-empty">
              <span>⌕</span>
              <h3>No matching records</h3>
              <p>Try changing your search or region filter.</p>
              <button onClick={clearFilters}>Clear filters</button>
            </div>
          )}
        </div>

        <div className="history-table-footer">
          <span>
            Showing {firstRecord.toLocaleString("en-IN")}–{lastRecord.toLocaleString("en-IN")}
            {" "}of {total.toLocaleString("en-IN")} records
          </span>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <button
              className="history-clear"
              disabled={loading || page === 0}
              onClick={() => setPage((current) => Math.max(0, current - 1))}
            >
              ← Previous
            </button>
            <span>Page {page + 1} of {totalPages}</span>
            <button
              className="history-clear"
              disabled={loading || page + 1 >= totalPages}
              onClick={() => setPage((current) => current + 1)}
            >
              Next →
            </button>
          </div>
        </div>
      </section>

      <div className="history-disclaimer">
        ⓘ Records are derived from the project's historical IBTrACS-based cleaned dataset.
        Peak wind is the maximum available WMO wind value converted from knots to km/h.
        These are historical observations, not live activity or official forecasts.
      </div>
    </div>
  );
}

export default History;
