
import { useState } from "react";
import "./Settings.css";

type SettingsProps = {
  historicalPath: boolean;
  predictedPath: boolean;
  onHistoricalPathChange: (value: boolean) => void;
  onPredictedPathChange: (value: boolean) => void;
};

function Settings({
  historicalPath,
  predictedPath,
  onHistoricalPathChange,
  onPredictedPathChange,
}: SettingsProps) {
  const [windUnit, setWindUnit] = useState("km/h");
  const [temperature, setTemperature] = useState("°C");
  const [saved, setSaved] = useState(false);

  const updateSetting = (callback: (value: any) => void, value: any) => {
    callback(value);
    setSaved(false);
  };

  const resetSettings = () => {
    setWindUnit("km/h");
    setTemperature("°C");
    onHistoricalPathChange(true);
    onPredictedPathChange(true);
    setSaved(false);
  };

  return (
    <div className="settings-page">
      <header className="settings-header">
        <div>
          <div className="settings-kicker">SYSTEM / CONFIGURATION</div>
          <h1>Workspace Settings</h1>
          <p>Configure your TrakCyclon monitoring experience.</p>
        </div>
        <span className="settings-mode"><i /> LOCAL SETTINGS</span>
      </header>

      <div className="settings-notice">
        <span>ⓘ</span>
        <p>
          Map layer preferences apply immediately to the Dashboard and Cyclone
          Map. They do not change backend data or official weather information.
        </p>
      </div>

      <section className="settings-section">
        <div className="settings-section-heading">
          <div className="settings-section-icon">◉</div>
          <div>
            <h2>Display preferences</h2>
            <p>Choose how measurement values appear in your workspace.</p>
          </div>
        </div>

        <div className="settings-row">
          <div>
            <strong>Wind speed unit</strong>
            <small>Select the unit used to display wind speed.</small>
          </div>
          <select
            value={windUnit}
            onChange={(e) => updateSetting(setWindUnit, e.target.value)}
          >
            <option value="km/h">Kilometers per hour (km/h)</option>
            <option value="knots">Knots (kn)</option>
          </select>
        </div>

        <div className="settings-row">
          <div>
            <strong>Temperature unit</strong>
            <small>Select your preferred temperature unit.</small>
          </div>
          <select
            value={temperature}
            onChange={(e) => updateSetting(setTemperature, e.target.value)}
          >
            <option value="°C">Celsius (°C)</option>
            <option value="°F">Fahrenheit (°F)</option>
          </select>
        </div>
      </section>

      <section className="settings-section">
        <div className="settings-section-heading">
          <div className="settings-section-icon">⌖</div>
          <div>
            <h2>Map visualization</h2>
            <p>Choose which track layers you want to display.</p>
          </div>
        </div>

        <div className="settings-row">
          <div>
            <strong>Historical cyclone path</strong>
            <small>Show the illustrative previous movement path.</small>
          </div>
          <button
            className={`settings-toggle ${historicalPath ? "enabled" : ""}`}
            role="switch"
            aria-checked={historicalPath}
            onClick={() =>
              updateSetting(onHistoricalPathChange, !historicalPath)
            }
          >
            <span />
          </button>
        </div>

        <div className="settings-row">
          <div>
            <strong>Predicted cyclone path</strong>
            <small>Show the illustrative projected movement path.</small>
          </div>
          <button
            className={`settings-toggle ${predictedPath ? "enabled" : ""}`}
            role="switch"
            aria-checked={predictedPath}
            onClick={() =>
              updateSetting(onPredictedPathChange, !predictedPath)
            }
          >
            <span />
          </button>
        </div>
      </section>

      <div className="settings-actions">
        <button className="settings-reset" onClick={resetSettings}>
          Reset to defaults
        </button>
        <button className="settings-save" onClick={() => setSaved(true)}>
          Save preferences <span>→</span>
        </button>
      </div>

      {saved && (
        <div className="settings-saved" role="status">
          ✓ Preferences applied to this page's current session.
        </div>
      )}

      <div className="settings-footer-note">
        TRAKCYCLON CONFIGURATION PANEL
      </div>
    </div>
  );
}

export default Settings;
