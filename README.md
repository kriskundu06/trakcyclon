TrakCyclon --- CycloneTrack AI
An end-to-end machine-learning application for exploring historical
tropical-cyclone tracks and generating experimental 6-, 12-, and 24-hour
trajectory predictions for the North Indian Ocean (Bay of Bengal and
Arabian Sea).
Research / educational project: Model predictions are experimental
and are not official meteorological forecasts, warnings, or a
substitute for guidance from national weather agencies. The historical
dataset is not a live cyclone-activity feed.

Overview
TrakCyclon combines a React-based interface with a Python FastAPI
backend and trained XGBoost models. Users can explore historical cyclone
records, inspect regional dataset statistics, and request
model-generated future positions at multiple forecast horizons.
Key features
- Cyclone trajectory prediction: Experimental 6-hour, 12-hour, and
  24-hour predictions.
- Historical cyclone explorer: Browse historical cyclone tracks
  and records.
- Interactive map: View historical and predicted track positions.
- Analytics dashboard: Explore dataset coverage and regional
  statistics.
- History search: Search and filter historical cyclone records.
- Model evaluation artifacts: Includes selected prediction and
  error-analysis outputs.
Technology stack
  Layer               Technologies
  Frontend            React, TypeScript, Vite, CSS
Backend             Python, FastAPI
Machine learning    XGBoost, scikit-learn
Data processing     Pandas, NumPy
Model persistence   Joblib
Mapping             Leaflet (frontend map integration)
Model evaluation
The current test-set results recorded for the XGBoost trajectory models
are:
  Forecast        Mean track   Median track           RMSE           90th
horizon              error          error                    percentile
error
  6 hours           32.73 km       27.37 km       40.51 km       60.55 km

  12 hours          67.61 km       57.09 km       81.56 km      126.48 km
24 hours         138.84 km      119.52 km      164.13 km      256.44 km
These metrics describe the model's evaluation results on its test data;
they do not guarantee future forecast accuracy. Evaluation methodology,
data split details, and further analysis should be consulted alongside
the corresponding scripts and artifacts in this repository.
Dataset
The project uses historical North Indian Ocean cyclone observations
based on IBTrACS-derived data.

Current dataset summary:
- 57,874 observations
- 1,860 unique cyclones
- Arabian Sea: 15,426 observations across 618 cyclones
- Bay of Bengal: 42,448 observations across 1,591 cyclones
- Recorded date range: 1842-10-25 to 2026-10-28
The application uses the processed historical dataset at
data/processed/cyclones_clean.csv. The raw source file and selected
processed data artifacts are included for reproducibility and analysis.
Check the source dataset's terms and attribution requirements before
redistributing it.
Repository structure
trakcyclon/
├── backend/
│   ├── __init__.py
│   └── main.py
├── data/
│   ├── raw/
│   │   └── ibtracs_ni.csv
│   └── processed/
│       ├── cyclones_clean.csv
│       ├── cyclone_features.csv
│       └── selected evaluation outputs
├── frontend/
│   ├── public/
│   └── src/
│       ├── components/
│       ├── pages/
│       └── services/
├── models/
│   ├── final_cyclone_model_6h.joblib
│   ├── final_cyclone_model_12h.joblib
│   └── final_cyclone_model_24h.joblib
├── scripts/
└── .gitignore
Large experimental Random Forest model files and other non-production
model artifacts are intentionally excluded from Git. The three
final_cyclone_model_*.joblib files are the model artifacts loaded by
the API.
Run locally
Prerequisites
- Python 3.10 or a compatible version supported by the project's
  dependencies
- Node.js and npm
- Git
1. Clone the repository
git clone https://github.com/kriskundu06/trakcyclon.git
cd trakcyclon
2. Set up the backend
Create and activate a virtual environment:

Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Install the Python packages required by the backend, then start the API
from the repository root:
python -m uvicorn backend.main:app --reload
The API will be available at http://127.0.0.1:8000.
Add and maintain a backend dependency file (for example,
requirements.txt) so the environment can be reproduced reliably. The
exact pinned dependency versions should match the versions used to
train and serialize the committed models.

3. Set up the frontend
In a second terminal:
cd frontend
npm install
Create frontend/.env with the local API URL:
VITE_API_BASE_URL=http://127.0.0.1:8000
Start the Vite development server:
npm run dev
Open the local URL printed by Vite in your browser.
API endpoints
The FastAPI application currently exposes these routes:

  Endpoint                 Purpose
  GET /                  API root / service information
GET /health            Backend health check
GET /dataset-stats     Dataset coverage and summary statistics
POST /predict          Generate cyclone trajectory predictions
GET /history           Retrieve historical cyclone records
GET /history/filters   Retrieve available history filter values

For the exact request and response schemas, inspect backend/main.py.
Deployment
The intended hosting approach for this portfolio project is:
- Frontend: Vercel
- Backend: Render
Deployment configuration and public demo links will be added once the
application has been deployed and verified.
Limitations and responsible use
- Predictions are generated by experimental machine-learning models
  and may be inaccurate, especially at longer forecast horizons.
- Historical records do not indicate current cyclone activity.
- This project is not an operational early-warning system.
- Always rely on official meteorological agencies for cyclone
  advisories, warnings, evacuation guidance, and safety decisions.
Author
Krishanu Kundu
- GitHub: @kriskundu06
- Project repository:
  kriskundu06/trakcyclon
If you find this project useful, feel free to explore the code, models,
and evaluation scripts.
