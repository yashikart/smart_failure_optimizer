# 🔧 Hydraulic System Condition Monitoring — Single MLOps Guide

## 1) What This Is
Production-ready MLOps pipeline for hydraulic condition monitoring with:
- Training (optional if you already have models)
- Reusable sklearn pipelines (scaler → SMOTE → classifier)
- Streamlit dashboard for inference
- CI/CD (GitHub Actions) and Docker deployment
- Basic monitoring hooks (Prometheus/Grafana)


## 2) Prerequisites
- Python 3.9+ (you have 3.13 — OK)
- pip (Python package manager)
- Recommended: Git, Docker Desktop (optional for containerized run)

Install Python packages:
```bash
pip install -r requirements.txt
```
If Jupyter is missing (for training):
```bash
pip install notebook
```


## 3) Repository Structure (Minimal Keys)
```
smart_maintenance_optimizer/
├── app.py                       # Streamlit dashboard (uses models/)
├── app_simple.py                # Demo dashboard (no models required)
├── generate_minimal_metadata.py # Create metadata_*.pkl from your models
├── requirements.txt             # Python dependencies
├── Dockerfile                   # App container
├── docker-compose.yml           # App + MLflow + Prometheus + Grafana
├── .github/workflows/ci-cd.yml  # CI/CD pipeline
└── models/                      # Place your .pkl models here
```


## 4) Model Artifacts (Expected Names)
Place your trained models in `models/` with these names:
```
models/
├── best_model_cooler_cond.pkl
├── best_model_valve_cond.pkl
├── best_model_pump_leak.pkl
└── best_model_accumulator_press.pkl
```
Optional but recommended metadata files (app shows metrics & features if present):
```
models/
├── metadata_cooler_cond.pkl
├── metadata_valve_cond.pkl
├── metadata_pump_leak.pkl
└── metadata_accumulator_press.pkl
```
No metadata? Auto-generate minimal ones:
```bash
python generate_minimal_metadata.py
```
This tries to extract feature names from pipelines and writes `metadata_*.pkl`.


## 5) End‑to‑End Pipeline

### 5.1 Data → Features → Models (Training)
- Load raw cycle data (pressures/flows/temperatures/power)
- Feature extraction (statistical summaries per cycle)
- Feature selection: top 15–20 correlated features (per target)
- Pipeline: `StandardScaler → [optional PCA] → SMOTE → Classifier`
- Classifiers compared: RF, GBM, SVM, LR (RF recommended)
- Evaluation: Accuracy + Weighted F1, 5‑fold Stratified CV
- Persist: `joblib.dump(pipeline, models/best_model_<target>.pkl)` + metadata

Training is already implemented inside `hydraulic_.ipynb` cells 36–58 (optional if you use your own models).

### 5.2 Inference (Streamlit)
- Loads all `best_model_*.pkl` + `metadata_*.pkl`
- Expects features listed in metadata; CSV/manual entry accepted
- Returns predicted class + probabilities and confidence
- Visual status (gauges/charts) + simple health heuristic


## 6) Quick Usage Paths

### A) Use Your Pre‑Trained Models (No Retraining)
1. Copy your `.pkl` models into `models/` using the names above
2. If you don’t have metadata files:
   ```bash
   python generate_minimal_metadata.py
   ```
3. Run the dashboard:
   ```bash
   streamlit run app.py
   ```
4. Open `http://localhost:8501` → use Prediction page

### B) Demo (If You Just Want to See the UI)
```bash
streamlit run app_simple.py
```
This uses simulated outputs, no models required.

### C) Train Locally (Optional)
```bash
jupyter notebook hydraulic_.ipynb
# Run cells 36–58 (Feature Engineering, Training, Saving models)
```
Models will be written into `models/` automatically.


## 7) CSV Input Expectations
- Header row with feature names matching the model’s training features
- Rows = cycles to score
- If your metadata has no features, ensure the CSV columns match the model’s expected inputs

Tip: When in doubt, test a single prediction in Python to confirm compatibility.


## 8) CI/CD (GitHub Actions)
- Linting & static checks (Black/isort/Flake8/Pylint)
- Unit tests (pytest + coverage)
- Model validation step (loads .pkl if present)
- Security scan (Trivy)
- Docker image build + push
- Deploy to staging on `develop`, production on `main`

Workflow file: `.github/workflows/ci-cd.yml`


## 9) Docker & Compose (Optional)
Build and run only the app:
```bash
docker build -t hydraulic-app .
docker run -p 8501:8501 -v %cd%/models:/app/models hydraulic-app   # Windows
# or
docker run -p 8501:8501 -v $(pwd)/models:/app/models hydraulic-app  # Linux/Mac
```

Full stack (app + MLflow + Prometheus + Grafana):
```bash
docker-compose up -d
# App:        http://localhost:8501
# MLflow:     http://localhost:5000
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin)
```


## 10) Monitoring Hooks
- Prometheus job scraping Streamlit and API (if present)
- Grafana dashboards (provision paths in `config/grafana` if used)
- Extend to export custom metrics (prediction counts, latency, drift)


## 11) Testing
Run unit tests:
```bash
pytest tests/ -v
pytest tests/ --cov=. --cov-report=html
```


## 12) Common Issues
- "Models not found": Ensure files are under `models/` with expected names
- Feature mismatch: Ensure your CSV columns/features match training features
- Port in use: run `streamlit run app.py --server.port 8502`
- Missing packages: `pip install -r requirements.txt`


## 13) One‑Command Helpers (Windows)
- `START_HERE.bat`: Guides training/demo and starts the right app
- `run_app.bat`: Launches Streamlit quickly


## 14) Operational Tips
- Version models and metadata together (same commit/release)
- Keep a feature list for each target in metadata for reproducibility
- Log predictions (timestamp, target, prediction, confidence) for monitoring
- Set thresholds for “low‑confidence” alerts in production


## 15) TL;DR Quick Start
```bash
# If you already have models
mkdir -p models           # (Windows: mkdir models)
# Copy your .pkl files into models/
python generate_minimal_metadata.py
streamlit run app.py
# Open http://localhost:8501
```

```bash
# If you want to see the UI without models
streamlit run app_simple.py
```

```bash
# If you want to train locally (optional)
pip install notebook
jupyter notebook hydraulic_.ipynb  # run cells 36–58
```

---
© 2025 Hydraulic MLOps — Single‑file guide for setup, pipeline and usage.

