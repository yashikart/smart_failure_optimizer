# Hydraulic System Condition Monitoring — MLOps Platform

A production-ready, modular MLOps setup for condition monitoring of hydraulic systems using Streamlit, scikit-learn pipelines, optional MLflow tracking, and Docker. Designed to run models reliably, repeatably, and at scale.

## Key Capabilities
- Reproducible training/inference pipelines (StandardScaler → SMOTE → Classifier)
- Model versioning via `models/` directory (pretrained `.pkl` + `metadata_*.pkl`)
- Interactive operations UI (Streamlit) for inference and tuning
- Optional experiment tracking with MLflow (params/metrics/artifacts)
- Dockerized runtime; CI/CD workflow scaffold
- Monitoring scaffold (Prometheus/Grafana config placeholders)

## Repository Structure
```
smart_maintenance_optimizer/
├── app.py                    # Full dashboard (inference, analysis & ops)
├── new_app.py                # Clean minimalist MLOps UI (recommended)
├── app_simple.py             # Demo-only app (no models required)
├── generate_minimal_metadata.py  # Generate metadata_*.pkl from models
├── models/                   # Place production models and metadata here
│   ├── best_model_cooler_cond.pkl
│   ├── best_model_valve_cond.pkl
│   ├── best_model_pump_leak.pkl
│   ├── best_model_accumulator_press.pkl
│   ├── metadata_cooler_cond.pkl
│   └── ...
├── requirements.txt          # Python dependencies
├── Dockerfile                # Build selectable Streamlit entry via APP_FILE
├── docker-compose.yml        # Optional full stack including MLflow
├── config/
│   └── prometheus.yml        # Monitoring scaffold (extend as needed)
├── tests/
│   └── test_models.py        # Basic model/pipeline tests (extend)
└── MLOPS_SINGLE_GUIDE.md     # Single-file operations and pipeline guide
```

## Quick Start (Local)
1. Python environment
```
pip install -r requirements.txt
```
2. Models and metadata
- Copy your `.pkl` models into `models/` with these names:
  - `best_model_cooler_cond.pkl`, `best_model_valve_cond.pkl`, `best_model_pump_leak.pkl`, `best_model_accumulator_press.pkl`
- If no metadata: generate minimal files
```
python generate_minimal_metadata.py
```
3. Run the modern UI
```
streamlit run new_app.py
```
Open http://localhost:8501

## MLflow (Optional)
- Install: `pip install mlflow` (use Python 3.9–3.11 for best compatibility)
- Start server (PowerShell):
```
python -m mlflow server \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --default-artifact-root ./mlflow-artifacts \
  --host 0.0.0.0 --port 5000
```
- In `new_app.py` → Experiments page, set Tracking URI: `http://localhost:5000`
- Enable logging toggles on Inference/Tuning to record runs

## Docker
Build and run app-only image:
```
docker build -t hydraulic-app .
# Run new_app.py
docker run -p 8501:8501 -e APP_FILE=new_app.py -v %cd%/models:/app/models hydraulic-app
```
Open http://localhost:8501

Full stack (MLflow service):
```
docker-compose up -d mlflow
# MLflow at http://localhost:5000
```

## CI/CD
- A GitHub Actions workflow (scaffold) performs code quality checks, tests, security scan, builds Docker image, and supports staged deployment.
- Use `APP_FILE` env in your deployment to select entry (e.g., `new_app.py`).

## Streamlit UIs
- `new_app.py` (recommended):
  - Overview: status and metrics
  - Inference: CSV/JSON inputs; optional MLflow logging; attachments support
  - Tuning: GridSearchCV + SMOTE for RF/SVM/LR; CV control; MLflow logging; downloadable model
  - Experiments: configure MLflow tracking URI
- `app.py`: richer operations dashboard
- `app_simple.py`: demo mode without models

## Troubleshooting
- MLflow not detected in app:
  - Ensure `pip install mlflow` in the same environment used to run Streamlit
  - Restart Streamlit after installation
- MLflow CLI not found:
  - Use module: `python -m mlflow server ...` or add `...\Python\Scripts` to PATH
- `models/` empty or missing files:
  - Place the four `best_model_*.pkl` files; run `generate_minimal_metadata.py` if needed
- Feature mismatch during inference:
  - Ensure input columns match the `features` list in corresponding `metadata_*.pkl`
- Port conflicts:
  - Run Streamlit on another port: `streamlit run new_app.py --server.port 8502`

## Known Issues / Future Work
- Add model registry integration (MLflow Model Registry) for promotion flows
- Add drift detection and automated retraining triggers
- Extend test coverage (end-to-end and performance tests)
- Harden security scan and SAST/DAST stages in CI
- Optional: add FastAPI endpoint for programmatic inference

## How to Push This Repository to GitHub
Target repo: `https://github.com/yashikart/smart_maintenance_and_failure_optimizer_of_Hydraulic-Systems.git`

PowerShell commands:
```
# from project root
git init
git add .
git commit -m "MLOps setup: Streamlit UIs, pipelines, MLflow hooks, Docker, CI scaffold"

# default branch
git branch -M main

# set remote (HTTPS)
git remote add origin https://github.com/yashikart/smart_maintenance_and_failure_optimizer_of_Hydraulic-Systems.git

# push (will prompt for auth on first push)
git push -u origin main
```
If you prefer a Personal Access Token (PAT) inline:
```
# Not recommended to keep in shell history; use once
# Replace <TOKEN> with your GitHub PAT, <USER> with your username
git remote set-url origin https://<USER>:<TOKEN>@github.com/yashikart/smart_maintenance_and_failure_optimizer_of_Hydraulic-Systems.git
git push -u origin main
```
Or via GitHub CLI (if installed):
```
# gh auth login (follow prompts)
# then
git push -u origin main
```

## References
- MLflow documentation: https://mlflow.org
- scikit-learn pipelines: https://scikit-learn.org/stable/modules/compose.html
- Imbalanced-learn (SMOTE): https://imbalanced-learn.org
