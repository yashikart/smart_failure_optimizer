"""
new_app.py — Clean minimalist MLOps UI (multimodal + MLflow + tuning)
- Simple, professional layout (no emojis, minimal styling)
- Multimodal inputs: CSV/JSON + optional attachments (logged to MLflow)
- Inference with models in ./models and metadata feature alignment
- Tuning with GridSearchCV + SMOTE (RF/SVM/LR)
- Optional MLflow tracking for inference and tuning
"""

import io
import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# MLflow (optional)
try:
	import mlflow
	import mlflow.sklearn  # noqa
	MLFLOW = True
except Exception:
	MLFLOW = False

st.set_page_config(page_title="Hydraulic MLOps", layout="wide")

MODELS_DIR = "models"
TARGETS = ["Cooler_Cond", "Valve_Cond", "Pump_Leak", "Accumulator_Press"]


def load_artifacts():
	pipelines, meta, feats = {}, {}, {}
	for t in TARGETS:
		k = t.lower()
		mp = os.path.join(MODELS_DIR, f"best_model_{k}.pkl")
		md = os.path.join(MODELS_DIR, f"metadata_{k}.pkl")
		if os.path.exists(mp):
			pipelines[t] = joblib.load(mp)
			meta[t] = joblib.load(md) if os.path.exists(md) else {}
			feats[t] = meta[t].get("features", [])
	return pipelines, meta, feats


@st.cache_resource
def get_artifacts():
	return load_artifacts()


def build_pipeline(alg: str):
	clf = {
		"rf": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
		"svm": SVC(kernel="rbf", probability=True, random_state=42),
		"lr": LogisticRegression(max_iter=2000, random_state=42),
	}[alg]
	return ImbPipeline([
		("scaler", StandardScaler()),
		("smote", SMOTE(random_state=42, k_neighbors=3)),
		("classifier", clf),
	])


def grid_for(alg: str):
	if alg == "rf":
		return {"classifier__n_estimators": [200, 400], "classifier__max_depth": [None, 10, 20]}
	if alg == "svm":
		return {"classifier__C": [0.5, 1, 2], "classifier__gamma": ["scale", 0.01]}
	if alg == "lr":
		return {"classifier__C": [0.5, 1.0, 2.0]}
	return {}


def mlf_log_inference(df: pd.DataFrame, preds: dict, attachments: dict | None = None):
	if not MLFLOW:
		return
	with mlflow.start_run(run_name="inference"):
		buf = io.StringIO()
		df.to_csv(buf, index=False)
		buf.seek(0)
		mlflow.log_text(buf.getvalue(), "input.csv")
		mlflow.log_text(json.dumps(preds, indent=2), "predictions.json")
		if attachments:
			for name, data in attachments.items():
				mlflow.log_bytes(data, f"attachments/{name}")


def mlf_log_training(params: dict, metrics: dict, model):
	if not MLFLOW:
		return
	with mlflow.start_run(run_name="tuning"):
		for k, v in params.items():
			mlflow.log_param(k, v)
		for k, v in metrics.items():
			mlflow.log_metric(k, v)
		mlflow.sklearn.log_model(model, "model")


# Sidebar
page = st.sidebar.selectbox("Navigation", ["Overview", "Inference", "Tuning", "Experiments"]) 

pipelines, metadata, features = get_artifacts()


if page == "Overview":
	st.header("Hydraulic MLOps — Overview")
	st.caption("Interactive inference and tuning with reproducible pipelines")
	c1, c2, c3 = st.columns(3)
	with c1:
		st.metric("Models", len(pipelines))
	with c2:
		st.metric("MLflow", "Enabled" if MLFLOW else "Disabled")
	with c3:
		st.metric("Targets", len(TARGETS))
	st.divider()
	st.subheader("About")
	st.write("This app reads models from the models directory and aligns inputs using stored feature lists. Training is optional if you already have .pkl models.")
	st.write("Use Tuning to run cross-validated grid search with SMOTE and optionally log results to MLflow.")

elif page == "Inference":
	st.header("Inference")
	st.caption("Batch CSV or single JSON; optional attachments can be logged to MLflow")
	c1, c2 = st.columns([2, 1])
	with c1:
		method = st.radio("Input method", ["CSV", "JSON"], horizontal=True)
		X = None
		if method == "CSV":
			up = st.file_uploader("Upload CSV", type=["csv"])
			if up is not None:
				X = pd.read_csv(up)
				st.dataframe(X.head(), use_container_width=True)
		else:
			row = st.text_area("Enter JSON row", value="{}")
			try:
				X = pd.DataFrame([json.loads(row)])
			except Exception:
				st.info("Invalid JSON")
				X = None
		# Optional additional files
		attach = st.file_uploader("Optional attachments (any type)", accept_multiple_files=True)
	with c2:
		sel = st.multiselect("Targets", TARGETS, default=TARGETS)
		log_inf = st.checkbox("Log inference to MLflow", value=False, disabled=not MLFLOW)
		run = st.button("Run Inference")
	if run and X is not None and len(X) > 0:
		out = {}
		for t in sel:
			if t in pipelines:
				Xi = X.copy()
				need = features.get(t, [])
				if need:
					for f in need:
						if f not in Xi.columns:
							Xi[f] = 0.0
				Xi = Xi[need] if need else Xi
				pred = pipelines[t].predict(Xi)
				prob = None
				try:
					prob = pipelines[t].predict_proba(Xi)
				except Exception:
					pass
				out[t] = {"prediction": pred.tolist(), "probabilities": prob.tolist() if prob is not None else None}
		st.subheader("Predictions")
		st.json(out)
		if log_inf and MLFLOW:
			attachments = None
			if attach:
				attachments = {f.name: f.read() for f in attach}
			mlf_log_inference(X, out, attachments)

elif page == "Tuning":
	st.header("Tuning")
	st.caption("Grid search with cross-validation and SMOTE")
	c1, c2 = st.columns([2, 1])
	with c1:
		csv = st.file_uploader("Upload labeled CSV", type=["csv"])
		alg = st.selectbox("Algorithm", ["rf", "svm", "lr"], index=0)
		target = st.text_input("Target column", value="Cooler_Cond")
		cv = st.slider("CV splits", 3, 10, 5)
	with c2:
		log_tr = st.checkbox("Log training to MLflow", value=False, disabled=not MLFLOW)
		btn = st.button("Run Tuning")
	if btn and csv is not None:
		df = pd.read_csv(csv)
		if target not in df.columns:
			st.warning("Target column not found")
			st.stop()
		X = df.drop(columns=[target])
		y = df[target]
		pipe = build_pipeline(alg)
		param_grid = grid_for(alg)
		skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
		search = GridSearchCV(pipe, param_grid=param_grid, scoring="f1_weighted", cv=skf, n_jobs=-1)
		start = time.time()
		search.fit(X, y)
		elapsed = round(time.time() - start, 2)
		best = search.best_estimator_
		y_pred = best.predict(X)
		acc = accuracy_score(y, y_pred)
		f1 = f1_score(y, y_pred, average="weighted")
		st.subheader("Results")
		st.write("Best parameters")
		st.json(search.best_params_)
		st.write("Metrics")
		st.write({"accuracy": acc, "f1_weighted": f1, "elapsed_s": elapsed})
		st.text("Classification report")
		st.code(classification_report(y, y_pred))
		if log_tr and MLFLOW:
			mlf_log_training({"algorithm": alg, **search.best_params_}, {"accuracy": acc, "f1_weighted": f1}, best)
		buf = io.BytesIO()
		joblib.dump(best, buf)
		buf.seek(0)
		st.download_button("Download tuned model", buf, file_name=f"tuned_{alg}.pkl", mime="application/octet-stream")


elif page == "Experiments":
	st.header("Experiments")
	if MLFLOW:
		uri = st.text_input("MLflow Tracking URI", value=os.environ.get("MLFLOW_TRACKING_URI", ""))
		if uri:
			mlflow.set_tracking_uri(uri)
		st.write("Enable logging on other tabs to record runs to this tracking server.")
	else:
		st.write("MLflow is not installed. Install mlflow to enable experiment tracking.")
	st.divider()
