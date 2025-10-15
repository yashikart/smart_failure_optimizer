"""
Train hydraulic models using FULL_DF.CSV and log to MLflow.

Runs one experiment per target (Cooler_Cond, Valve_Cond, Pump_Leak, Accumulator_Press),
logs params/metrics/artifacts/models, and saves model/metadata files compatible with app.py.

Usage:
  python train_full_df_mlflow.py                 # auto-detect FULL_DF.CSV
  python train_full_df_mlflow.py --data ./FULL_DF.CSV
  python train_full_df_mlflow.py --target Cooler_Cond

Environment (defaults work with Docker Compose):
  MLFLOW_TRACKING_URI=http://mlflow:5000
  MLFLOW_EXPERIMENT=hydraulic_condition_monitoring
"""

import argparse
import os
import json
import tempfile
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
import joblib

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

TARGETS = ['Cooler_Cond', 'Valve_Cond', 'Pump_Leak', 'Accumulator_Press']


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train hydraulic models from FULL_DF.CSV and log to MLflow')
    p.add_argument('--data', type=str, default=None, help='Path to FULL_DF.CSV (auto-detects in ./ or ./data/)')
    p.add_argument('--target', type=str, default=None, choices=TARGETS, help='Train only one target')
    p.add_argument('--test-size', type=float, default=0.2, help='Test split fraction')
    p.add_argument('--random-state', type=int, default=42, help='Random seed')
    p.add_argument('--n-estimators', type=int, default=300, help='RandomForest n_estimators')
    p.add_argument('--max-depth', type=int, default=None, help='RandomForest max_depth')
    p.add_argument('--k-best', type=int, default=20, help='Top-K features to select')
    p.add_argument('--use-pca', action='store_true', help='Enable PCA after top-K selection')
    p.add_argument('--pca-components', type=int, default=20, help='Number of PCA components when --use-pca')
    return p.parse_args()


def find_full_df() -> str:
    candidates = [
        'FULL_DF.CSV', 'full_df.csv', 'Full_DF.csv',
        os.path.join('data', 'FULL_DF.CSV'), os.path.join('data', 'full_df.csv'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError('FULL_DF.CSV not found (tried ./ and ./data).')


def prepare_X_y(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in dataset")
    y = df[target]
    X = df.drop(columns=[t for t in TARGETS if t in df.columns]).select_dtypes(include=[np.number]).copy()
    return X, y


def build_pipeline(n_estimators: int, max_depth: int, random_state: int, k_best: int, use_pca: bool, pca_components: int) -> Pipeline:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1,
    )
    steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('kbest', SelectKBest(score_func=mutual_info_classif, k=max(1, k_best))),
    ]
    if use_pca:
        steps.append(('pca', PCA(n_components=pca_components, random_state=random_state)))
    steps.extend([
        ('scaler', StandardScaler(with_mean=False)),
        ('model', model),
    ])
    return Pipeline(steps)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: List, out_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_model_and_metadata(target: str, pipe: Pipeline, features: List[str], acc: float, f1m: float, cv_mean: float) -> None:
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipe, os.path.join('models', f'best_model_{target.lower()}.pkl'))
    meta = {
        'model': 'RandomForest',
        'features': features,
        'accuracy': float(acc),
        'f1_score': float(f1m),
        'cv_mean': float(cv_mean),
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    joblib.dump(meta, os.path.join('models', f'metadata_{target.lower()}.pkl'))


def main():
    args = parse_args()

    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    experiment = os.getenv('MLFLOW_EXPERIMENT', 'hydraulic_condition_monitoring')
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    mlflow.sklearn.autolog(log_models=True)

    data_path = args.data or find_full_df()
    df = pd.read_csv(data_path)

    targets = [args.target] if args.target else TARGETS
    for target in targets:
        X, y = prepare_X_y(df, target)
        features = list(X.columns)

        strat = y if len(pd.unique(y)) > 1 else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=strat
        )

        with mlflow.start_run(run_name=f'train-{target}'):
            mlflow.log_param('target', target)
            mlflow.log_param('num_features', len(features))
            mlflow.log_param('n_estimators', args.n_estimators)
            mlflow.log_param('max_depth', args.max_depth if args.max_depth is not None else -1)
            mlflow.log_param('k_best', args.k_best)
            mlflow.log_param('use_pca', bool(args.use_pca))
            if args.use_pca:
                mlflow.log_param('pca_components', args.pca_components)

            pipe = build_pipeline(
                args.n_estimators,
                args.max_depth,
                args.random_state,
                args.k_best,
                args.use_pca,
                args.pca_components,
            )

            # quick CV
            try:
                cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=3, scoring='accuracy', n_jobs=-1)
                cv_mean = float(np.mean(cv_scores))
                mlflow.log_metric('cv_mean', cv_mean)
            except Exception:
                cv_mean = 0.0

            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)
            acc = float(accuracy_score(y_te, y_pred))
            f1m = float(f1_score(y_te, y_pred, average='macro'))
            mlflow.log_metric('test_accuracy', acc)
            mlflow.log_metric('test_f1_macro', f1m)

            with tempfile.TemporaryDirectory() as tmp:
                labels_sorted = sorted(list(pd.unique(y_te)))
                cm_path = os.path.join(tmp, f'{target}_confusion_matrix.png')
                plot_confusion(y_te.to_numpy(), y_pred, labels_sorted, cm_path)
                mlflow.log_artifact(cm_path, artifact_path=f'evaluation/{target}')

                rpt = classification_report(y_te, y_pred, output_dict=True)
                rpt_path = os.path.join(tmp, f'{target}_classification_report.json')
                with open(rpt_path, 'w', encoding='utf-8') as f:
                    json.dump(rpt, f, indent=2)
                mlflow.log_artifact(rpt_path, artifact_path=f'evaluation/{target}')

            mlflow.sklearn.log_model(pipe, artifact_path=f'model_{target}')
            save_model_and_metadata(target, pipe, features, acc, f1m, cv_mean)

            print(f"{target}: acc={acc:.4f}, f1_macro={f1m:.4f}, cv_mean={cv_mean:.4f}")

    print('Training complete. Check MLflow UI and Streamlit MLflow Dashboard.')


if __name__ == '__main__':
    main()


