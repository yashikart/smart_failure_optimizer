import os
import joblib
from datetime import datetime

TARGET_NAME_MAP = {
    'cooler_cond': 'Cooler_Cond',
    'valve_cond': 'Valve_Cond',
    'pump_leak': 'Pump_Leak',
    'accumulator_press': 'Accumulator_Press',
}


def get_feature_names_from_pipeline(pipeline):
    # Try to extract feature names from any step that has feature_names_in_
    try:
        if hasattr(pipeline, 'feature_names_in_'):
            return list(pipeline.feature_names_in_)
        if hasattr(pipeline, 'named_steps'):
            for step_name, step in pipeline.named_steps.items():
                if hasattr(step, 'feature_names_in_'):
                    return list(step.feature_names_in_)
    except Exception:
        pass
    return []


def main():
    models_dir = 'models'
    if not os.path.isdir(models_dir):
        print('models/ directory not found. Create it and place your .pkl models inside.')
        return

    files = [f for f in os.listdir(models_dir) if f.startswith('best_model_') and f.endswith('.pkl')]
    if not files:
        print('No model files found. Expected files like best_model_cooler_cond.pkl, etc.')
        return

    for f in files:
        model_path = os.path.join(models_dir, f)
        key = f.replace('best_model_', '').replace('.pkl', '')  # e.g., cooler_cond
        target = TARGET_NAME_MAP.get(key)
        if not target:
            print(f'Skipping {f} (unknown target key: {key})')
            continue

        print(f'Processing {f} -> target {target}')
        try:
            pipeline = joblib.load(model_path)
        except Exception as e:
            print(f'  ERROR loading model: {e}')
            continue

        features = get_feature_names_from_pipeline(pipeline)
        if features:
            print(f'  Extracted {len(features)} feature names from pipeline')
        else:
            print('  WARNING: Could not infer feature names. Using empty list. You may need to provide feature names manually.')

        metadata = {
            'target': target,
            'model': type(getattr(pipeline, 'named_steps', {}).get('classifier', pipeline)).__name__,
            'accuracy': 0.0,
            'f1_score': 0.0,
            'cv_mean': 0.0,
            'cv_std': 0.0,
            'features': features,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        meta_path = os.path.join(models_dir, f'metadata_{key}.pkl')
        try:
            joblib.dump(metadata, meta_path)
            print(f'  Wrote metadata: {meta_path}')
        except Exception as e:
            print(f'  ERROR writing metadata: {e}')

    print('\nDone. If there were warnings about missing features, ensure your input CSV/fields match the model training features in the correct order.')


if __name__ == '__main__':
    main()
