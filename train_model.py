"""Leakage-free training, cross-validation, evaluation and model export."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from features import FEATURES, LABELS, validate_frame

def train(data='data/synthetic_valve_dataset.csv', out='results', source='synthetic'):
    df = pd.read_csv(data)
    validate_frame(df)
    if set(df.label) != set(LABELS) or df.label.value_counts().min() < 10:
        raise ValueError('Need all five classes with at least 10 observations each.')
    meta_path = Path(data).with_suffix('.metadata.json')
    if meta_path.exists() and json.loads(meta_path.read_text())['source'] == 'synthetic':
        if source != 'synthetic':
            raise ValueError('Synthetic metadata cannot be relabeled as measured.')
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df.label, test_size=.2, stratify=df.label, random_state=42)
    model = make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=100, max_features='sqrt', random_state=42, n_jobs=-1))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_val_score(model, X_train, y_train, cv=folds, n_jobs=1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    report = classification_report(y_test, pred, labels=LABELS, output_dict=True, zero_division=0)
    result = Path(out)
    result.mkdir(parents=True, exist_ok=True)
    metrics = {'source': source, 'dataset_sha256': hashlib.sha256(Path(data).read_bytes()).hexdigest(),
               'sklearn_version': sklearn.__version__, 'train_rows': len(X_train),
               'test_rows': len(X_test), 'seed': 42, 'cv_scores': cv.tolist(),
               'cv_mean': float(cv.mean()), 'cv_std': float(cv.std()),
               'test_accuracy': accuracy_score(y_test, pred), 'classification_report': report,
               'limitation': 'Random window split; real repeated runs need grouped evaluation. '
                             'Synthetic scores do not validate industrial performance.'}
    (result/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    joblib.dump({'pipeline': model, 'features': FEATURES, 'source': source,
                 'sklearn_version': sklearn.__version__}, result/'valve_model.joblib')
    importance = pd.Series(model.named_steps['randomforestclassifier'].feature_importances_,
                           index=FEATURES).sort_values()
    importance.rename('importance').to_csv(result/'feature_importance.csv', index_label='feature')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=LABELS,
        xticks_rotation=35, colorbar=False, ax=axes[0], cmap='Blues')
    axes[0].set_title('Held-out confusion matrix')
    importance.plot.barh(ax=axes[1], color='#2784b8')
    axes[1].set_title('Random Forest feature importance')
    fig.suptitle(f'Smart Valve Fault Detection | {source.upper()} DATA | Accuracy {metrics["test_accuracy"]:.2%}')
    fig.tight_layout()
    fig.savefig(result/'model_results.png', dpi=150)
    plt.close(fig)
    text = (f'# Evaluation ({source} data)\n\n'
            f'Train: {len(X_train)}; test: {len(X_test)}. Seed: 42.\n\n'
            f'5-fold CV: {cv.mean():.4f} +/- {cv.std():.4f}. '
            f'Test accuracy: {metrics["test_accuracy"]:.4f}.\n\n'
            '```text\n'+classification_report(y_test, pred, labels=LABELS, zero_division=0)+'```\n\n'
            'Scaling is fitted inside each CV fold. The hold-out set is excluded from CV.\n\n'
            + metrics['limitation']+'\n')
    (result/'evaluation.md').write_text(text, encoding='utf-8')
    print(text)
    return metrics

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/synthetic_valve_dataset.csv')
    p.add_argument('--out', default='results')
    p.add_argument('--source', choices=['synthetic', 'measured'], default='synthetic')
    train(**vars(p.parse_args()))
