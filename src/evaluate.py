import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    try:
        return model.predict(X, verbose=0).flatten()
    except TypeError:
        return model.predict_proba(X)[:, 1]


def predict_class(model, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (predict_proba(model, X) >= threshold).astype(int)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        'Acuracia': round(accuracy_score(y_true, y_pred), 4),
        'Precisao': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1-Score': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'ROC-AUC': round(roc_auc_score(y_true, y_proba), 4),
        'PR-AUC': round(average_precision_score(y_true, y_proba), 4),
    }


def evaluate_all(models: dict, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    results = {}
    for name, model in models.items():
        y_proba = predict_proba(model, X_test)
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = compute_metrics(y_test, y_pred, y_proba)
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'roc': (fpr, tpr),
            'thresholds': thresholds,
            'y_proba': y_proba,
            'y_pred': y_pred,
        }
    return results


def cross_validate_sklearn(model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    acc_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    return {
        'auc_mean': round(float(auc_scores.mean()), 4),
        'auc_std': round(float(auc_scores.std()), 4),
        'acc_mean': round(float(acc_scores.mean()), 4),
        'acc_std': round(float(acc_scores.std()), 4),
        'auc_scores': auc_scores.tolist(),
    }


def compute_shap_values(model, model_name: str, X_background: np.ndarray, X_patient: np.ndarray):
    try:
        import shap

        if 'Logistica' in model_name or 'Logistic' in model_name:
            explainer = shap.LinearExplainer(model, X_background)
            vals = explainer.shap_values(X_patient)

        elif 'Forest' in model_name or 'Boosting' in model_name or 'Gradient' in model_name:
            explainer = shap.TreeExplainer(model)
            vals = explainer.shap_values(X_patient)
            if isinstance(vals, list) and len(vals) == 2:
                vals = vals[1]

        elif 'Keras' in model_name:
            bg = X_background[:50]
            explainer = shap.GradientExplainer(model, bg)
            vals = explainer.shap_values(X_patient)
            if isinstance(vals, list):
                vals = vals[0]

        else:
            bg = shap.sample(X_background, 60)
            explainer = shap.KernelExplainer(model.predict_proba, bg)
            vals = explainer.shap_values(X_patient, nsamples=150)
            if isinstance(vals, list):
                vals = vals[1]

        return np.array(vals).flatten()

    except Exception:
        return None
