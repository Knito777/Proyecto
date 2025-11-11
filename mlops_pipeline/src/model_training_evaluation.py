"""
Entrenamiento y evaluación integral de modelos para Students Performance.

Incluye:
- Modelos de regresión (métricas R², MAE, RMSE) compatibles con el pipeline original.
- Clasificadores derivados (aprobado vs. no aprobado) con métricas de clasificación
  y gráficos ROC/matriz de confusión.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

sns.set_theme(style="whitegrid")


def load_config() -> Dict:
    """Carga config.json y devuelve rutas absolutas."""
    config_path = Path("config.json")
    if not config_path.exists():
        config_path = Path(__file__).resolve().parents[2] / "config.json"
    with config_path.open(encoding="utf-8") as cfg_file:
        config = json.load(cfg_file)
    config["__root_path__"] = config_path.parent
    return config


def load_datasets(config: Dict) -> Tuple[Dict, Dict]:
    """Recupera los conjuntos procesados desde model_store."""
    root = Path(config["__root_path__"])
    train_payload = joblib.load(root / config["paths"]["train_set"])
    test_payload = joblib.load(root / config["paths"]["test_set"])
    return train_payload, test_payload


def build_model(task: str, random_state: int):
    """Devuelve el diccionario de modelos según la tarea."""
    if task == "regression":
        return {
            "linear_regression": LinearRegression(),
            "decision_tree": DecisionTreeRegressor(
                max_depth=8,
                min_samples_split=4,
                random_state=random_state,
            ),
            "random_forest": RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1,
            ),
            "xgboost": XGBRegressor(
                objective="reg:squarederror",
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                reg_lambda=1.0,
            ),
        }
    if task == "classification":
        return {
            "logistic_regression": LogisticRegression(
                max_iter=1000, class_weight="balanced"
            ),
            "random_forest_clf": RandomForestClassifier(
                n_estimators=400,
                max_depth=12,
                random_state=random_state,
                class_weight="balanced",
                n_jobs=-1,
            ),
            "xgboost_clf": XGBClassifier(
                objective="binary:logistic",
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                reg_lambda=1.0,
                use_label_encoder=False,
                eval_metric="logloss",
            ),
        }
    raise ValueError(f"Tarea no soportada: {task}")


def summarize_regression(
    models: Dict, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> pd.DataFrame:
    """Entrena y evalúa modelos de regresión."""
    results = []
    best_model = None
    best_score = float("-inf")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "model": name,
            "r2": r2_score(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
        }
        results.append(metrics)
        if metrics["r2"] > best_score:
            best_score = metrics["r2"]
            best_model = model
    return pd.DataFrame(results).sort_values(by="r2", ascending=False), best_model


def summarize_classification(
    models: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[pd.DataFrame, str, Dict]:
    """Entrena clasificadores y devuelve métricas + info para visualizaciones."""
    records: List[Dict] = []
    roc_payload: Dict[str, Dict[str, np.ndarray]] = {}
    best_name = None
    best_f1 = float("-inf")
    best_model = None
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        records.append(metrics)
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_payload[name] = {"fpr": fpr, "tpr": tpr}
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_name = name
            best_model = model
    metrics_df = pd.DataFrame(records).sort_values(by="f1", ascending=False)
    return metrics_df, best_name, {"best_model": best_model, "roc": roc_payload}


def plot_classification_diagnostics(
    roc_payload: Dict[str, Dict[str, np.ndarray]],
    best_model_name: str,
    best_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: Path,
) -> None:
    """Genera curvas ROC y matriz de confusión para el mejor clasificador."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name, info in roc_payload.items():
        axes[0].plot(info["fpr"], info["tpr"], label=f"{name}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[0].set_title("Curvas ROC")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()

    preds = best_model.predict(X_test)
    matrix = confusion_matrix(y_test, preds)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar=False,
        ax=axes[1],
        xticklabels=["No aprueba", "Aprueba"],
        yticklabels=["No aprueba", "Aprueba"],
    )
    axes[1].set_title(f"Matriz de confusión ({best_model_name})")
    axes[1].set_xlabel("Predicción")
    axes[1].set_ylabel("Valor real")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    """Ejecuta el entrenamiento completo y persiste artefactos/registros."""
    config = load_config()
    train_payload, test_payload = load_datasets(config)
    X_train = train_payload["X"]
    y_train = train_payload["y"]
    X_test = test_payload["X"]
    y_test = test_payload["y"]

    random_state = config["training"]["random_state"]

    # --- Regressors (en línea con el notebook original) ---
    reg_models = build_model("regression", random_state)
    regression_results, best_regressor = summarize_regression(
        reg_models, X_train, y_train, X_test, y_test
    )
    print("Resultados de regresión:")
    print(regression_results)

    # --- Clasificadores (aprobado si math_score >= 70) ---
    pass_threshold = 70
    y_train_cls = (y_train >= pass_threshold).astype(int)
    y_test_cls = (y_test >= pass_threshold).astype(int)
    cls_models = build_model("classification", random_state)
    cls_metrics, best_cls_name, payload = summarize_classification(
        cls_models, X_train, y_train_cls, X_test, y_test_cls
    )
    print("\nResultados de clasificación:")
    print(cls_metrics)

    # Guardar artefactos
    root = Path(config["__root_path__"])
    regression_report = root / config["paths"]["metrics_report"]
    regression_results.to_csv(regression_report, index=False, encoding="utf-8")

    cls_report = root / "reports" / "classification_metrics.csv"
    cls_metrics.to_csv(cls_report, index=False, encoding="utf-8")

    # Persistir mejor regresor y clasificador
    model_store = root / config["paths"]["model_store"]
    model_store.mkdir(parents=True, exist_ok=True)
    best_reg_path = root / config["paths"]["best_model"]
    joblib.dump(
        {
            "model_name": regression_results.iloc[0]["model"],
            "model": best_regressor,
            "metrics": regression_results.iloc[0].to_dict(),
            "feature_names": train_payload.get("feature_names"),
        },
        best_reg_path,
    )
    print(f"Regresor principal actualizado en {best_reg_path}")

    best_cls_path = model_store / "best_classifier.joblib"
    joblib.dump(
        {
            "model_name": best_cls_name,
            "model": payload["best_model"],
            "threshold": pass_threshold,
            "metrics": cls_metrics.iloc[0].to_dict(),
        },
        best_cls_path,
    )
    print(f"Clasificador principal guardado en {best_cls_path}")

    diagnostics_path = root / "reports" / "classification_diagnostics.png"
    plot_classification_diagnostics(
        payload["roc"], best_cls_name, payload["best_model"], X_test, y_test_cls, diagnostics_path
    )
    print(f"Gráficos de clasificación en {diagnostics_path}")


if __name__ == "__main__":
    main()

