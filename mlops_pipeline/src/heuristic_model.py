"""
Heuristic baseline for Students Performance.

This script uses simple domain-inspired rules (reading/writing scores +
categorical bonuses/penalties) to approximate the math score. It is
useful as a lightweight benchmark before training ML models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def load_config() -> Dict:
    """Load config.json regardless of the current working directory."""
    config_path = Path("config.json")
    if not config_path.exists():
        config_path = Path(__file__).resolve().parents[2] / "config.json"
    with config_path.open(encoding="utf-8") as cfg_file:
        config = json.load(cfg_file)
    config["__root_path__"] = config_path.parent
    return config


def heuristic_prediction(row: pd.Series) -> float:
    """
    Simple rule-based estimation for math_score.

    - Base: average of reading and writing scores.
    - Bonus if lunch is standard (better nutricional status).
    - Bonus if test preparation course completed.
    - Small uplift for higher parental education.
    """
    base = (row["reading_score"] + row["writing_score"]) / 2

    lunch_bonus = 3 if row["lunch"] == "standard" else -3
    prep_bonus = 5 if row["test_preparation_course"] == "completed" else -2

    parental_map = {
        "master's degree": 4,
        "bachelor's degree": 3,
        "associate's degree": 2,
        "some college": 1,
        "some high school": -1,
        "high school": -2,
    }
    parental_bonus = parental_map.get(row["parental_level_of_education"], 0)

    estimate = base + lunch_bonus + prep_bonus + parental_bonus
    return float(np.clip(estimate, 0, 100))


def evaluate_heuristic(df: pd.DataFrame, target_col: str) -> Dict[str, float]:
    """Compute predictions for the entire dataset and return metrics."""
    preds = df.apply(heuristic_prediction, axis=1)
    y_true = df[target_col].values
    return {
        "r2": r2_score(y_true, preds),
        "mae": mean_absolute_error(y_true, preds),
        "rmse": np.sqrt(mean_squared_error(y_true, preds)),
    }


def main():
    """Run the heuristic baseline against the clean dataset."""
    config = load_config()
    root = Path(config["__root_path__"])
    clean_path = root / config["data"]["clean_dataset"]
    if not clean_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset limpio en {clean_path}. "
            "Ejecuta Cargar_datos.ipynb primero."
        )

    df = pd.read_csv(clean_path)
    target_col = config["features"]["target"]
    metrics = evaluate_heuristic(df, target_col)

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "heuristic_metrics.csv"
    pd.DataFrame([metrics]).to_csv(report_path, index=False, encoding="utf-8")

    print("Evaluación del modelo heurístico:")
    for key, value in metrics.items():
        print(f"{key.upper()}: {value:.4f}")
    print(f"Reporte guardado en {report_path}")


if __name__ == "__main__":
    main()
