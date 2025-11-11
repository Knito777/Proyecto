"""API de inferencia para Students Performance."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field


def _resolve_config_path() -> Path:
    """Ubica config.json tanto en local como en ejecución empaquetada."""
    candidate = Path("config.json")
    if candidate.exists():
        return candidate
    return Path(__file__).resolve().parents[2] / "config.json"


CONFIG_PATH = _resolve_config_path()
with CONFIG_PATH.open(encoding="utf-8") as cfg_file:
    CONFIG = json.load(cfg_file)

ROOT_PATH = CONFIG_PATH.parent
PREPROCESSOR_PATH = ROOT_PATH / CONFIG["paths"]["preprocessor"]
MODEL_PATH = ROOT_PATH / CONFIG["paths"]["best_model"]
FEATURE_ORDER = CONFIG["features"]["categorical"] + CONFIG["features"]["numeric"]


def _to_snake(value: str) -> str:
    """Normaliza nombres heredados desde CSVs externos."""
    cleaned = value.strip().lower()
    for old in [" ", "/", "-", "."]:
        cleaned = cleaned.replace(old, "_")
    return cleaned


class StudentFeatures(BaseModel):
    """Esquema de entrada para una fila."""

    gender: str = Field(..., description="Género del estudiante")
    race_ethnicity: str = Field(..., description="Grupo racial/étnico")
    parental_level_of_education: str = Field(..., description="Nivel educativo")
    lunch: str = Field(..., description="Tipo de almuerzo")
    test_preparation_course: str = Field(..., description="Curso de preparación")
    reading_score: float = Field(..., ge=0, le=100, description="Puntaje de lectura")
    writing_score: float = Field(..., ge=0, le=100, description="Puntaje de escritura")


class PredictionResponse(BaseModel):
    """Respuesta para inferencia simple."""

    pred_math_score: float


class BatchPredictionRequest(BaseModel):
    """Payload para predicciones por lote."""

    items: List[StudentFeatures]


class BatchPredictionResponse(BaseModel):
    """Respuesta para lotes."""

    predictions: List[float]


app = FastAPI(
    title="Students Performance API",
    version="1.0.0",
    description="Servicio de inferencia para el modelo entrenado en math_score.",
)

_PREPROCESSOR = None
_MODEL = None


def _load_artifacts() -> None:
    """Carga perezosa del preprocesador y modelo."""
    global _PREPROCESSOR, _MODEL  # pylint: disable=global-statement
    if _PREPROCESSOR is not None and _MODEL is not None:
        return
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"No se encuentra el preprocesador en {PREPROCESSOR_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encuentra el modelo en {MODEL_PATH}")
    _PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)
    model_artifact = joblib.load(MODEL_PATH)
    if isinstance(model_artifact, dict) and "model" in model_artifact:
        _MODEL = model_artifact["model"]
    else:
        _MODEL = model_artifact


def _predict_records(records: List[StudentFeatures]) -> List[float]:
    """Transforma registros y devuelve predicciones."""
    _load_artifacts()
    payload = pd.DataFrame([row.model_dump() for row in records])
    payload = payload[FEATURE_ORDER]
    transformed = _PREPROCESSOR.transform(payload)
    predictions = _MODEL.predict(transformed)
    return [round(float(value), 4) for value in predictions]


def _records_from_dataframe(df: pd.DataFrame) -> List[StudentFeatures]:
    """Convierte un DataFrame ya validado en objetos StudentFeatures."""
    normalized_cols = {col: _to_snake(col) for col in df.columns}
    df = df.rename(columns=normalized_cols)
    missing = [col for col in FEATURE_ORDER if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Columnas faltantes en el CSV: {missing}. Se esperan {FEATURE_ORDER}.",
        )
    ordered = df[FEATURE_ORDER].copy()
    records = []
    for _, row in ordered.iterrows():
        payload = {feature: row[feature] for feature in FEATURE_ORDER}
        records.append(StudentFeatures(**payload))
    return records


@app.get("/health")
def read_health() -> dict:
    """Endpoint de salud básico."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: StudentFeatures) -> PredictionResponse:
    """Predicción para un solo estudiante."""
    try:
        prediction = _predict_records([features])[0]
    except FileNotFoundError as missing_artifact:
        raise HTTPException(status_code=500, detail=str(missing_artifact)) from missing_artifact
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"Error en la inferencia: {exc}") from exc
    return PredictionResponse(pred_math_score=prediction)


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Predicciones para varios estudiantes."""
    if not request.items:
        raise HTTPException(status_code=400, detail="La lista 'items' no puede estar vacía.")
    try:
        predictions = _predict_records(request.items)
    except FileNotFoundError as missing_artifact:
        raise HTTPException(status_code=500, detail=str(missing_artifact)) from missing_artifact
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"Error en la inferencia: {exc}") from exc
    return BatchPredictionResponse(predictions=predictions)


@app.post("/predict_csv", response_model=BatchPredictionResponse)
async def predict_csv(file: UploadFile = File(...)) -> BatchPredictionResponse:
    """Acepta un CSV con la misma estructura que el dataset limpio."""
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="El archivo recibido está vacío.")
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo leer el CSV proporcionado: {exc}",
        ) from exc
    if df.empty:
        raise HTTPException(status_code=400, detail="El CSV no contiene registros.")
    try:
        records = _records_from_dataframe(df)
        predictions = _predict_records(records)
    except HTTPException:
        raise
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"Error al procesar el CSV: {exc}") from exc
    return BatchPredictionResponse(predictions=predictions)


@app.get("/")
def root() -> dict:
    """Devuelve estado general."""
    return {"status": "ok", "project": CONFIG.get("project_name", "students_performance_mlops")}
