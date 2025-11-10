# Proyecto MLOps: Students Performance

Pipeline completo para predecir la nota de matemáticas del dataset **Students Performance in Exams**. El flujo incluye limpieza, análisis exploratorio, ingeniería de características, entrenamiento, despliegue y monitoreo del modelo.

## 1. Preparación del entorno

1. Clona o descarga el repositorio.
2. Ejecuta `set_up.bat` (doble clic o desde PowerShell: `.\set_up.bat`). El script crea/activa `env/` e instala las dependencias de `requirements.txt`.
3. Activa el entorno cuando trabajes manualmente: `env\Scripts\activate`.

## 2. Configuración clave

Toda la parametrización se centraliza en `config.json`:

- `data.raw_dataset`: CSV original (`Base_de_datos.csv`).
- `data.clean_dataset`: salida de datos limpios (`mlops_pipeline/data/clean_students.csv`).
- `features`: definición del target (`math_score`) y columnas numéricas/categóricas.
- `paths`: rutas para reportes, artefactos del modelo y monitoreo.
- `training`: `test_size` y `random_state` utilizados en los scripts.

## 3. Pipeline paso a paso

### 3.1 Limpieza (`mlops_pipeline/src/Cargar_datos.ipynb`)

1. Ejecuta todas las celdas para:
   - Cargar `Base_de_datos.csv`.
   - Normalizar nombres de columnas a snake_case.
   - Revisar nulos/duplicados.
   - Guardar el dataset limpio en `mlops_pipeline/data/clean_students.csv`.

### 3.2 EDA (`mlops_pipeline/src/comprension_eda.ipynb`)

Genera:

- Descriptivos (`describe`, conteo categórico).
- Matriz de correlación + `heatmap_correlaciones.png`.
- Histogramas por variable (`mlops_pipeline/reports/hist_*.png`).
- `boxplot_math_por_genero.png`.

### 3.3 Feature engineering (`mlops_pipeline/src/ft_engineering.py`)

```
python mlops_pipeline/src/ft_engineering.py
```

- Lee la configuración y el CSV limpio.
- Ajusta un `ColumnTransformer` con `OneHotEncoder + StandardScaler`.
- Divide train/test y guarda:
  - `model_store/preprocessor.joblib`
  - `model_store/train_set.pkl` / `model_store/test_set.pkl`

### 3.4 Entrenamiento y evaluación (`mlops_pipeline/src/model_training.ipynb` + `model_evaluation.ipynb`)

1. Ejecuta `model_training.ipynb` para:
   - Cargar los conjuntos preprocesados.
   - Entrenar `LinearRegression`, `DecisionTree`, `RandomForest` y `XGBoost`.
   - Calcular R², MAE y RMSE.
   - Guardar el mejor modelo en `model_store/best_model.pkl`.
   - Exportar las métricas a `reports/model_metrics.csv`.
2. Abre `model_evaluation.ipynb` para validar el modelo seleccionado y generar los gráficos de diagnóstico en `reports/residuals_diagnostics.png`.

### 3.5 Baseline heurístico (`mlops_pipeline/src/heuristic_model.py`)

```
python mlops_pipeline/src/heuristic_model.py
```

- Calcula una predicción basada en reglas simples (lectura/escritura + bonificaciones por lunch y test prep).
- Genera métricas rápidas (R², MAE, RMSE) antes de entrenar modelos complejos.
- Exporta resultados a `reports/heuristic_metrics.csv`.

### 3.6 Despliegue FastAPI (`mlops_pipeline/src/model_deploy.ipynb`)

1. Ejecuta el notebook hasta la celda que define la aplicación.
2. Corre la última celda (`uvicorn.run(...)`) para levantar el servicio local en `http://localhost:8000`.
3. Interrumpe el kernel cuando quieras detener la API.

Endpoints disponibles:

- `GET /health` → estado básico.
- `POST /predict`

Ejemplo de payload:

```json
{
  "gender": "female",
  "race_ethnicity": "group B",
  "parental_level_of_education": "bachelor's degree",
  "lunch": "standard",
  "test_preparation_course": "completed",
  "reading_score": 92,
  "writing_score": 95
}
```

Respuesta: `{"pred_math_score": 89.7}`

### 3.7 Monitoreo de drift (`mlops_pipeline/src/model_monitoring.ipynb`)

1. Ejecuta el notebook para cargar el dataset limpio.
2. Se simula drift alterando la proporción de `test_preparation_course` y el rendimiento asociado.
3. Se calcula el KS test entre distribuciones originales vs. alteradas.
4. Resultados:
   - CSV: `reports/drift_report.csv`.
   - Figura: `reports/drift_math_score.png`.

## 4. Recomendaciones adicionales

- Versiona los artefactos críticos (`config.json`, notebooks, scripts y reportes CSV) para garantizar reproducibilidad.
- Antes de desplegar, vuelve a ejecutar `ft_engineering.py` y `model_training_evaluation.py` si el dataset cambia.
- Automatiza el pipeline con un orquestador (GitHub Actions, Prefect, etc.) para ambientes productivos.
