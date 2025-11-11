# Proyecto MLOps: Students Performance

Pipeline de extremo a extremo para predecir `math_score` en el dataset **Students Performance in Exams**. Incluye limpieza, EDA enriquecido, ingeniería de características, entrenamiento (regresión + clasificación derivada), despliegue FastAPI, monitoreo (notebook + Streamlit), Docker y análisis de calidad con SonarCloud listo para CI.

## Estructura del proyecto

```
.
├─ Base_de_datos.csv
├─ config.json
├─ Dockerfile
├─ STREAMLIT_app.py                # Tablero de monitoreo
├─ mlops_pipeline/
│  ├─ data/                        # clean_students.csv
│  ├─ reports/                     # Figuras/estadísticas EDA
│  └─ src/
│     ├─ Cargar_datos.ipynb
│     ├─ comprension_eda.ipynb
│     ├─ ft_engineering.py
│     ├─ heuristic_model.py
│     ├─ model_training.ipynb
│     ├─ model_training_evaluation.py
│     ├─ model_evaluation.ipynb
│     ├─ model_monitoring.ipynb
│     ├─ model_deploy.ipynb
│     └─ model_deploy.py
├─ model_store/                    # preprocessor.joblib, best_model.pkl, best_classifier.joblib
├─ reports/                        # métricas, drift y diagnósticos
├─ requirements.txt
└─ .github/workflows/sonar.yml
```

## Dataset

- **Fuente**: [Kaggle - Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) (1 000 estudiantes).
- **Variables**:
  - Categóricas: `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`.
  - Numéricas: `reading_score`, `writing_score`.
  - Objetivo: `math_score` (regresión). Para clasificación se deriva `math_passed` (1 si `math_score` ≥ 70).
- **Notas de limpieza**: Se insertó un `record_id` auxiliar para detectar duplicados/nulos y se elimina del dataset final junto con cualquier columna fuera del diccionario de datos.

## Requisitos y entorno

1. Python 3.11 recomendado.
2. Instalar dependencias:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate   # En bash: source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Alternativa rápida: `set_up.bat` crea/activa `env/` y ejecuta `pip install -r requirements.txt`.

## Orden de ejecución (1 → 7)

1. **`mlops_pipeline/src/Cargar_datos.ipynb`**: limpia `Base_de_datos.csv`, normaliza columnas a snake_case, une tokens nulos, convierte tipos, elimina duplicados/`record_id` y persiste `mlops_pipeline/data/clean_students.csv`.
2. **`mlops_pipeline/src/comprension_eda.ipynb`**: EDA completo (describe post-corrección, medidas avanzadas, histogramas/boxplots, countplots, pivotes, pairplot, scatter con `hue`, reglas de validación y features sugeridos). Figuras en `mlops_pipeline/reports/`.
3. **`mlops_pipeline/src/ft_engineering.py`**: ejecuta
   ```powershell
   python mlops_pipeline/src/ft_engineering.py
   ```
   Ajusta `ColumnTransformer` (OneHot + StandardScaler), divide train/test y guarda `model_store/preprocessor.joblib`, `train_set.pkl`, `test_set.pkl`.
4. **`mlops_pipeline/src/heuristic_model.py`**: baseline con reglas (`reports/heuristic_metrics.csv`).
5. **Entrenamiento supervisado**:
   - `mlops_pipeline/src/model_training.ipynb`: entrena regresores (LinearRegression, DecisionTree, RandomForest, XGBoost), registra R²/MAE/RMSE (`reports/model_metrics.csv`) y guarda `model_store/best_model.pkl`.
   - `python mlops_pipeline/src/model_training_evaluation.py`: versión reproducible que reentrena regresores y añade clasificadores (LogisticRegression, RandomForestClf, XGBoostClf), exporta `reports/classification_metrics.csv`, `reports/classification_diagnostics.png` y `model_store/best_classifier.joblib`.
6. **`mlops_pipeline/src/model_evaluation.ipynb`**: gráficos de residuos (`reports/residuals_diagnostics.png`).
7. **`mlops_pipeline/src/model_monitoring.ipynb`**: simula drift, calcula KS-test con umbral (`reports/drift_report.csv`, `reports/drift_math_score.png`). Opcional: `streamlit run STREAMLIT_app.py` para un dashboard interactivo con semáforo y carga de CSVs.

## Métricas y comparación de modelos

### Regresión (`reports/model_metrics.csv`)
| Modelo            | R²    | MAE   | RMSE  |
|-------------------|-------|-------|-------|
| LinearRegression  | 0.880 | 4.215 | 5.394 |
| RandomForest      | 0.854 | 4.586 | 5.952 |
| XGBoost           | 0.849 | 4.676 | 6.060 |
| DecisionTree      | 0.817 | 5.301 | 6.674 |

**Modelo final**: `LinearRegression` (mejor balance entre sesgo y varianza).

### Clasificación derivada (aprobado si `math_score` ≥ 70)
| Modelo               | Accuracy | Precision | Recall | F1    | ROC-AUC |
|----------------------|----------|-----------|--------|-------|---------|
| LogisticRegression   | 0.895    | 0.852     | 0.885  | 0.868 | 0.966   |
| RandomForest (clf)   | 0.850    | 0.816     | 0.795  | 0.805 | 0.944   |
| XGBoost (clf)        | 0.835    | 0.808     | 0.756  | 0.781 | 0.932   |

`reports/classification_diagnostics.png` contiene curvas ROC y la matriz de confusión del mejor clasificador.

## Gráficos clave (`mlops_pipeline/reports/`)
- Histogramas y boxplots (`hist_*.png`, `boxplot_math_por_genero.png`).
- Correlaciones (`heatmap_correlaciones.png`, `pairplot_numeric.png`).
- Distribuciones categóricas (`countplots_categoricas.png`, `pivot_*_target.csv`).
- Relación con el objetivo (`scatter_vs_target.png`).
- Diagnósticos de entrenamiento (`reports/residuals_diagnostics.png`) y clasificación (`reports/classification_diagnostics.png`).
- Monitoreo: `reports/drift_math_score.png`.

## API FastAPI (`mlops_pipeline/src/model_deploy.py`)

1. Levantar local:
   ```powershell
   uvicorn mlops_pipeline.src.model_deploy:app --reload --port 8000
   ```
2. Swagger: <http://127.0.0.1:8000/docs>.
3. `GET /health` → `{"status": "ok"}`.
4. `POST /predict` (1 registro, JSON):
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
   Respuesta → `{"pred_math_score": 80.10}`.
5. `POST /predict_batch`: `{"items": [{...}, {...}]}` → `{"predictions": [...]}`.
6. `POST /predict_csv`: enviar archivo CSV con las mismas columnas (`gender`, ..., `writing_score`). Devuelve `{"predictions": [...]}` para todos los registros cargados.

Todos los endpoints usan el `ColumnTransformer` y `best_model.pkl` guardados en `model_store/` y validan orden `categorical + numeric`.

## Docker (opcional promovido)

```bash
docker build -t students-ml-api .
docker run -p 8000:8000 students-ml-api
```
La imagen ejecuta `uvicorn mlops_pipeline.src.model_deploy:app --host 0.0.0.0 --port 8000`.

## Monitoring y drift

- Notebook: `mlops_pipeline/src/model_monitoring.ipynb` (simulación de drift y KS-test con `alpha=0.05`). Último resultado (`reports/drift_report.csv`): `statistic=0.057`, `pvalue=0.0776`, `drift_detected=False` → estable.
- Tablero: `streamlit run STREAMLIT_app.py`. Permite subir un CSV o usar la simulación integrada, muestra `st.metric` + barra de riesgo y curvas de distribución con Plotly. Se cambia a alerta roja si `pvalue < α`.
- Reportes persistentes: `reports/drift_report.csv`, `reports/drift_math_score.png`.

## Reglas de validación y features sugeridos

- Validar que `math_score`, `reading_score`, `writing_score` ∈ [0, 100].
- `test_preparation_course` ∈ {`none`, `completed`} y `lunch` ∈ {`standard`, `free/reduced`}.
- Alertar si `reading_score` o `writing_score` < 20.
- Features sugeridos: `language_avg`, `language_gap`, codificación ordinal de `parental_level_of_education`, banderas binarias para `test_preparation_course` y `lunch`.

## SonarCloud (automatizado)

1. Se añadió `sonar-project.properties` y el workflow `.github/workflows/sonar.yml`.
2. Crea secretos en GitHub: `SONAR_TOKEN`, `SONAR_PROJECT_KEY`, `SONAR_ORG`.
3. Cada push/PR ejecuta la acción `SonarSource/sonarcloud-github-action@v2` (Python 3.11) calculando *Code Smells*, *Duplications* y *Maintainability*.
4. Ejecución local opcional:
   ```bash
   sonar-scanner \
     -Dsonar.projectKey=$SONAR_PROJECT_KEY \
     -Dsonar.organization=$SONAR_ORG \
     -Dsonar.login=$SONAR_TOKEN
   ```
5. Badge sugerido:
   ```
   [![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=students_performance_mlops&metric=alert_status)](https://sonarcloud.io/dashboard?id=students_performance_mlops)
   ```

## Checklist de la rúbrica

- ✅ **Estructura y configuraciones**: carpetas mínimas respetadas, `requirements.txt` y guías de entorno (PowerShell + bash + `set_up.bat`).
- ✅ **Análisis de datos / EDA**: descripción del dataset, tipos, nulos unidos, eliminación de columnas auxiliares (`record_id`), conversión de tipos, `describe()` post-corrección, histogramas/boxplots, countplots, tablas pivote, estadísticas (media/mediana/moda/rango/IQR/var/std/skew/kurtosis), análisis de distribución, pairplots, correlaciones, relaciones con el target, reglas y features sugeridos.
- ✅ **Ingeniería de características**: `ft_engineering.py` documentado (ColumnTransformer, escalado, OHE, split, artefactos en `model_store/`).
- ✅ **Entrenamiento y evaluación**: múltiples modelos supervisados, `build_model()` reutilizable, `train_test_split`, métricas R²/MAE/RMSE, `summarize_classification()` con accuracy/precision/recall/F1/ROC-AUC, gráficos ROC + matriz de confusión, justificación del modelo final.
- ✅ **Monitoring**: notebook y tablero Streamlit con KS-test, gráficos comparativos e indicadores visuales (semáforo + barra de riesgo) y alertas automáticas.
- ✅ **Deploy**: FastAPI con `/health`, `/predict`, `/predict_batch`, `/predict_csv`; validación Pydantic, respuestas JSON, Dockerfile funcional.
- ✅ **SonarCloud**: configuración lista (workflow + propiedades) con instrucciones para tokens y comando local.

## Créditos y licencia

- Datos: Kaggle - Students Performance in Exams.
- Uso académico. Respeta la licencia del dataset y del repositorio donde se aloje.

---
¿Siguiente paso? Conectar el pipeline a CI/CD completo (lint, pruebas, Sonar, despliegue) y llevar métricas de drift a un tablero corporativo (Grafana/Prometheus) para monitoreo continuo.
