"""
Dashboard de monitoreo para Students Performance.

Permite comparar la distribución histórica vs. actual,
calcular KS-test y visualizar un semáforo de riesgo.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import ks_2samp


def load_config() -> tuple[dict, Path]:
    config_path = Path("config.json")
    if not config_path.exists():
        raise FileNotFoundError("No se encontró config.json en la raíz del proyecto.")
    with config_path.open(encoding="utf-8") as cfg_file:
        config = json.load(cfg_file)
    return config, config_path.parent


def load_baseline(root: Path, config: dict) -> pd.DataFrame:
    clean_path = root / config["data"]["clean_dataset"]
    return pd.read_csv(clean_path)


def simulate_drift(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    drift_df = df.copy()
    completed_mask = drift_df["test_preparation_course"].eq("completed")
    if completed_mask.any():
        subset = drift_df[completed_mask].sample(frac=0.7, random_state=42)
        drift_df.loc[subset.index, "test_preparation_course"] = "none"
        drift_df.loc[subset.index, target] = (
            drift_df.loc[subset.index, target]
            - rng.normal(loc=7, scale=3, size=len(subset))
        ).clip(lower=0, upper=100)
    return drift_df


def compute_ks(base: pd.Series, sample: pd.Series) -> tuple[float, float]:
    statistic, pvalue = ks_2samp(base, sample)
    return float(statistic), float(pvalue)


def plot_distributions(base: pd.Series, sample: pd.Series, target: str):
    df_plot = pd.DataFrame(
        {
            target: np.concatenate([base.values, sample.values]),
            "dataset": ["Histórico"] * len(base) + ["Actual"] * len(sample),
        }
    )
    fig = px.histogram(
        df_plot,
        x=target,
        color="dataset",
        barmode="overlay",
        nbins=30,
        opacity=0.65,
        color_discrete_sequence=["#1f77b4", "#ff7f0e"],
    )
    fig.update_layout(title=f"Distribución de {target}", bargap=0.1)
    return fig


def main():
    st.set_page_config(page_title="Monitoreo Students Performance", layout="wide")
    st.title("📊 Monitoreo y Drift del modelo")
    st.write(
        "Compara el comportamiento histórico del dataset con nuevos lotes. "
        "El tablero calcula KS-test y muestra alertas visuales."
    )

    config, root = load_config()
    target = config["features"]["target"]
    baseline_df = load_baseline(root, config)

    st.sidebar.header("Configuración")
    alpha = st.sidebar.slider("Umbral α (significancia)", 0.01, 0.2, 0.05, 0.01)
    source_mode = st.sidebar.radio(
        "Fuente de datos actual",
        options=["Simulación de drift", "Subir CSV"],
        help="El CSV debe tener las mismas columnas que el dataset limpio.",
    )
    current_df = None
    if source_mode == "Subir CSV":
        uploaded = st.sidebar.file_uploader("Carga un CSV", type=["csv"])
        if uploaded:
            current_df = pd.read_csv(uploaded)
            current_df.columns = [col.strip().lower().replace(" ", "_") for col in current_df.columns]
    if current_df is None:
        current_df = simulate_drift(baseline_df, target)
    elif target not in current_df.columns:
        st.error(f"El CSV debe contener la columna '{target}'.")
        st.stop()

    statistic, pvalue = compute_ks(baseline_df[target], current_df[target])
    drift_detected = pvalue < alpha

    left, right = st.columns(2)
    left.metric("KS statistic", f"{statistic:.3f}")
    right.metric("KS p-value", f"{pvalue:.4f}", help="Valores < α indican drift.")

    if drift_detected:
        st.error("⚠️ Drift detectado: p-value < α. Considera reentrenar el modelo.")
    else:
        st.success("✅ Distribución estable según KS-test.")

    progress_value = min(statistic / 0.2, 1.0)
    st.progress(progress_value, text="Índice de riesgo (0=estable / 1=alto)")

    fig = plot_distributions(baseline_df[target], current_df[target], target)
    st.plotly_chart(fig, use_container_width=True)

    report_path = root / config["paths"]["drift_report"]
    if report_path.exists():
        st.subheader("Histórico de monitoreo")
        drift_report = pd.read_csv(report_path)
        st.dataframe(drift_report)

    st.sidebar.write("---")
    st.sidebar.caption(
        "Ejecuta `streamlit run STREAMLIT_app.py` para lanzar este tablero. "
        "Los datos simulados replican el proceso del notebook de monitoreo."
    )


if __name__ == "__main__":
    main()
