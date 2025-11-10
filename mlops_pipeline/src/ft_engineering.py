"""
MÃ³dulo de ingenierÃ­a de caracterÃ­sticas para Students Performance.

Lee la configuraciÃ³n global, procesa el dataset limpio con codificaciÃ³n
One-Hot y escalado estÃ¡ndar, realiza la separaciÃ³n train/test y guarda
los artefactos resultantes en el directorio model_store.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_config() -> Dict:
    """Carga archivo config.json detectando la ruta automÃ¡ticamente."""
    config_path = Path("config.json")
    if not config_path.exists():
        config_path = Path(__file__).resolve().parents[2] / "config.json"
    with config_path.open(encoding="utf-8") as cfg_file:
        config = json.load(cfg_file)
    config["__root_path__"] = config_path.parent
    return config


def build_preprocessor(cat_features, num_features) -> ColumnTransformer:
    """Construye el preprocesador combinado para variables categÃ³ricas y numÃ©ricas."""
    try:
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
    except TypeError:
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore", sparse=False
        )
    numeric_transformer = StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, cat_features),
            ("numeric", numeric_transformer, num_features),
        ],
        remainder="drop",
    )
    return preprocessor


def save_dataset_parts(
    train_data: Tuple, test_data: Tuple, feature_names, config: Dict
) -> None:
    """Persiste los conjuntos transformados junto con las etiquetas."""
    store_path = Path(config["__root_path__"]) / config["paths"]["model_store"]
    store_path.mkdir(parents=True, exist_ok=True)

    payload_train = {
        "X": train_data[0],
        "y": train_data[1],
        "feature_names": feature_names,
    }
    payload_test = {
        "X": test_data[0],
        "y": test_data[1],
        "feature_names": feature_names,
    }

    joblib.dump(payload_train, store_path / Path(config["paths"]["train_set"]).name)
    joblib.dump(payload_test, store_path / Path(config["paths"]["test_set"]).name)


def main():
    try:
        config = load_config()
        root = Path(config["__root_path__"])
        clean_path = root / config["data"]["clean_dataset"]
        if not clean_path.exists():
            raise FileNotFoundError(f"No se encuentra el dataset limpio en {clean_path}")

        df = pd.read_csv(clean_path)
        categorical_features = config["features"]["categorical"]
        numeric_features = config["features"]["numeric"]
        target_col = config["features"]["target"]

        X = df[categorical_features + numeric_features]
        y = df[target_col]

        print(f"Total de registros: {df.shape[0]} / Variables: {X.shape[1]}")
        preprocessor = build_preprocessor(categorical_features, numeric_features)

        test_size = config["training"]["test_size"]
        random_state = config["training"]["random_state"]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

        print("Ajustando preprocesador...")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        feature_names = getattr(preprocessor, "get_feature_names_out", lambda: None)()

        preprocessor_path = root / config["paths"]["preprocessor"]
        preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, preprocessor_path)
        print(f"Preprocesador guardado en {preprocessor_path}")

        save_dataset_parts(
            (X_train_processed, y_train),
            (X_test_processed, y_test),
            feature_names,
            config,
        )
        print("Conjuntos train/test preprocesados almacenados correctamente.")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error durante la ingenierÃ­a de caracterÃ­sticas: {exc}")
        raise


if __name__ == "__main__":
    main()

