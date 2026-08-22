from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


# ---------------------------------------------------------
# Configuración
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Base_de_datos.csv"

VALID_INCOME_TRENDS = [
    "Decreciente",
    "Estable",
    "Creciente",
]

NUMERIC_FEATURES = [
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "saldo_mora_codeudor",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito",
    "anio_prestamo",
    "mes_prestamo",
    "dia_semana_prestamo",
    "es_fin_semana",
    "ratio_cuota_capital",
    "ratio_otros_prestamos_salario",
    "ratio_saldo_principal_total",
]

NOMINAL_FEATURES = [
    "tipo_credito",
    "tipo_laboral",
]

ORDINAL_FEATURES = [
    "tendencia_ingresos",
]


# ---------------------------------------------------------
# 1. Carga de datos
# ---------------------------------------------------------

def load_data(data_path=DATA_PATH):
    """
    Carga el dataset original del proyecto.

    El archivo utiliza punto y coma (;) como separador.
    """
    df = pd.read_csv(data_path, sep=";")

    return df


# ---------------------------------------------------------
# 2. Reglas de calidad provenientes del EDA
# ---------------------------------------------------------

def apply_quality_rules(df):
    """
    Aplica las reglas de calidad identificadas durante
    el análisis exploratorio del Avance 1.

    No realiza imputación de valores faltantes.
    """
    df = df.copy()

    # Convertir fecha_prestamo al tipo datetime.
    df["fecha_prestamo"] = pd.to_datetime(
        df["fecha_prestamo"],
        format="mixed",
        dayfirst=True,
        errors="coerce",
    )

    # Convertir puntaje de texto con coma decimal a numérico.
    df["puntaje"] = pd.to_numeric(
        df["puntaje"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )

    # Edades mayores a 100 se consideran inválidas.
    df.loc[df["edad_cliente"] > 100, "edad_cliente"] = np.nan

    # Mantener únicamente las categorías válidas
    # de tendencia_ingresos.
    df["tendencia_ingresos"] = df["tendencia_ingresos"].where(
        df["tendencia_ingresos"].isin(VALID_INCOME_TRENDS),
        np.nan,
    )

    return df


# ---------------------------------------------------------
# 3. Ingeniería de características
# ---------------------------------------------------------

def create_features(df):
    """
    Genera características derivadas identificadas como
    potencialmente útiles durante el EDA del Avance 1.

    Las transformaciones de esta función son determinísticas:
    no aprenden parámetros estadísticos del dataset.
    """
    df = df.copy()

    # -----------------------------------------------------
    # Features temporales
    # -----------------------------------------------------

    df["anio_prestamo"] = df["fecha_prestamo"].dt.year
    df["mes_prestamo"] = df["fecha_prestamo"].dt.month
    df["dia_semana_prestamo"] = df["fecha_prestamo"].dt.dayofweek

    # En pandas: lunes = 0 y domingo = 6.
    df["es_fin_semana"] = (
        df["dia_semana_prestamo"] >= 5
    ).astype(int)

    # -----------------------------------------------------
    # Ratios financieros
    # -----------------------------------------------------

    df["ratio_cuota_capital"] = (
        df["cuota_pactada"]
        / df["capital_prestado"].replace(0, np.nan)
    )

    df["ratio_otros_prestamos_salario"] = (
        df["total_otros_prestamos"]
        / df["salario_cliente"].replace(0, np.nan)
    )

    df["ratio_saldo_principal_total"] = (
        df["saldo_principal"]
        / df["saldo_total"].replace(0, np.nan)
    )

    # La fecha original ya fue representada mediante
    # características numéricas más utilizables por los modelos.
    df = df.drop(columns=["fecha_prestamo"])

    return df


# ---------------------------------------------------------
# 4. Preparación de la variable objetivo
# ---------------------------------------------------------

def prepare_target(df):
    """
    Separa las variables predictoras y redefine el evento
    positivo como incumplimiento crediticio.

    riesgo_incumplimiento = 1 -> no pagó a tiempo
    riesgo_incumplimiento = 0 -> pagó a tiempo
    """
    df = df.copy()

    y = 1 - df["Pago_atiempo"]
    y = y.astype(int)
    y.name = "riesgo_incumplimiento"

    X = df.drop(columns=["Pago_atiempo"])

    return X, y


# ---------------------------------------------------------
# 5. División de entrenamiento y evaluación
# ---------------------------------------------------------

def split_data(
    X,
    y,
    test_size=0.20,
    random_state=42,
):
    """
    Divide los datos en conjuntos de entrenamiento y evaluación.

    Se utiliza estratificación para preservar la proporción de
    la clase minoritaria de riesgo en ambos conjuntos.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# 6. Construcción del preprocesador
# ---------------------------------------------------------

def build_preprocessor():
    """
    Construye el pipeline de preprocesamiento según el tipo
    de variable.

    - Numéricas: imputación por mediana + estandarización.
    - Nominales: imputación por moda + One-Hot Encoding.
    - Ordinales: imputación por moda + Ordinal Encoding.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    nominal_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    ordinal_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[VALID_INCOME_TRENDS],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "nominal",
                nominal_pipeline,
                NOMINAL_FEATURES,
            ),
            (
                "ordinal",
                ordinal_pipeline,
                ORDINAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# ---------------------------------------------------------
# 7. Preparación completa de datos para modelado
# ---------------------------------------------------------

def prepare_training_data(
    data_path=DATA_PATH,
    test_size=0.20,
    random_state=42,
):
    """
    Ejecuta el flujo completo de preparación de datos.

    Retorna los conjuntos de entrenamiento y evaluación junto
    con un preprocesador sin ajustar, para que pueda integrarse
    posteriormente dentro de los pipelines de modelado.
    """
    df = load_data(data_path)

    df = apply_quality_rules(df)

    df = create_features(df)

    X, y = prepare_target(df)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = build_preprocessor()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    )
