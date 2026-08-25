from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from src.ft_engineering import (
    apply_quality_rules,
    create_features,
    prepare_training_data,
)

from src.model_training_evaluation import (
    build_model_pipeline,
    build_model_preprocessor,
    build_models,
)


DEPLOYMENT_MODEL_NAME = "Gradient Boosting"

INPUT_FEATURES = [
    "fecha_prestamo",
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
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
    "tipo_credito",
    "tipo_laboral",
    "tendencia_ingresos",
]

# ---------------------------------------------------------
# Esquema de entrada para un registro
# ---------------------------------------------------------

class PredictionInput(BaseModel):
    """
    Representa un registro individual recibido por la API para generar una predicción de riesgo.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    fecha_prestamo: date | None
    capital_prestado: float | None
    plazo_meses: int | None
    edad_cliente: float | None
    salario_cliente: float | None
    total_otros_prestamos: float | None
    cuota_pactada: float | None
    puntaje_datacredito: float | None
    cant_creditosvigentes: int | None
    huella_consulta: float | None
    saldo_mora: float | None
    saldo_total: float | None
    saldo_principal: float | None
    saldo_mora_codeudor: float | None
    creditos_sectorFinanciero: float | None
    creditos_sectorCooperativo: float | None
    creditos_sectorReal: float | None
    promedio_ingresos_datacredito: float | None

    tipo_credito: int | None
    tipo_laboral: str | None

    tendencia_ingresos: (
        Literal[
            "Decreciente",
            "Estable",
            "Creciente",
        ]
        | None
    )

# ---------------------------------------------------------
# Esquema de solicitud para predicción por lotes
# ---------------------------------------------------------

class PredictionRequest(BaseModel):
    """
    Representa una solicitud de predicción que contiene uno o varios registros de entrada.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    records: list[PredictionInput] = Field(
        min_length=1
    )

# ---------------------------------------------------------
# Esquema de resultado individual
# ---------------------------------------------------------

class PredictionResult(BaseModel):
    """
    Representa el resultado de predicción correspondiente a un único registro.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    prediccion_riesgo: Literal[0, 1]

    probabilidad_riesgo: float = Field(
        ge=0.0,
        le=1.0,
    )

# ---------------------------------------------------------
# Esquema de respuesta para predicción por lotes
# ---------------------------------------------------------

class PredictionResponse(BaseModel):
    """
    Representa la respuesta de la API para uno o varios registros procesados.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    predictions: list[PredictionResult] = Field(
        min_length=1
    )

# ---------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------

app = FastAPI(
    title="API de Riesgo Crediticio",
    description=(
        "API para generar predicciones de riesgo "
        "utilizando el modelo Gradient Boosting."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------
# 1. Construcción del pipeline de despliegue
# ---------------------------------------------------------

def build_deployment_pipeline():
    """
    Construye y entrena el pipeline utilizado para inferencia.

    Reutiliza la preparación de datos y el preprocesamiento definidos en los avances anteriores, manteniendo la
    exclusión conservadora de la variable 'puntaje'.

    El modelo desplegado es Gradient Boosting, seleccionado como mejor modelo base durante el Avance 2.
    """
    (
        X_train,
        _,
        y_train,
        _,
        preprocessor,
    ) = prepare_training_data()

    model_preprocessor = build_model_preprocessor(
        preprocessor
    )

    models = build_models()

    model = models[
        DEPLOYMENT_MODEL_NAME
    ]

    pipeline = build_model_pipeline(
        model_preprocessor,
        model,
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    return pipeline

deployment_pipeline = build_deployment_pipeline()


# ---------------------------------------------------------
# 2. Preparación de nuevos registros para inferencia
# ---------------------------------------------------------

def prepare_inference_data(records):
    """
    Prepara nuevos registros para que puedan ser consumidos
    por el pipeline de despliegue.

    Los registros deben contener únicamente las variables
    originales necesarias para inferencia. Las reglas de
    calidad y el Feature Engineering se reutilizan desde
    los componentes desarrollados previamente.
    """
    if not records:
        raise ValueError(
            "Debe proporcionarse al menos un registro."
        )

    df = pd.DataFrame(records)

    missing_features = [
        feature
        for feature in INPUT_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Faltan variables requeridas para inferencia: "
            f"{missing_features}"
        )

    df = df[INPUT_FEATURES].copy()

    # 'puntaje' se mantiene fuera del contrato de la API
    # porque fue excluido conservadoramente del modelamiento.
    # Se crea únicamente para reutilizar las reglas de calidad
    # existentes sin modificar ft_engineering.py.
    df["puntaje"] = np.nan

    df = apply_quality_rules(df)

    df = create_features(df)

    return df


# ---------------------------------------------------------
# 3. Generación de predicciones
# ---------------------------------------------------------

def predict_records(records, pipeline):
    """
    Genera predicciones de riesgo para uno o varios
    registros nuevos.

    Retorna tanto la clase predicha como la probabilidad
    estimada de pertenecer a la clase de riesgo.
    """
    inference_df = prepare_inference_data(
        records
    )

    predictions = pipeline.predict(
        inference_df
    )

    probabilities = pipeline.predict_proba(
        inference_df
    )[:, 1]

    results = []

    for prediction, probability in zip(
        predictions,
        probabilities,
    ):
        results.append(
            {
                "prediccion_riesgo": int(
                    prediction
                ),
                "probabilidad_riesgo": float(
                    probability
                ),
            }
        )

    return results


# ---------------------------------------------------------
# 4. Endpoint de predicción
# ---------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
):
    """
    Recibe uno o varios registros validados por Pydantic y devuelve la predicción y probabilidad de riesgo
    correspondiente a cada observación.
    """
    records = [
        record.model_dump()
        for record in request.records
    ]

    results = predict_records(
        records,
        deployment_pipeline,
    )

    return PredictionResponse(
        predictions=results
    )