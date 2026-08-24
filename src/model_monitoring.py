import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from scipy.spatial.distance import jensenshannon
from scipy.stats import (
    chi2_contingency,
    ks_2samp,
)


from src.ft_engineering import (
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    apply_quality_rules,
    build_preprocessor,
    create_features,
    load_data,
    prepare_target,
)

from src.model_training_evaluation import (
    build_model_pipeline,
    build_model_preprocessor,
    build_models,
)


MONITORING_PERIOD_ORDER = [
    "Referencia histórica",
    "2025-Q3",
    "2025-Q4",
    "2026-Q1",
    "2026-Q2 incompleto",
]

REFERENCE_PERIOD = "Referencia histórica"

COMPLETE_MONITORING_PERIODS = [
    "2025-Q3",
    "2025-Q4",
    "2026-Q1",
]

EXPECTED_TEMPORAL_FEATURES = [
    "anio_prestamo",
    "mes_prestamo",
]


# ---------------------------------------------------------
# 1. Preparación de datos para monitoreo
# ---------------------------------------------------------

def prepare_monitoring_data():
    """
    Prepara la base que será utilizada para monitorear
    el comportamiento de los datos a lo largo del tiempo.

    Conserva la fecha original para poder construir ventanas
    temporales de monitoreo y reutiliza las transformaciones
    definidas previamente en Feature Engineering.
    """
    df = load_data()

    df = apply_quality_rules(df)

    monitoring_dates = df["fecha_prestamo"].copy()

    df = create_features(df)

    X, y = prepare_target(df)

    monitoring_df = X.copy()

    monitoring_df["riesgo_incumplimiento"] = y

    monitoring_df["fecha_prestamo"] = monitoring_dates

    return monitoring_df


# ---------------------------------------------------------
# 2. Asignación de períodos de monitoreo
# ---------------------------------------------------------

def assign_monitoring_period(df):
    """
    Asigna cada observación a una ventana temporal de
    referencia o monitoreo.

    La periodicidad definida es trimestral debido a la
    disminución del volumen de registros hacia el final
    del período observado.
    """
    df = df.copy()

    if "fecha_prestamo" not in df.columns:
        raise ValueError(
            "El DataFrame debe contener la columna "
            "'fecha_prestamo'."
        )

    dates = df["fecha_prestamo"]

    df["periodo_monitoreo"] = pd.NA

    df.loc[
        dates < "2025-07-01",
        "periodo_monitoreo",
    ] = "Referencia histórica"

    df.loc[
        (dates >= "2025-07-01")
        & (dates < "2025-10-01"),
        "periodo_monitoreo",
    ] = "2025-Q3"

    df.loc[
        (dates >= "2025-10-01")
        & (dates < "2026-01-01"),
        "periodo_monitoreo",
    ] = "2025-Q4"

    df.loc[
        (dates >= "2026-01-01")
        & (dates < "2026-04-01"),
        "periodo_monitoreo",
    ] = "2026-Q1"

    df.loc[
        dates >= "2026-04-01",
        "periodo_monitoreo",
    ] = "2026-Q2 incompleto"

    if df["periodo_monitoreo"].isna().any():
        raise ValueError(
            "Existen observaciones que no pudieron "
            "asignarse a un período de monitoreo."
        )

    df["periodo_monitoreo"] = pd.Categorical(
        df["periodo_monitoreo"],
        categories=MONITORING_PERIOD_ORDER,
        ordered=True,
    )

    return df


# ---------------------------------------------------------
# 3. Separación de ventanas para análisis de drift
# ---------------------------------------------------------

def split_monitoring_windows(df):
    """
    Separa la población histórica de referencia y las
    ventanas temporales completas utilizadas para evaluar
    data drift.

    El período 2026-Q2 se conserva en la base general,
    pero se excluye del análisis formal porque está
    incompleto y contiene muy pocas observaciones.
    """
    if "periodo_monitoreo" not in df.columns:
        raise ValueError(
            "El DataFrame debe contener la columna "
            "'periodo_monitoreo'."
        )

    reference_df = df[
        df["periodo_monitoreo"] == REFERENCE_PERIOD
    ].copy()

    monitoring_windows = {}

    for period in COMPLETE_MONITORING_PERIODS:
        monitoring_windows[period] = df[
            df["periodo_monitoreo"] == period
        ].copy()

    return reference_df, monitoring_windows


# ---------------------------------------------------------
# 4. Preparación de variables para el modelo
# ---------------------------------------------------------

def prepare_model_features(df):
    """
    Separa las columnas auxiliares utilizadas exclusivamente
    para monitoreo y devuelve las variables que pueden ser
    consumidas por el pipeline de modelamiento.

    El tratamiento final de las variables y la exclusión de
    'puntaje' se mantienen centralizados en los componentes
    desarrollados durante el Avance 2.
    """
    auxiliary_columns = [
        "riesgo_incumplimiento",
        "fecha_prestamo",
        "periodo_monitoreo",
    ]

    missing_columns = [
        column
        for column in auxiliary_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Faltan columnas auxiliares requeridas: "
            f"{missing_columns}"
        )

    X = df.drop(
        columns=auxiliary_columns,
    ).copy()

    y = df[
        "riesgo_incumplimiento"
    ].copy()

    return X, y


# ---------------------------------------------------------
# 5. Construcción del pipeline de referencia
# ---------------------------------------------------------

def build_monitoring_model_pipeline():
    """
    Construye el pipeline de Gradient Boosting utilizado
    como modelo de referencia para el monitoreo temporal.

    Reutiliza el preprocesamiento y la configuración de
    modelamiento definidos durante el Avance 2, incluyendo
    la exclusión de la variable 'puntaje'.
    """
    preprocessor = build_preprocessor()

    model_preprocessor = build_model_preprocessor(
        preprocessor
    )

    models = build_models()

    model = models["Gradient Boosting"]

    pipeline = build_model_pipeline(
        model_preprocessor,
        model,
    )

    return pipeline


# ---------------------------------------------------------
# 6. Entrenamiento del modelo de referencia
# ---------------------------------------------------------

def fit_monitoring_reference_model(reference_df):
    """
    Entrena el pipeline de Gradient Boosting utilizando
    exclusivamente la población histórica de referencia.

    El modelo resultante se utilizará posteriormente para
    generar pronósticos sobre las ventanas temporales
    futuras sin realizar reentrenamiento.
    """
    X_reference, y_reference = prepare_model_features(
        reference_df
    )

    pipeline = build_monitoring_model_pipeline()

    pipeline.fit(
        X_reference,
        y_reference,
    )

    return pipeline


# ---------------------------------------------------------
# 7. Generación de pronósticos para monitoreo
# ---------------------------------------------------------

def add_model_predictions(df, pipeline):
    """
    Agrega a la base de monitoreo la predicción binaria
    y la probabilidad estimada de riesgo.

    El pipeline recibido debe estar previamente entrenado.
    Las columnas auxiliares de monitoreo no participan
    como variables predictoras.
    """
    df = df.copy()

    X, _ = prepare_model_features(df)

    df["prediccion_riesgo"] = pipeline.predict(
        X
    )

    df["probabilidad_riesgo"] = pipeline.predict_proba(
        X
    )[:, 1]

    return df


# ---------------------------------------------------------
# 8. Kolmogorov-Smirnov para variables numéricas
# ---------------------------------------------------------

def classify_ks_drift(ks_statistic):
    """
    Clasifica la magnitud del drift según el estadístico KS.

    Umbrales utilizados:
    - KS < 0.15: estable.
    - KS entre 0.15 y 0.25: drift moderado.
    - KS > 0.25: drift significativo.
    """
    if ks_statistic < 0.15:
        return "Estable"

    if ks_statistic <= 0.25:
        return "Drift moderado"

    return "Drift significativo"


def calculate_ks_drift(
    reference_df,
    current_df,
    feature,
):
    """
    Calcula el estadístico Kolmogorov-Smirnov entre una
    distribución histórica de referencia y una distribución
    actual para una variable numérica.

    Retorna el estadístico KS, su p-value asociado,
    la clasificación del drift y los tamaños de muestra
    utilizados en la comparación.
    """
    if feature not in reference_df.columns:
        raise ValueError(
            f"La variable '{feature}' no existe "
            "en la población de referencia."
        )

    if feature not in current_df.columns:
        raise ValueError(
            f"La variable '{feature}' no existe "
            "en la población actual."
        )

    reference_values = pd.to_numeric(
        reference_df[feature],
        errors="coerce",
    ).dropna()

    current_values = pd.to_numeric(
        current_df[feature],
        errors="coerce",
    ).dropna()

    if reference_values.empty:
        raise ValueError(
            f"La variable '{feature}' no contiene "
            "valores válidos en la referencia."
        )

    if current_values.empty:
        raise ValueError(
            f"La variable '{feature}' no contiene "
            "valores válidos en la población actual."
        )

    result = ks_2samp(
        reference_values,
        current_values,
        alternative="two-sided",
        method="auto",
    )

    ks_statistic = float(result.statistic)

    return {
        "feature": feature,
        "ks_statistic": ks_statistic,
        "p_value": float(result.pvalue),
        "drift_status": classify_ks_drift(
            ks_statistic
        ),
        "n_reference": len(reference_values),
        "n_current": len(current_values),
    }


# ---------------------------------------------------------
# 9. Evaluación KS para variables numéricas
# ---------------------------------------------------------

def evaluate_numeric_ks_drift(
    reference_df,
    current_df,
):
    """
    Calcula Kolmogorov-Smirnov para todas las variables
    numéricas utilizadas por el modelo principal.

    La variable 'puntaje' se excluye para mantener
    consistencia con el modelamiento del Avance 2.

    Las variables temporales se mantienen en el análisis,
    pero se identifican como cambios temporales esperados
    y no se consideran elegibles para generar por sí solas
    una alerta operacional.
    """
    numeric_features = [
        feature
        for feature in NUMERIC_FEATURES
        if feature != "puntaje"
    ]

    results = []

    for feature in numeric_features:
        result = calculate_ks_drift(
            reference_df,
            current_df,
            feature,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="ks_statistic",
        ascending=False,
    ).reset_index(drop=True)

    results_df["drift_context"] = results_df[
        "feature"
    ].apply(
        lambda feature: (
            "Cambio temporal esperado"
            if feature in EXPECTED_TEMPORAL_FEATURES
            else "Drift poblacional evaluable"
        )
    )

    results_df["alert_eligible"] = ~results_df[
        "feature"
    ].isin(EXPECTED_TEMPORAL_FEATURES)

    return results_df


# ---------------------------------------------------------
# 10. Population Stability Index para variables numéricas
# ---------------------------------------------------------

def classify_psi_drift(psi_value):
    """
    Clasifica la magnitud del drift según el PSI.

    Umbrales utilizados:
    - PSI < 0.10: estable.
    - PSI entre 0.10 y 0.25: drift moderado.
    - PSI > 0.25: drift significativo.
    """
    if psi_value < 0.10:
        return "Estable"

    if psi_value <= 0.25:
        return "Drift moderado"

    return "Drift significativo"


def calculate_psi_drift(
    reference_df,
    current_df,
    feature,
    n_bins=10,
    epsilon=1e-6,
):
    """
    Calcula el Population Stability Index (PSI) entre una
    distribución histórica de referencia y una distribución
    actual para una variable numérica.

    Para variables con pocos valores únicos se comparan
    directamente sus proporciones. Para variables continuas,
    los intervalos se construyen a partir de cuantiles de la
    población histórica de referencia.
    """
    if feature not in reference_df.columns:
        raise ValueError(
            f"La variable '{feature}' no existe "
            "en la población de referencia."
        )

    if feature not in current_df.columns:
        raise ValueError(
            f"La variable '{feature}' no existe "
            "en la población actual."
        )

    reference_values = pd.to_numeric(
        reference_df[feature],
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    current_values = pd.to_numeric(
        current_df[feature],
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if reference_values.empty:
        raise ValueError(
            f"La variable '{feature}' no contiene "
            "valores válidos en la referencia."
        )

    if current_values.empty:
        raise ValueError(
            f"La variable '{feature}' no contiene "
            "valores válidos en la población actual."
        )

    if reference_values.nunique() <= n_bins:
        categories = sorted(
            set(reference_values.unique())
            | set(current_values.unique())
        )

        reference_distribution = (
            reference_values.value_counts(
                normalize=True
            )
            .reindex(
                categories,
                fill_value=0,
            )
            .sort_index()
        )

        current_distribution = (
            current_values.value_counts(
                normalize=True
            )
            .reindex(
                categories,
                fill_value=0,
            )
            .sort_index()
        )

    else:
        quantiles = np.linspace(
            0,
            1,
            n_bins + 1,
        )

        bin_edges = np.unique(
            np.quantile(
                reference_values,
                quantiles,
            )
        )

        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        reference_binned = pd.cut(
            reference_values,
            bins=bin_edges,
            include_lowest=True,
        )

        current_binned = pd.cut(
            current_values,
            bins=bin_edges,
            include_lowest=True,
        )

        reference_distribution = (
            reference_binned.value_counts(
                normalize=True,
                sort=False,
            )
        )

        current_distribution = (
            current_binned.value_counts(
                normalize=True,
                sort=False,
            )
        )

    reference_proportions = np.clip(
        reference_distribution.to_numpy(),
        epsilon,
        None,
    )

    current_proportions = np.clip(
        current_distribution.to_numpy(),
        epsilon,
        None,
    )

    psi_value = np.sum(
        (
            current_proportions
            - reference_proportions
        )
        * np.log(
            current_proportions
            / reference_proportions
        )
    )

    psi_value = float(psi_value)

    return {
        "feature": feature,
        "psi": psi_value,
        "drift_status": classify_psi_drift(
            psi_value
        ),
        "n_reference": len(reference_values),
        "n_current": len(current_values),
    }


# ---------------------------------------------------------
# 11. Evaluación PSI para variables numéricas
# ---------------------------------------------------------

def evaluate_numeric_psi_drift(
    reference_df,
    current_df,
):
    """
    Calcula Population Stability Index para todas las
    variables numéricas utilizadas por el modelo principal.

    La variable 'puntaje' se excluye para mantener
    consistencia con el modelamiento del Avance 2.

    Las variables temporales se mantienen en el análisis,
    pero no se consideran elegibles para generar por sí
    solas una alerta operacional.
    """
    numeric_features = [
        feature
        for feature in NUMERIC_FEATURES
        if feature != "puntaje"
    ]

    results = []

    for feature in numeric_features:
        result = calculate_psi_drift(
            reference_df,
            current_df,
            feature,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="psi",
        ascending=False,
    ).reset_index(drop=True)

    results_df["drift_context"] = results_df[
        "feature"
    ].apply(
        lambda feature: (
            "Cambio temporal esperado"
            if feature in EXPECTED_TEMPORAL_FEATURES
            else "Drift poblacional evaluable"
        )
    )

    results_df["alert_eligible"] = ~results_df[
        "feature"
    ].isin(EXPECTED_TEMPORAL_FEATURES)

    return results_df


# ---------------------------------------------------------
# 12. Jensen-Shannon para variables numéricas
# ---------------------------------------------------------

def classify_js_drift(js_divergence):
    """
    Clasifica la magnitud del drift según la divergencia
    de Jensen-Shannon.

    Umbrales utilizados:
    - JSD < 0.10: estable.
    - JSD entre 0.10 y 0.30: drift moderado.
    - JSD > 0.30: drift significativo.
    """
    if js_divergence < 0.10:
        return "Estable"

    if js_divergence <= 0.30:
        return "Drift moderado"

    return "Drift significativo"


def calculate_js_drift(
    reference_df,
    current_df,
    feature,
    n_bins=10,
    epsilon=1e-6,
):
    """
    Calcula la divergencia de Jensen-Shannon entre una
    distribución histórica de referencia y una distribución
    actual para una variable numérica.

    Los intervalos se construyen utilizando cuantiles de la
    población histórica para mantener una base de comparación
    consistente entre períodos.
    """
    if feature not in reference_df.columns:
        raise ValueError(
            f"La variable '{feature}' no existe "
            "en la población de referencia."
        )

    if feature not in current_df.columns:
        raise ValueError(
            f"La variable '{feature}' no existe "
            "en la población actual."
        )

    reference_values = pd.to_numeric(
        reference_df[feature],
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    current_values = pd.to_numeric(
        current_df[feature],
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if reference_values.empty:
        raise ValueError(
            f"La variable '{feature}' no contiene "
            "valores válidos en la referencia."
        )

    if current_values.empty:
        raise ValueError(
            f"La variable '{feature}' no contiene "
            "valores válidos en la población actual."
        )

    if reference_values.nunique() <= n_bins:
        categories = sorted(
            set(reference_values.unique())
            | set(current_values.unique())
        )

        reference_distribution = (
            reference_values.value_counts(
                normalize=True
            )
            .reindex(
                categories,
                fill_value=0,
            )
            .sort_index()
        )

        current_distribution = (
            current_values.value_counts(
                normalize=True
            )
            .reindex(
                categories,
                fill_value=0,
            )
            .sort_index()
        )

    else:
        quantiles = np.linspace(
            0,
            1,
            n_bins + 1,
        )

        bin_edges = np.unique(
            np.quantile(
                reference_values,
                quantiles,
            )
        )

        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        reference_binned = pd.cut(
            reference_values,
            bins=bin_edges,
            include_lowest=True,
        )

        current_binned = pd.cut(
            current_values,
            bins=bin_edges,
            include_lowest=True,
        )

        reference_distribution = (
            reference_binned.value_counts(
                normalize=True,
                sort=False,
            )
        )

        current_distribution = (
            current_binned.value_counts(
                normalize=True,
                sort=False,
            )
        )

    reference_proportions = np.clip(
        reference_distribution.to_numpy(),
        epsilon,
        None,
    )

    current_proportions = np.clip(
        current_distribution.to_numpy(),
        epsilon,
        None,
    )

    reference_proportions = (
        reference_proportions
        / reference_proportions.sum()
    )

    current_proportions = (
        current_proportions
        / current_proportions.sum()
    )

    js_distance = jensenshannon(
        reference_proportions,
        current_proportions,
        base=2,
    )

    js_divergence = float(
        js_distance ** 2
    )

    return {
        "feature": feature,
        "js_divergence": js_divergence,
        "drift_status": classify_js_drift(
            js_divergence
        ),
        "n_reference": len(reference_values),
        "n_current": len(current_values),
    }


# ---------------------------------------------------------
# 13. Evaluación Jensen-Shannon para variables numéricas
# ---------------------------------------------------------

def evaluate_numeric_js_drift(
    reference_df,
    current_df,
):
    """
    Calcula la divergencia de Jensen-Shannon para todas las
    variables numéricas utilizadas por el modelo principal.

    La variable 'puntaje' se excluye para mantener
    consistencia con el modelamiento del Avance 2.

    Las variables temporales se mantienen en el análisis,
    pero no se consideran elegibles para generar por sí
    solas una alerta operacional.
    """
    numeric_features = [
        feature
        for feature in NUMERIC_FEATURES
        if feature != "puntaje"
    ]

    results = []

    for feature in numeric_features:
        result = calculate_js_drift(
            reference_df,
            current_df,
            feature,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="js_divergence",
        ascending=False,
    ).reset_index(drop=True)

    results_df["drift_context"] = results_df[
        "feature"
    ].apply(
        lambda feature: (
            "Cambio temporal esperado"
            if feature in EXPECTED_TEMPORAL_FEATURES
            else "Drift poblacional evaluable"
        )
    )

    results_df["alert_eligible"] = ~results_df[
        "feature"
    ].isin(EXPECTED_TEMPORAL_FEATURES)

    return results_df


# ---------------------------------------------------------
# 14. Chi-cuadrado para variables categóricas
# ---------------------------------------------------------

def classify_chi_square_drift(
    p_value,
    alpha=0.05,
):
    """
    Clasifica el resultado del test Chi-cuadrado según
    su significancia estadística.

    Un p-value menor que alpha indica evidencia de que la
    distribución categórica cambió respecto de la referencia.
    """
    if p_value < alpha:
        return "Cambio significativo"

    return "Sin evidencia de cambio"


def calculate_chi_square_drift(
    reference_df,
    current_df,
    feature,
    alpha=0.05,
):
    """
    Compara mediante Chi-cuadrado la distribución de una
    variable categórica entre la población histórica de
    referencia y la población actual.

    Los valores faltantes se representan como una categoría
    explícita para que cambios en su frecuencia también
    puedan ser detectados.
    """
    if feature not in reference_df.columns:
        raise ValueError(
            f"La variable '{feature}' no existe "
            "en la población de referencia."
        )

    if feature not in current_df.columns:
        raise ValueError(
            f"La variable '{feature}' no existe "
            "en la población actual."
        )

    reference_values = (
        reference_df[feature]
        .astype("object")
        .fillna("__MISSING__")
    )

    current_values = (
        current_df[feature]
        .astype("object")
        .fillna("__MISSING__")
    )

    categories = sorted(
        set(reference_values.unique())
        | set(current_values.unique()),
        key=str,
    )

    reference_counts = (
        reference_values.value_counts()
        .reindex(
            categories,
            fill_value=0,
        )
    )

    current_counts = (
        current_values.value_counts()
        .reindex(
            categories,
            fill_value=0,
        )
    )

    contingency_table = np.array(
        [
            reference_counts.to_numpy(),
            current_counts.to_numpy(),
        ]
    )

    chi2_statistic, p_value, _, _ = chi2_contingency(
        contingency_table
    )

    p_value = float(p_value)

    return {
        "feature": feature,
        "chi2_statistic": float(chi2_statistic),
        "p_value": p_value,
        "drift_status": classify_chi_square_drift(
            p_value,
            alpha=alpha,
        ),
        "n_categories": len(categories),
        "n_reference": len(reference_values),
        "n_current": len(current_values),
    }


# ---------------------------------------------------------
# 15. Evaluación Chi-cuadrado para variables categóricas
# ---------------------------------------------------------

def evaluate_categorical_chi_square_drift(
    reference_df,
    current_df,
    alpha=0.05,
):
    """
    Calcula Chi-cuadrado para las variables categóricas
    utilizadas por el modelo principal.

    Incluye tanto variables nominales como ordinales y
    devuelve una tabla ordenada según el p-value, mostrando
    primero las variables con mayor evidencia estadística
    de cambio.
    """
    categorical_features = (
        NOMINAL_FEATURES
        + ORDINAL_FEATURES
    )

    results = []

    for feature in categorical_features:
        result = calculate_chi_square_drift(
            reference_df,
            current_df,
            feature,
            alpha=alpha,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="p_value",
        ascending=True,
    ).reset_index(drop=True)

    return results_df


# ---------------------------------------------------------
# 16. Consolidación de métricas numéricas de drift
# ---------------------------------------------------------

def build_numeric_drift_summary(
    reference_df,
    current_df,
):
    """
    Consolida las métricas KS, PSI y Jensen-Shannon de las
    variables numéricas utilizadas por el modelo principal.

    Retorna una única tabla por variable para facilitar
    el monitoreo y la posterior generación de alertas.
    """
    ks_results = evaluate_numeric_ks_drift(
        reference_df,
        current_df,
    )

    psi_results = evaluate_numeric_psi_drift(
        reference_df,
        current_df,
    )

    js_results = evaluate_numeric_js_drift(
        reference_df,
        current_df,
    )

    ks_summary = ks_results[
        [
            "feature",
            "ks_statistic",
            "p_value",
            "drift_status",
            "drift_context",
            "alert_eligible",
        ]
    ].rename(
        columns={
            "p_value": "ks_p_value",
            "drift_status": "ks_status",
        }
    )

    psi_summary = psi_results[
        [
            "feature",
            "psi",
            "drift_status",
        ]
    ].rename(
        columns={
            "drift_status": "psi_status",
        }
    )

    js_summary = js_results[
        [
            "feature",
            "js_divergence",
            "drift_status",
        ]
    ].rename(
        columns={
            "drift_status": "js_status",
        }
    )

    summary = ks_summary.merge(
        psi_summary,
        on="feature",
        how="inner",
    )

    summary = summary.merge(
        js_summary,
        on="feature",
        how="inner",
    )

    return summary


# ---------------------------------------------------------
# 17. Sistema de alertas para drift numérico
# ---------------------------------------------------------

def assign_numeric_drift_alerts(summary_df):
    """
    Asigna un nivel de alerta consolidado utilizando
    conjuntamente KS, PSI y Jensen-Shannon.

    La lógica prioriza la coincidencia entre métricas para
    evitar generar alertas críticas a partir de un único
    indicador aislado.
    """
    summary_df = summary_df.copy()

    metric_columns = [
        "ks_status",
        "psi_status",
        "js_status",
    ]

    summary_df["significant_count"] = (
        summary_df[metric_columns]
        .eq("Drift significativo")
        .sum(axis=1)
    )

    summary_df["moderate_count"] = (
        summary_df[metric_columns]
        .eq("Drift moderado")
        .sum(axis=1)
    )

    def determine_alert(row):
        if not row["alert_eligible"]:
            return "Informativa"

        if row["significant_count"] >= 2:
            return "Crítica"

        if (
            row["significant_count"] == 1
            or row["moderate_count"] >= 2
        ):
            return "Alerta"

        if row["moderate_count"] == 1:
            return "Vigilancia"

        return "Verde"

    summary_df["alert_level"] = summary_df.apply(
        determine_alert,
        axis=1,
    )

    return summary_df


# ---------------------------------------------------------
# 18. Sistema de alertas para drift categórico
# ---------------------------------------------------------

def assign_categorical_drift_alerts(summary_df):
    """
    Asigna un nivel de alerta a las variables categóricas
    evaluadas mediante Chi-cuadrado.

    Debido a que el test informa principalmente evidencia
    estadística de cambio y no una magnitud operacional
    directamente comparable con KS, PSI o Jensen-Shannon,
    un resultado significativo se clasifica como 'Alerta'
    y no como 'Crítica'.
    """
    summary_df = summary_df.copy()

    summary_df["alert_level"] = np.where(
        summary_df["p_value"] < 0.05,
        "Alerta",
        "Verde",
    )

    return summary_df


# ---------------------------------------------------------
# 19. Tabla general de monitoreo por período
# ---------------------------------------------------------

def build_period_drift_summary(
    reference_df,
    current_df,
    period_name,
):
    """
    Consolida en una única tabla los resultados de drift
    numérico y categórico correspondientes a un período
    de monitoreo.

    Cada variable conserva las métricas aplicables según
    su tipo y recibe un nivel de alerta operacional.
    """
    numeric_summary = build_numeric_drift_summary(
        reference_df,
        current_df,
    )

    numeric_summary = assign_numeric_drift_alerts(
        numeric_summary
    )

    numeric_summary = numeric_summary.copy()

    numeric_summary["variable_type"] = "Numérica"

    categorical_summary = (
        evaluate_categorical_chi_square_drift(
            reference_df,
            current_df,
        )
    )

    categorical_summary = (
        assign_categorical_drift_alerts(
            categorical_summary
        )
    )

    categorical_summary = categorical_summary.copy()

    categorical_summary["variable_type"] = "Categórica"

    numeric_columns = [
        "feature",
        "variable_type",
        "ks_statistic",
        "ks_p_value",
        "psi",
        "js_divergence",
        "drift_context",
        "alert_eligible",
        "alert_level",
    ]

    categorical_columns = [
        "feature",
        "variable_type",
        "chi2_statistic",
        "p_value",
        "alert_level",
    ]

    numeric_output = numeric_summary[
        numeric_columns
    ].copy()

    categorical_output = categorical_summary[
        categorical_columns
    ].copy()

    categorical_output = categorical_output.rename(
        columns={
            "p_value": "chi2_p_value",
        }
    )

    summary = pd.concat(
        [
            numeric_output,
            categorical_output,
        ],
        ignore_index=True,
        sort=False,
    )

    summary["periodo_monitoreo"] = period_name

    return summary


# ---------------------------------------------------------
# 20. Análisis temporal de Data Drift
# ---------------------------------------------------------

def build_temporal_drift_summary(
    reference_df,
    monitoring_windows,
):
    """
    Ejecuta el análisis de drift para todas las ventanas
    temporales completas utilizando una misma población
    histórica de referencia.

    Retorna una tabla consolidada que permite analizar
    cómo evoluciona el drift a lo largo del tiempo.
    """
    results = []

    for period in COMPLETE_MONITORING_PERIODS:
        if period not in monitoring_windows:
            raise ValueError(
                f"No se encontró la ventana '{period}' "
                "en los datos de monitoreo."
            )

        period_summary = build_period_drift_summary(
            reference_df,
            monitoring_windows[period],
            period,
        )

        results.append(period_summary)

    temporal_summary = pd.concat(
        results,
        ignore_index=True,
        sort=False,
    )

    return temporal_summary


# ---------------------------------------------------------
# 21. Identificación de tendencias críticas de drift
# ---------------------------------------------------------

def identify_critical_drift_trends(temporal_summary):
    """
    Resume la presencia de alertas críticas de cada variable
    a lo largo de las ventanas temporales de monitoreo.

    Permite distinguir entre drift crítico persistente,
    drift crítico reciente y variables sin criticidad.
    """
    critical_flags = (
        temporal_summary
        .assign(
            is_critical=(
                temporal_summary["alert_level"]
                == "Crítica"
            )
        )
        .pivot_table(
            index="feature",
            columns="periodo_monitoreo",
            values="is_critical",
            aggfunc="max",
            fill_value=False,
        )
    )

    critical_flags = critical_flags.reindex(
        columns=COMPLETE_MONITORING_PERIODS,
        fill_value=False,
    )

    critical_counts = critical_flags.sum(
        axis=1
    )

    latest_period = COMPLETE_MONITORING_PERIODS[-1]

    trend_summary = pd.DataFrame(
        {
            "feature": critical_flags.index,
            "critical_periods": critical_counts.values,
            "latest_period_critical": (
                critical_flags[latest_period].values
            ),
        }
    )

    def classify_trend(row):
        if row["critical_periods"] >= 2:
            return "Drift crítico persistente"

        if (
            row["critical_periods"] == 1
            and row["latest_period_critical"]
        ):
            return "Drift crítico reciente"

        if row["critical_periods"] == 1:
            return "Drift crítico aislado"

        return "Sin drift crítico"

    trend_summary["trend_status"] = trend_summary.apply(
        classify_trend,
        axis=1,
    )

    trend_summary = trend_summary.sort_values(
        by=[
            "critical_periods",
            "feature",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    return trend_summary


# ---------------------------------------------------------
# 22. Recomendaciones automáticas de monitoreo
# ---------------------------------------------------------

def build_monitoring_recommendations(
    temporal_summary,
):
    """
    Genera recomendaciones automáticas a partir de la
    evolución temporal de las alertas críticas.

    Las recomendaciones no ordenan reentrenar el modelo
    automáticamente, ya que la presencia de data drift
    no implica necesariamente degradación de performance.
    """
    trends = identify_critical_drift_trends(
        temporal_summary
    )

    relevant_trends = trends[
        trends["critical_periods"] > 0
    ].copy()

    def generate_recommendation(row):
        if (
            row["trend_status"]
            == "Drift crítico persistente"
        ):
            return (
                "Prioridad alta: revisar la variable, "
                "analizar su impacto sobre las predicciones "
                "y evaluar reentrenamiento del modelo."
            )

        if (
            row["trend_status"]
            == "Drift crítico reciente"
        ):
            return (
                "Prioridad media-alta: investigar el cambio "
                "reciente y confirmar su persistencia en la "
                "próxima ventana de monitoreo."
            )

        if (
            row["trend_status"]
            == "Drift crítico aislado"
        ):
            return (
                "Mantener bajo observación. El cambio crítico "
                "no persiste en la ventana más reciente."
            )

        return (
            "Continuar con el monitoreo periódico."
        )

    relevant_trends["recommendation"] = (
        relevant_trends.apply(
            generate_recommendation,
            axis=1,
        )
    )

    return relevant_trends


# ---------------------------------------------------------
# 23. Aplicación Streamlit
# ---------------------------------------------------------

@st.cache_data
def prepare_dashboard_data():
    """
    Prepara y consolida los resultados necesarios para
    mostrar el dashboard de monitoreo en Streamlit.

    El uso de caché evita repetir todos los cálculos de drift
    cada vez que la interfaz se actualiza.
    """
    df = prepare_monitoring_data()

    df = assign_monitoring_period(
        df
    )

    reference_df, monitoring_windows = (
        split_monitoring_windows(
            df
        )
    )

    monitoring_model = (
        fit_monitoring_reference_model(
            reference_df
        )
    )

    df = add_model_predictions(
        df,
        monitoring_model,
    )

    temporal_summary = (
        build_temporal_drift_summary(
            reference_df,
            monitoring_windows,
        )
    )

    recommendations = (
        build_monitoring_recommendations(
            temporal_summary
        )
    )

    return (
        df,
        temporal_summary,
        recommendations,
    )


def run_streamlit_app():
    """
    Ejecuta la interfaz principal de monitoreo del modelo
    de riesgo crediticio.
    """
    st.set_page_config(
        page_title="Monitoreo de Riesgo Crediticio",
        page_icon="📊",
        layout="wide",
    )

    # -----------------------------------------------------
    # Encabezado
    # -----------------------------------------------------

    st.title(
        "Monitoreo del Modelo de Riesgo Crediticio"
    )

    st.write(
        "Dashboard para el seguimiento de Data Drift "
        "y la evolución temporal de las variables "
        "utilizadas por el modelo."
    )

    st.info(
        "Referencia histórica: 26/11/2024 al 30/06/2025. "
        "Periodicidad de monitoreo: trimestral."
    )

    # -----------------------------------------------------
    # Preparación de datos del dashboard
    # -----------------------------------------------------

    df, temporal_summary, recommendations = (
        prepare_dashboard_data()
    )

    # -----------------------------------------------------
    # Selector de período
    # -----------------------------------------------------

    selected_period = st.selectbox(
        "Seleccionar período de monitoreo",
        options=COMPLETE_MONITORING_PERIODS,
        index=len(COMPLETE_MONITORING_PERIODS) - 1,
    )

    selected_summary = temporal_summary[
        temporal_summary["periodo_monitoreo"]
        == selected_period
    ].copy()

    # -----------------------------------------------------
    # Resumen ejecutivo
    # -----------------------------------------------------

    critical_count = (
        selected_summary["alert_level"]
        .eq("Crítica")
        .sum()
    )

    alert_count = (
        selected_summary["alert_level"]
        .eq("Alerta")
        .sum()
    )

    st.subheader(
        "Resumen del período seleccionado"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Período",
        selected_period,
    )

    col2.metric(
        "Variables monitoreadas",
        len(selected_summary),
    )

    col3.metric(
        "Variables críticas",
        int(critical_count),
    )

    col4.metric(
        "Variables en alerta",
        int(alert_count),
    )

    # -----------------------------------------------------
    # Pronósticos del modelo
    # -----------------------------------------------------

    st.subheader(
        "Pronósticos del modelo"
    )

    st.write(
        "Predicciones generadas por el Gradient Boosting "
        "entrenado exclusivamente con la población histórica "
        "de referencia."
    )

    selected_predictions = df[
        df["periodo_monitoreo"] == selected_period
    ].copy()

    predicted_risk_count = int(
        selected_predictions[
            "prediccion_riesgo"
        ].sum()
    )

    predicted_risk_rate = (
        selected_predictions[
            "prediccion_riesgo"
        ].mean()
        * 100
    )

    average_risk_probability = (
        selected_predictions[
            "probabilidad_riesgo"
        ].mean()
        * 100
    )

    pred_col1, pred_col2, pred_col3, pred_col4 = (
        st.columns(4)
    )

    pred_col1.metric(
        "Registros del período",
        len(selected_predictions),
    )

    pred_col2.metric(
        "Predichos como riesgo",
        predicted_risk_count,
    )

    pred_col3.metric(
        "Tasa predicha de riesgo",
        f"{predicted_risk_rate:.2f}%",
    )

    pred_col4.metric(
        "Probabilidad media de riesgo",
        f"{average_risk_probability:.2f}%",
    )

    st.caption(
        "La clasificación binaria utiliza el threshold "
        "estándar de 0.50. Estas métricas describen los "
        "pronósticos del modelo y no constituyen por sí "
        "mismas una medida de Data Drift."
    )

    st.write(
        "Muestra de datos y pronósticos del período seleccionado"
    )

    prediction_columns = [
        "fecha_prestamo",
        "capital_prestado",
        "plazo_meses",
        "edad_cliente",
        "salario_cliente",
        "riesgo_incumplimiento",
        "prediccion_riesgo",
        "probabilidad_riesgo",
    ]

    prediction_sample = selected_predictions[
        prediction_columns
    ].copy()

    prediction_sample["probabilidad_riesgo"] = (
        prediction_sample["probabilidad_riesgo"]
        * 100
    )

    prediction_sample = prediction_sample.rename(
        columns={
            "fecha_prestamo": "Fecha préstamo",
            "capital_prestado": "Capital prestado",
            "plazo_meses": "Plazo (meses)",
            "edad_cliente": "Edad",
            "salario_cliente": "Salario",
            "riesgo_incumplimiento": "Resultado real",
            "prediccion_riesgo": "Predicción",
            "probabilidad_riesgo": "Probabilidad riesgo (%)",
        }
    )

    st.dataframe(
        prediction_sample.head(20),
        hide_index=True,
        use_container_width=True,
    )

    # -----------------------------------------------------
    # Tabla detallada de Data Drift
    # -----------------------------------------------------

    st.subheader(
        "Detalle de métricas de Data Drift"
    )

    alert_order = {
        "Crítica": 0,
        "Alerta": 1,
        "Vigilancia": 2,
        "Informativa": 3,
        "Verde": 4,
    }

    display_summary = selected_summary.copy()

    display_summary["alert_order"] = (
        display_summary["alert_level"]
        .map(alert_order)
    )

    display_summary = display_summary.sort_values(
        by=[
            "alert_order",
            "feature",
        ]
    )

    display_summary = display_summary[
        [
            "feature",
            "variable_type",
            "ks_statistic",
            "psi",
            "js_divergence",
            "chi2_statistic",
            "alert_level",
        ]
    ].rename(
        columns={
            "feature": "Variable",
            "variable_type": "Tipo",
            "ks_statistic": "KS",
            "psi": "PSI",
            "js_divergence": "Jensen-Shannon",
            "chi2_statistic": "Chi-cuadrado",
            "alert_level": "Nivel de alerta",
        }
    )

    st.dataframe(
        display_summary,
        hide_index=True,
        use_container_width=True,
    )

    # -----------------------------------------------------
    # Comparación de distribuciones numéricas
    # -----------------------------------------------------

    st.subheader(
        "Comparación de distribuciones"
    )

    numeric_features_available = (
        selected_summary.loc[
            selected_summary["variable_type"] == "Numérica",
            "feature",
        ]
        .sort_values()
        .tolist()
    )

    selected_numeric_feature = st.selectbox(
        "Seleccionar variable numérica",
        options=numeric_features_available,
    )

    reference_plot_df = df[
        df["periodo_monitoreo"]
        == REFERENCE_PERIOD
    ]

    current_plot_df = df[
        df["periodo_monitoreo"]
        == selected_period
    ]

    reference_values = pd.to_numeric(
        reference_plot_df[
            selected_numeric_feature
        ],
        errors="coerce",
    ).dropna()

    current_values = pd.to_numeric(
        current_plot_df[
            selected_numeric_feature
        ],
        errors="coerce",
    ).dropna()

    fig, ax = plt.subplots(
        figsize=(7, 3.5)
    )

    ax.hist(
        reference_values,
        bins=20,
        alpha=0.5,
        density=True,
        label="Referencia histórica",
    )

    ax.hist(
        current_values,
        bins=20,
        alpha=0.5,
        density=True,
        label=selected_period,
    )

    ax.set_title(
        f"Distribución de {selected_numeric_feature}"
    )

    ax.set_xlabel(
        selected_numeric_feature
    )

    ax.set_ylabel(
        "Densidad"
    )

    ax.legend()

    fig.tight_layout()

    st.pyplot(
        fig,
        width=700,
    )

    plt.close(fig)

    # -----------------------------------------------------
    # Comparación de distribuciones categóricas
    # -----------------------------------------------------

    st.subheader(
        "Comparación de distribuciones categóricas"
    )

    categorical_features_available = (
        selected_summary.loc[
            selected_summary["variable_type"] == "Categórica",
            "feature",
        ]
        .sort_values()
        .tolist()
    )

    selected_categorical_feature = st.selectbox(
        "Seleccionar variable categórica",
        options=categorical_features_available,
    )

    reference_categories = (
        reference_plot_df[
            selected_categorical_feature
        ]
        .astype("object")
        .fillna("__MISSING__")
        .value_counts(
            normalize=True
        )
    )

    current_categories = (
        current_plot_df[
            selected_categorical_feature
        ]
        .astype("object")
        .fillna("__MISSING__")
        .value_counts(
            normalize=True
        )
    )

    categories = sorted(
        set(reference_categories.index)
        | set(current_categories.index),
        key=str,
    )

    categorical_plot_df = pd.DataFrame(
        {
            "Referencia histórica": (
                reference_categories.reindex(
                    categories,
                    fill_value=0,
                )
            ),
            selected_period: (
                current_categories.reindex(
                    categories,
                    fill_value=0,
                )
            ),
        }
    )

    ax = categorical_plot_df.plot(
        kind="bar",
        figsize=(7, 3.5),
    )

    ax.set_title(
        f"Distribución de {selected_categorical_feature}"
    )

    ax.set_xlabel(
        selected_categorical_feature
    )

    ax.set_ylabel(
        "Proporción"
    )

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    fig = ax.get_figure()

    fig.tight_layout()

    st.pyplot(
        fig,
        width=700,
    )

    plt.close(fig)

    # -----------------------------------------------------
    # Evolución temporal de alertas
    # -----------------------------------------------------

    st.subheader(
        "Evolución temporal de alertas"
    )

    alert_evolution = (
        temporal_summary
        .groupby(
            [
                "periodo_monitoreo",
                "alert_level",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reindex(
            COMPLETE_MONITORING_PERIODS
        )
    )

    alert_columns = [
        level
        for level in [
            "Crítica",
            "Alerta",
            "Vigilancia",
            "Informativa",
            "Verde",
        ]
        if level in alert_evolution.columns
    ]

    alert_evolution = alert_evolution[
        alert_columns
    ]

    fig, ax = plt.subplots(
        figsize=(7, 3.5)
    )

    for alert_level in alert_columns:
        ax.plot(
            alert_evolution.index,
            alert_evolution[alert_level],
            marker="o",
            label=alert_level,
        )

    ax.set_title(
        "Evolución de variables por nivel de alerta"
    )

    ax.set_xlabel(
        "Período de monitoreo"
    )

    ax.set_ylabel(
        "Cantidad de variables"
    )

    ax.legend()

    fig.tight_layout()

    st.pyplot(
        fig,
        width=700,
    )

    plt.close(fig)

    # -----------------------------------------------------
    # Recomendaciones automáticas
    # -----------------------------------------------------

    st.subheader(
        "Recomendaciones de monitoreo"
    )

    st.write(
        "Las siguientes recomendaciones se generan a partir "
        "de la persistencia de alertas críticas observadas "
        "en las ventanas temporales de monitoreo."
    )

    if recommendations.empty:
        st.success(
            "No se detectaron variables con drift crítico "
            "durante los períodos analizados."
        )

    else:
        for _, row in recommendations.iterrows():
            if (
                row["trend_status"]
                == "Drift crítico persistente"
            ):
                st.error(
                    f"**{row['feature']}** — "
                    f"{row['trend_status']}. "
                    f"{row['recommendation']}"
                )

            elif (
                row["trend_status"]
                == "Drift crítico reciente"
            ):
                st.warning(
                    f"**{row['feature']}** — "
                    f"{row['trend_status']}. "
                    f"{row['recommendation']}"
                )

            else:
                st.info(
                    f"**{row['feature']}** — "
                    f"{row['trend_status']}. "
                    f"{row['recommendation']}"
                )

if __name__ == "__main__":
    run_streamlit_app()
