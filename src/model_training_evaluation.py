import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)
from sklearn.base import clone


# ---------------------------------------------------------
# Configuración de modelamiento
# ---------------------------------------------------------

EXCLUDED_MODEL_FEATURES = [
    "puntaje",
]


# ---------------------------------------------------------
# Preparación del preprocesador para modelamiento
# ---------------------------------------------------------

def build_model_preprocessor(
    preprocessor,
    excluded_features=None,
):
    """
    Crea una copia del preprocesador excluyendo las variables
    que no deben participar del modelamiento principal.

    La exclusión se realiza sin modificar el preprocesador
    original proveniente de Feature Engineering.
    """
    if excluded_features is None:
        excluded_features = EXCLUDED_MODEL_FEATURES

    excluded_features = set(excluded_features)

    model_preprocessor = clone(preprocessor)

    configured_features = {
        column
        for _, _, columns in model_preprocessor.transformers
        for column in columns
    }

    unknown_features = excluded_features - configured_features

    if unknown_features:
        raise ValueError(
            "Las siguientes variables a excluir no están "
            f"configuradas en el preprocesador: {sorted(unknown_features)}"
        )

    filtered_transformers = []

    for name, transformer, columns in model_preprocessor.transformers:
        filtered_columns = [
            column
            for column in columns
            if column not in excluded_features
        ]

        filtered_transformers.append(
            (
                name,
                transformer,
                filtered_columns,
            )
        )

    model_preprocessor.transformers = filtered_transformers

    return model_preprocessor


# ---------------------------------------------------------
# 1. Construcción de modelos
# ---------------------------------------------------------

def build_models(random_state=42):
    """
    Construye los modelos supervisados iniciales que serán
    comparados bajo el mismo esquema de preprocesamiento
    y evaluación.

    Se utilizan configuraciones base y reproducibles,
    sin optimización de hiperparámetros.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
        ),
        "Random Forest": RandomForestClassifier(
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=random_state,
        ),
    }

    return models


# ---------------------------------------------------------
# 2. Construcción del pipeline de modelado
# ---------------------------------------------------------

def build_model_pipeline(preprocessor, model):
    """
    Combina el preprocesamiento y el modelo dentro de un
    único Pipeline de scikit-learn.

    Esto permite que todas las transformaciones se ajusten
    exclusivamente con los datos de entrenamiento.
    """
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


# ---------------------------------------------------------
# 3. Resumen de métricas de clasificación
# ---------------------------------------------------------

def summarize_classification(
    y_true,
    y_pred,
    y_score,
):
    """
    Calcula las principales métricas de clasificación.

    La clase positiva (1) representa riesgo de incumplimiento.
    """
    metrics = {
        "Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "ROC-AUC": roc_auc_score(
            y_true,
            y_score,
        ),
        "PR-AUC": average_precision_score(
            y_true,
            y_score,
        ),
    }

    return metrics


# ---------------------------------------------------------
# 4. Estrategia de validación cruzada
# ---------------------------------------------------------

def build_cv_strategy(
    n_splits=5,
    random_state=42,
):
    """
    Construye una estrategia de validación cruzada estratificada.

    La estratificación permite preservar aproximadamente la
    proporción de la clase minoritaria en cada fold.
    """
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    return cv


# ---------------------------------------------------------
# 5. Evaluación mediante validación cruzada
# ---------------------------------------------------------

def evaluate_model_cv(
    pipeline,
    X_train,
    y_train,
    cv,
):
    """
    Evalúa un pipeline mediante validación cruzada
    estratificada.

    Retorna la media y el desvío estándar de las principales
    métricas obtenidas entre los folds.
    """
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(
            precision_score,
            zero_division=0,
        ),
        "recall": make_scorer(
            recall_score,
            zero_division=0,
        ),
        "f1": make_scorer(
            f1_score,
            zero_division=0,
        ),
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )

    summary = {}

    for metric in scoring:
        values = cv_results[f"test_{metric}"]

        summary[f"{metric}_mean"] = np.mean(values)
        summary[f"{metric}_std"] = np.std(values)

    return summary


# ---------------------------------------------------------
# 6. Comparación de modelos mediante validación cruzada
# ---------------------------------------------------------

def evaluate_models_cv(
    models,
    preprocessor,
    X_train,
    y_train,
    cv,
):
    """
    Evalúa varios modelos bajo las mismas condiciones de
    preprocesamiento y validación cruzada.

    Retorna una tabla comparativa con la media y el desvío
    estándar de cada métrica.
    """
    results = []

    for model_name, model in models.items():
        pipeline = build_model_pipeline(
            clone(preprocessor),
            clone(model),
        )

        summary = evaluate_model_cv(
            pipeline,
            X_train,
            y_train,
            cv,
        )

        summary["Model"] = model_name

        results.append(summary)

    results_df = pd.DataFrame(results)

    ordered_columns = [
        "Model",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "pr_auc_std",
    ]

    results_df = results_df[ordered_columns]

    return results_df


# ---------------------------------------------------------
# 7. Evaluación final sobre el conjunto de test
# ---------------------------------------------------------

def evaluate_model_test(
    pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Entrena un pipeline utilizando todo el conjunto de
    entrenamiento y lo evalúa sobre el conjunto de test.
    """
    pipeline.fit(
        X_train,
        y_train,
    )

    y_pred = pipeline.predict(
        X_test,
    )

    y_score = pipeline.predict_proba(
        X_test,
    )[:, 1]

    metrics = summarize_classification(
        y_test,
        y_pred,
        y_score,
    )

    return metrics


# ---------------------------------------------------------
# 8. Comparación de modelos sobre el conjunto de test
# ---------------------------------------------------------

def evaluate_models_test(
    models,
    preprocessor,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Entrena y evalúa varios modelos bajo las mismas
    condiciones sobre el conjunto de test.

    Retorna una tabla comparativa con las principales
    métricas de clasificación.
    """
    results = []

    for model_name, model in models.items():
        pipeline = build_model_pipeline(
            clone(preprocessor),
            clone(model),
        )

        metrics = evaluate_model_test(
            pipeline,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        metrics["Model"] = model_name

        results.append(metrics)

    results_df = pd.DataFrame(results)

    ordered_columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "PR-AUC",
    ]

    results_df = results_df[ordered_columns]

    return results_df


# ---------------------------------------------------------
# 9. Selección del mejor modelo
# ---------------------------------------------------------

def select_best_model(cv_results):
    """
    Selecciona el mejor modelo a partir de los resultados
    de validación cruzada.

    El criterio principal es PR-AUC. Como criterios
    complementarios se utilizan ROC-AUC, Recall y F1.
    """
    ranking = cv_results.sort_values(
        by=[
            "pr_auc_mean",
            "roc_auc_mean",
            "recall_mean",
            "f1_mean",
        ],
        ascending=False,
    ).reset_index(drop=True)

    best_model_name = ranking.loc[0, "Model"]
    best_model_metrics = ranking.loc[0].copy()

    return best_model_name, best_model_metrics


# ---------------------------------------------------------
# 10. Visualización comparativa de modelos
# ---------------------------------------------------------

def plot_model_comparison(
    results_df,
    metric_columns,
    title,
):
    """
    Genera un gráfico comparativo de métricas para los
    modelos evaluados.

    Recibe una tabla de resultados, las métricas que se
    desean comparar y el título del gráfico.
    """
    plot_data = results_df[
        ["Model"] + metric_columns
    ].copy()

    plot_data = plot_data.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score",
    )

    fig, ax = plt.subplots(
        figsize=(10, 6),
    )

    sns.barplot(
        data=plot_data,
        x="Model",
        y="Score",
        hue="Metric",
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)

    plt.xticks(
        rotation=15,
        ha="right",
    )

    plt.tight_layout()

    return fig, ax


# ---------------------------------------------------------
# 11. Matriz de confusión
# ---------------------------------------------------------

def get_confusion_matrix(
    pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Entrena el pipeline y calcula la matriz de confusión
    sobre el conjunto de test.
    """
    pipeline.fit(
        X_train,
        y_train,
    )

    y_pred = pipeline.predict(
        X_test,
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
    )

    return matrix


# ---------------------------------------------------------
# 12. Visualización de la matriz de confusión
# ---------------------------------------------------------

def plot_confusion_matrix(
    matrix,
    title="Matriz de confusión",
):
    """
    Genera una visualización de la matriz de confusión.
    """
    fig, ax = plt.subplots(
        figsize=(6, 5),
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")

    ax.set_xticklabels(
        ["No riesgo", "Riesgo"],
    )
    ax.set_yticklabels(
        ["No riesgo", "Riesgo"],
        rotation=0,
    )

    plt.tight_layout()

    return fig, ax


# ---------------------------------------------------------
# 13. Flujo completo de evaluación de modelos
# ---------------------------------------------------------

def run_modeling_workflow(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
):
    """
    Ejecuta el flujo completo de entrenamiento y evaluación
    de los modelos supervisados.
    """
    model_preprocessor = build_model_preprocessor(
        preprocessor
    )

    models = build_models()
    cv = build_cv_strategy()

    cv_results = evaluate_models_cv(
        models,
        model_preprocessor,
        X_train,
        y_train,
        cv,
    )

    test_results = evaluate_models_test(
        models,
        model_preprocessor,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    best_model_name, best_model_metrics = select_best_model(
        cv_results
    )

    best_pipeline = build_model_pipeline(
        clone(model_preprocessor),
        clone(models[best_model_name]),
    )

    matrix = get_confusion_matrix(
        best_pipeline,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    return {
        "cv_results": cv_results,
        "test_results": test_results,
        "best_model_name": best_model_name,
        "best_model_metrics": best_model_metrics,
        "confusion_matrix": matrix,
        "best_pipeline": best_pipeline,
    }


# ---------------------------------------------------------
# 14. Tabla resumen de evaluación
# ---------------------------------------------------------

def build_evaluation_summary(
    cv_results,
    test_results,
):
    """
    Combina las principales métricas de validación cruzada
    y test en una única tabla resumen.
    """
    cv_summary = cv_results[
        [
            "Model",
            "recall_mean",
            "f1_mean",
            "roc_auc_mean",
            "pr_auc_mean",
        ]
    ].copy()

    cv_summary = cv_summary.rename(
        columns={
            "recall_mean": "CV_Recall",
            "f1_mean": "CV_F1",
            "roc_auc_mean": "CV_ROC_AUC",
            "pr_auc_mean": "CV_PR_AUC",
        }
    )

    test_summary = test_results[
        [
            "Model",
            "Recall",
            "F1",
            "ROC-AUC",
            "PR-AUC",
        ]
    ].copy()

    test_summary = test_summary.rename(
        columns={
            "Recall": "Test_Recall",
            "F1": "Test_F1",
            "ROC-AUC": "Test_ROC_AUC",
            "PR-AUC": "Test_PR_AUC",
        }
    )

    summary = cv_summary.merge(
        test_summary,
        on="Model",
    )

    return summary
