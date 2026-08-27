import pandas as pd
import pytest

import src.model_monitoring as mm


def test_monitoring_constants():
    assert mm.REFERENCE_PERIOD == "Referencia histórica"
    assert mm.DRIFT_MODERATE == "Drift moderado"
    assert mm.DRIFT_SIGNIFICANT == "Drift significativo"
    assert mm.EXPECTED_TEMPORAL_CHANGE == "Cambio temporal esperado"
    assert mm.EVALUABLE_POPULATION_DRIFT == "Drift poblacional evaluable"
    assert mm.CRITICAL_ALERT == "Crítica"
    assert mm.PERSISTENT_CRITICAL_DRIFT == "Drift crítico persistente"
    assert mm.RECENT_CRITICAL_DRIFT == "Drift crítico reciente"


@pytest.mark.parametrize(
    "classifier,value,expected",
    [
        (mm.classify_ks_drift, 0.20, mm.DRIFT_MODERATE),
        (mm.classify_ks_drift, 0.30, mm.DRIFT_SIGNIFICANT),
        (mm.classify_psi_drift, 0.20, mm.DRIFT_MODERATE),
        (mm.classify_psi_drift, 0.30, mm.DRIFT_SIGNIFICANT),
        (mm.classify_js_drift, 0.20, mm.DRIFT_MODERATE),
        (mm.classify_js_drift, 0.35, mm.DRIFT_SIGNIFICANT),
    ],
)
def test_drift_classifiers(classifier, value, expected):
    assert classifier(value) == expected


def test_assign_monitoring_period_uses_reference_constant():
    df = pd.DataFrame(
        {
            "fecha_prestamo": pd.to_datetime(
                [
                    "2025-06-30",
                    "2025-07-01",
                    "2025-10-01",
                    "2026-01-01",
                    "2026-04-01",
                ]
            )
        }
    )

    result = mm.assign_monitoring_period(df)

    assert result.loc[0, "periodo_monitoreo"] == mm.REFERENCE_PERIOD
    assert result.loc[1, "periodo_monitoreo"] == "2025-Q3"
    assert result.loc[2, "periodo_monitoreo"] == "2025-Q4"
    assert result.loc[3, "periodo_monitoreo"] == "2026-Q1"
    assert result.loc[4, "periodo_monitoreo"] == "2026-Q2 incompleto"


def test_numeric_drift_contexts_use_constants(monkeypatch):
    monkeypatch.setattr(
        mm,
        "NUMERIC_FEATURES",
        ["anio_prestamo", "capital_prestado"],
    )

    reference_df = pd.DataFrame(
        {
            "anio_prestamo": [2024, 2024, 2025, 2025],
            "capital_prestado": [100, 200, 300, 400],
        }
    )

    current_df = pd.DataFrame(
        {
            "anio_prestamo": [2025, 2025, 2026, 2026],
            "capital_prestado": [150, 250, 350, 450],
        }
    )

    evaluators = [
        mm.evaluate_numeric_ks_drift,
        mm.evaluate_numeric_psi_drift,
        mm.evaluate_numeric_js_drift,
    ]

    for evaluator in evaluators:
        result = evaluator(reference_df, current_df).set_index("feature")

        assert (
            result.loc["anio_prestamo", "drift_context"]
            == mm.EXPECTED_TEMPORAL_CHANGE
        )
        assert not bool(
            result.loc["anio_prestamo", "alert_eligible"]
        )

        assert (
            result.loc["capital_prestado", "drift_context"]
            == mm.EVALUABLE_POPULATION_DRIFT
        )
        assert bool(
            result.loc["capital_prestado", "alert_eligible"]
        )


def test_assign_numeric_drift_alerts_uses_constants():
    summary_df = pd.DataFrame(
        [
            {
                "ks_status": mm.DRIFT_SIGNIFICANT,
                "psi_status": mm.DRIFT_SIGNIFICANT,
                "js_status": mm.DRIFT_MODERATE,
                "alert_eligible": True,
            }
        ]
    )

    result = mm.assign_numeric_drift_alerts(summary_df)

    assert result.loc[0, "significant_count"] == 2
    assert result.loc[0, "moderate_count"] == 1
    assert result.loc[0, "alert_level"] == mm.CRITICAL_ALERT


def _build_temporal_summary():
    rows = []

    for period in mm.COMPLETE_MONITORING_PERIODS:
        rows.append(
            {
                "feature": "variable_persistente",
                "periodo_monitoreo": period,
                "alert_level": (
                    mm.CRITICAL_ALERT
                    if period in ["2025-Q3", "2025-Q4"]
                    else "Verde"
                ),
            }
        )

        rows.append(
            {
                "feature": "variable_reciente",
                "periodo_monitoreo": period,
                "alert_level": (
                    mm.CRITICAL_ALERT
                    if period == "2026-Q1"
                    else "Verde"
                ),
            }
        )

    return pd.DataFrame(rows)


def test_identify_critical_drift_trends_uses_constants():
    temporal_summary = _build_temporal_summary()

    result = mm.identify_critical_drift_trends(
        temporal_summary
    ).set_index("feature")

    assert (
        result.loc["variable_persistente", "trend_status"]
        == mm.PERSISTENT_CRITICAL_DRIFT
    )

    assert (
        result.loc["variable_reciente", "trend_status"]
        == mm.RECENT_CRITICAL_DRIFT
    )


def test_monitoring_recommendations_for_critical_trends():
    temporal_summary = _build_temporal_summary()

    result = mm.build_monitoring_recommendations(
        temporal_summary
    ).set_index("feature")

    assert (
        result.loc["variable_persistente", "trend_status"]
        == mm.PERSISTENT_CRITICAL_DRIFT
    )
    assert "Prioridad alta" in result.loc[
        "variable_persistente",
        "recommendation",
    ]

    assert (
        result.loc["variable_reciente", "trend_status"]
        == mm.RECENT_CRITICAL_DRIFT
    )
    assert "Prioridad media-alta" in result.loc[
        "variable_reciente",
        "recommendation",
    ]
    