# Proyecto Integrador Módulo 5 - MLOps Pipeline

Proyecto Integrador correspondiente al Módulo 5 **Fundamentos de Nube y Ciencia de Datos de Producción** de la carrera de Data Science de Henry.

El proyecto desarrolla un flujo de Machine Learning orientado a un caso de riesgo crediticio, incorporando prácticas de versionamiento, reproducibilidad, modelamiento supervisado, monitoreo de Data Drift y despliegue mediante una API contenerizada.

## Caso de negocio

El proyecto simula el trabajo del equipo de Datos y Analítica de una empresa financiera.

El objetivo de negocio es utilizar información histórica de créditos para anticipar el riesgo de incumplimiento de nuevos usuarios. A partir de esta necesidad se construyó un flujo progresivo que incluye:

- Carga y comprensión de datos;
- Análisis exploratorio;
- Reglas reproducibles de calidad;
- Feature Engineering;
- Entrenamiento y comparación de modelos supervisados;
- Selección de un modelo base;
- Monitoreo temporal;
- Detección de Data Drift;
- Generación de alertas y recomendaciones;
- Visualización mediante Streamlit;
- Exposición del modelo mediante una API con FastAPI;
- Validación de solicitudes y respuestas con Pydantic;
- Soporte para predicción individual y por lotes;
- Documentación automática mediante OpenAPI y Swagger UI;
- Contenerización de la API con Docker;
- Publicación y recuperación de la imagen mediante Docker Hub.

## Objetivo técnico

Construir un proyecto de Machine Learning reproducible y trazable, respetando una arquitectura de archivos predefinida y un flujo de promoción de cambios entre desarrollo, certificación y producción.

La estructura del repositorio no se modifica durante los avances, ya que representa una arquitectura compatible con procesos automatizados de validación y despliegue.

## Estructura del proyecto

```text
mlops_pipeline/
│
├── src/
│   ├── Cargar_datos.ipynb
│   ├── comprension_eda.ipynb
│   ├── ft_engineering.py
│   ├── model_training_evaluation.py
│   ├── model_deploy.py
│   └── model_monitoring.py
│
├── Base_de_datos.csv
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── readme.md
```

## Flujo de trabajo con Git

El proyecto utiliza tres ramas permanentes:

- `developer`: desarrollo y experimentación.
- `certification`: validación de los cambios antes de producción.
- `main`: versión estable del proyecto.

El flujo utilizado es:

```text
developer
    ↓
Pull Request
    ↓
certification
    ↓
Pull Request
    ↓
main
```

Los tags se crean únicamente después de que una versión ha sido certificada e integrada en `main`.

## Datos y análisis exploratorio

El dataset contiene información histórica relacionada con comportamiento crediticio.

La carga validada contiene:

- 10.763 registros;
- 23 variables originales;
- separador `;`;
- información numérica, categórica y temporal.

Durante el EDA se revisaron tipos de datos, valores faltantes, inconsistencias, distribuciones y relaciones entre variables.

Entre las reglas de calidad posteriormente reproducidas en código se encuentran:

- conversión de `fecha_prestamo` a formato datetime;
- conversión de `puntaje` a formato numérico;
- edades mayores a 100 años tratadas como valores faltantes;
- categorías inválidas de `tendencia_ingresos` tratadas como faltantes.

## Feature Engineering

El procesamiento reproducible se encuentra en:

```text
src/ft_engineering.py
```

Se generaron variables temporales:

- `anio_prestamo`;
- `mes_prestamo`;
- `dia_semana_prestamo`;
- `es_fin_semana`.

También se generaron ratios financieros:

- `ratio_cuota_capital`;
- `ratio_otros_prestamos_salario`;
- `ratio_saldo_principal_total`.

La variable objetivo original fue transformada en:

```text
riesgo_incumplimiento
```

donde:

```text
1 = riesgo de incumplimiento
0 = no riesgo
```

El dataset presenta un fuerte desbalance de clases:

```text
No riesgo: 10.252 registros
Riesgo:       511 registros
```

## Preprocesamiento y modelamiento

El preprocesamiento utiliza un `ColumnTransformer` con pipelines diferenciados para variables:

- numéricas;
- nominales;
- ordinales.

El preprocesador se ajusta dentro del pipeline de modelamiento para reducir el riesgo de Data Leakage.

Se evaluaron tres modelos supervisados:

- Logistic Regression;
- Random Forest;
- Gradient Boosting.

La evaluación se realizó mediante `StratifiedKFold` de 5 folds y utilizando:

- Accuracy;
- Precision;
- Recall;
- F1;
- ROC-AUC;
- PR-AUC.

## Variable `puntaje`

Durante el modelamiento se detectó que `puntaje` separaba perfectamente las clases.

Su AUC discriminativa fue:

```text
1.0
```

No se afirmó que esto demostrara Data Leakage, ya que el proyecto no dispone de suficiente información temporal y de negocio para determinar cómo se construye la variable.

Como decisión conservadora, `puntaje` se mantiene en el dataset y en Feature Engineering, pero se excluye del modelamiento principal.

## Modelo base seleccionado

El mejor modelo relativo fue:

```text
Gradient Boosting
```

Resultados principales de Cross Validation:

```text
Recall   ≈ 0.0465
F1       ≈ 0.0836
ROC-AUC  ≈ 0.6920
PR-AUC   ≈ 0.1470
```

Resultados sobre test:

```text
Accuracy  ≈ 0.9531
Precision ≈ 0.5556
Recall    ≈ 0.0490
F1        ≈ 0.0901
ROC-AUC   ≈ 0.7302
PR-AUC    ≈ 0.1856
```

Matriz de confusión:

```text
[[2047,    4],
 [  97,    5]]
```

El modelo se considera un **modelo base**, no un modelo productivo definitivo, debido principalmente a su bajo Recall sobre la clase de riesgo.

## Monitoreo temporal

El monitoreo se implementa en:

```text
src/model_monitoring.py
```

El dataset contiene registros desde:

```text
26/11/2024
```

hasta:

```text
26/04/2026
```

La cantidad de registros disminuye considerablemente hacia el final del dataset. Por este motivo se definió una periodicidad **trimestral**, evitando comparaciones mensuales con tamaños muestrales demasiado pequeños.

La población histórica de referencia quedó definida como:

```text
26/11/2024 → 30/06/2025
8.378 registros
```

Las ventanas completas de monitoreo son:

| Período | Registros |
|---|---:|
| 2025-Q3 | 1.560 |
| 2025-Q4 | 571 |
| 2026-Q1 | 243 |

El período `2026-Q2` contiene solamente 11 registros y está incompleto, por lo que se conserva en la base general pero no se utiliza para emitir conclusiones formales de drift.

## Modelo de referencia temporal

Para simular un escenario de monitoreo productivo, se entrena Gradient Boosting exclusivamente con la población histórica de referencia.

El mismo pipeline y preprocesamiento del Avance 2 son reutilizados, incluida la exclusión conservadora de `puntaje`.

Las ventanas posteriores se utilizan solamente para inferencia y monitoreo:

```text
Referencia histórica
        ↓
Entrenamiento
        ↓
Gradient Boosting
        ↓
2025-Q3
2025-Q4
2026-Q1
        ↓
Predicciones + monitoreo
```

## Métricas de Data Drift

Para variables numéricas se implementaron tres métricas complementarias.

### Kolmogorov-Smirnov

Umbrales utilizados:

```text
KS < 0.15          → Estable
0.15 a 0.25        → Drift moderado
KS > 0.25          → Drift significativo
```

### Population Stability Index

Umbrales utilizados:

```text
PSI < 0.10         → Estable
0.10 a 0.25        → Drift moderado
PSI > 0.25         → Drift significativo
```

### Jensen-Shannon Divergence

Umbrales utilizados:

```text
JSD < 0.10         → Estable
0.10 a 0.30        → Drift moderado
JSD > 0.30         → Drift significativo
```

Para variables categóricas se utiliza:

### Chi-cuadrado

```text
p-value >= 0.05 → sin evidencia estadística de cambio
p-value < 0.05  → evidencia estadística de cambio
```

La significancia estadística no se interpreta automáticamente como magnitud operacional del drift.

## Sistema de alertas

Las métricas numéricas se consolidan para evitar generar alertas críticas a partir de un único indicador aislado.

Los niveles utilizados son:

```text
Verde
Vigilancia
Alerta
Crítica
Informativa
```

Las variables temporales:

```text
anio_prestamo
mes_prestamo
```

se siguen monitoreando, pero sus cambios se consideran temporalmente esperados y no pueden generar por sí solos una alerta operacional.

## Principales hallazgos del monitoreo

La evolución de alertas fue:

| Nivel | 2025-Q3 | 2025-Q4 | 2026-Q1 |
|---|---:|---:|---:|
| Crítica | 1 | 3 | 4 |
| Alerta | 5 | 4 | 3 |
| Vigilancia | 2 | 2 | 2 |
| Informativa | 2 | 2 | 2 |
| Verde | 17 | 16 | 16 |

Las principales señales detectadas fueron:

### `ratio_cuota_capital`

Presenta drift crítico en los tres períodos completos:

```text
2025-Q3 → Crítica
2025-Q4 → Crítica
2026-Q1 → Crítica
```

Se clasifica como **drift crítico persistente**.

### `promedio_ingresos_datacredito`

Se vuelve crítica en `2025-Q4` y mantiene ese estado en `2026-Q1`.

Se clasifica como **drift crítico persistente**.

### `ratio_otros_prestamos_salario`

Presenta comportamiento crítico en `2025-Q4` y `2026-Q1`.

Se clasifica como **drift crítico persistente**.

### `capital_prestado`

Se vuelve crítica únicamente en la última ventana completa, `2026-Q1`.

Se clasifica como **drift crítico reciente**.

Los resultados de las ventanas más recientes deben interpretarse con cautela debido a la disminución del tamaño muestral.

## Pronósticos durante el monitoreo

El modelo de referencia generó:

| Período | Registros | Predichos como riesgo | Tasa predicha | Probabilidad media |
|---|---:|---:|---:|---:|
| 2025-Q3 | 1.560 | 16 | 1,03 % | 5,66 % |
| 2025-Q4 | 571 | 7 | 1,23 % | 7,15 % |
| 2026-Q1 | 243 | 0 | 0,00 % | 5,27 % |

Los cambios en los pronósticos no se interpretan automáticamente como Data Drift ni como degradación de performance.

## Aplicación Streamlit

`model_monitoring.py` también funciona como aplicación Streamlit.

El dashboard permite:

- seleccionar el período de monitoreo;
- visualizar cantidad de variables críticas y en alerta;
- consultar pronósticos del modelo;
- visualizar una muestra de datos junto con predicción y probabilidad;
- consultar la tabla consolidada de métricas de Data Drift;
- comparar distribuciones históricas y actuales de variables numéricas;
- comparar proporciones de variables categóricas;
- observar la evolución temporal de alertas;
- recibir recomendaciones automáticas según persistencia del drift.

Para ejecutarlo:

```powershell
python -m streamlit run src/model_monitoring.py
```

La aplicación se abre localmente en:

```text
http://localhost:8501
```

## Recomendaciones automáticas

El sistema identifica tendencias críticas y genera recomendaciones.

Para drift crítico persistente se recomienda revisar la variable, analizar su impacto sobre las predicciones y evaluar un posible reentrenamiento.

Para drift crítico reciente se recomienda investigar el cambio y comprobar si persiste en la siguiente ventana.

La detección de Data Drift no implica automáticamente que el modelo deba reentrenarse. Idealmente debe complementarse con monitoreo de performance cuando existan resultados reales disponibles.

## Despliegue del modelo

El despliegue se implementa en:
```text
src/model_deploy.py
```
El modelo base seleccionado, Gradient Boosting, se expone mediante una API desarrollada con FastAPI.

La implementación reutiliza las reglas de calidad, el Feature Engineering y el preprocesamiento definidos en los avances anteriores. También mantiene la decisión conservadora de excluir `puntaje` del modelamiento principal.

Al iniciar la aplicación se reconstruye y entrena el pipeline de despliegue utilizando la misma configuración reproducible del Avance 2.

### Contrato de entrada

La API recibe las variables originales necesarias para realizar inferencia.

No se solicitan al cliente:

- la variable objetivo;
- `puntaje`;
- las features temporales derivadas;
- los ratios financieros derivados.

Las transformaciones necesarias se generan internamente antes de ejecutar la predicción.

Pydantic se utiliza para validar tipos, estructura y categorías permitidas.

### Endpoint de predicción

El endpoint principal es:

```text
POST /predict
```

## Instalación

Crear un entorno virtual:

```powershell
python -m venv .venv
```

Activarlo en PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar las dependencias:

```powershell
python -m pip install -r requirements.txt
```

## Tecnologías principales

- Python
- Git
- GitHub
- Jupyter Notebook
- Pandas
- NumPy
- SciPy
- Matplotlib
- Seaborn
- Scikit-learn
- Feature-engine
- Streamlit
- FastAPI
- Pydantic
- Uvicorn
- Docker
- Docker Hub
- WSL 2

## Versionamiento

Versiones estables publicadas hasta el momento:

| Versión | Contenido |
|---|---|
| `V1.0.0` | Estructura inicial del proyecto |
| `V1.0.1` | Carga de datos y EDA |
| `V1.1.0` | Feature Engineering |
| `V1.2.0` | Model Training and Evaluation |
| `V1.3.0` | Model Monitoring and Streamlit Dashboard |

El Avance 3 fue desarrollado en `developer`, validado en `certification` e integrado en `main` mediante Pull Requests.

El Avance 4 se encuentra actualmente desarrollado y validado técnicamente en `developer`. La certificación, integración en `main` y publicación de una nueva versión se realizarán después de completar las verificaciones finales.

## Limitaciones

El proyecto presenta algunas limitaciones que deben considerarse al interpretar los resultados:

- Gradient Boosting mantiene un Recall bajo para la clase de riesgo.
- Data Drift no demuestra por sí solo degradación del desempeño predictivo.
- Las ventanas temporales más recientes tienen menor cantidad de observaciones.
- `2026-Q2` está incompleto y no se utiliza para conclusiones formales.
- Las variables temporales cambian naturalmente al avanzar el calendario.
- La variable `puntaje` presenta separación perfecta de las clases, pero no existe suficiente información de negocio para demostrar Data Leakage.
- El monitoreo realizado simula una operación temporal utilizando el dataset histórico disponible; no corresponde a un flujo real de datos productivos en tiempo real.
- El pipeline de despliegue se reconstruye y entrena al iniciar la API; en un entorno productivo más maduro sería conveniente serializar y versionar el artefacto entrenado para desacoplar entrenamiento e inferencia.
- La API y la imagen Docker representan un despliegue reproducible del modelo base, pero no convierten al modelo en un modelo productivo definitivo; las limitaciones predictivas detectadas anteriormente, especialmente el bajo Recall, continúan vigentes.

## Estado actual del proyecto

Avances completados:

```text
Avance 1 → Versionamiento, carga y EDA
Avance 2 → Feature Engineering y Model Training and Evaluation
Avance 3 → Model Monitoring and Streamlit Dashboard
Avance 4 → FastAPI, Docker y publicación de imagen en Docker Hub
```

Estado del Avance 4:

```text
API, validaciones, contenerización y publicación en Docker Hub completadas en developer.
Pendiente de certificación, integración en main y publicación de nueva versión Git.
```

Última versión estable publicada:

```text
V1.3.0
```