# Auditoría de sesgos en un modelo de predicción de ingresos

Proyecto de práctica para el curso **Experto en Arquitectura y Desarrollo de
Inteligencia Artificial** (Fundación Educativa Santísima Trinidad).

## Objetivo

Entrenar un modelo simple de Machine Learning que prediga si una persona
gana más o menos de 50.000 USD/año (dataset **Adult Income / UCI Census**),
y aplicar sobre él el proceso de identificación y mitigación de sesgos
algorítmicos visto en el módulo de ética de IA, siguiendo criterios
alineados a **ISO/IEC 42001** (gestión responsable de sistemas de IA).

## Por qué este dataset

Es un caso clásico y bien documentado para practicar *fairness* en ML:
tiene atributos sensibles (sexo, raza, edad) con desbalance conocido en la
variable objetivo, es chico (~48.000 filas) y corre rápido en cualquier
notebook.

## Estructura del repo

```
ai-bias-audit/
├── data/               # datasets crudos y procesados (no versionados en git)
├── docs/               # ficha del proyecto, informe final de mitigación
├── notebooks/          # exploración interactiva (opcional)
├── src/
│   ├── 01_load_and_explore.py   # carga + primera detección de sesgo en datos crudos
│   ├── 02_baseline_model.py     # (próximo paso) modelo base + métricas de desempeño
│   ├── 03_fairness_audit.py     # (próximo paso) métricas de equidad + mitigación
│   └── 04_monitor_agent.py      # (próximo paso) agente simple de monitoreo de deriva
├── requirements.txt
└── README.md
```

## Cómo correrlo

```bash
python -m venv venv
source venv/bin/activate    # en Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/01_load_and_explore.py
```

> La primera vez que se ejecuta, `01_load_and_explore.py` descarga el
> dataset desde OpenML (necesita conexión a internet) y lo cachea. Las
> corridas siguientes usan el archivo guardado en `data/`.

## Estado del proyecto

- [x] Paso 1 — Carga de datos y primera detección de sesgo (`índice de
      impacto dispar`: 0.36 sexo / 0.435 raza)
- [x] Paso 2 — Modelo base (regresión logística, accuracy 84.6%) +
      confirmación de que el sesgo se amplifica en las predicciones
      (0.299 sexo / 0.221 raza)
- [x] Paso 3 — Mitigación de sesgo por sexo con Fairlearn
      (`ThresholdOptimizer`, demographic parity): impacto dispar 0.299 → 0.935
      con costo de solo 1.4 puntos de accuracy
- [x] Informe de mitigación estilo ISO 42001 (`docs/informe_mitigacion.md`)
- [ ] Paso 4 — Mitigar también el sesgo por raza (pendiente)
- [ ] Paso 5 — Agente simple de monitoreo de deriva del modelo
- [ ] Paso 6 — Dashboard de evaluación final

## Referencias

- UCI Machine Learning Repository — Adult Data Set
- IBM AI Fairness 360
- ISO/IEC 42001:2023 — Sistemas de Gestión de Inteligencia Artificial
