# Informe de mitigación de sesgos — Modelo de predicción de ingresos

**Proyecto:** Auditoría de sesgos en un modelo de predicción de ingresos (Adult Income)
**Estándar de referencia:** ISO/IEC 42001 — Gestión de riesgos de IA
**Fecha:** Agosto 2026

## 1. Hallazgos principales

### 1.1 Sesgo presente en los datos crudos

Antes de entrenar cualquier modelo, se midió el **índice de impacto
dispar** (disparate impact ratio) entre grupos, comparando la proporción
de personas con ingreso ">50K" en cada grupo. Un valor por debajo de 0.8
se considera señal de alerta (regla del 80%, EEOC).

| Atributo sensible | Índice de impacto dispar (datos crudos) |
|---|---|
| Sexo | 0.36 |
| Raza | 0.435 |

Ambos valores están muy por debajo del umbral de 0.8, confirmando que el
dataset ya contiene una disparidad estructural entre grupos, previa a
cualquier intervención algorítmica.

### 1.2 El modelo base amplificó el sesgo existente

Se entrenó un modelo de regresión logística sin ninguna corrección de
equidad (accuracy: 84.6%). Al medir el impacto dispar sobre sus
**predicciones**, el sesgo no se mantuvo igual: **empeoró**.

| Atributo sensible | Datos crudos | Predicciones del modelo base |
|---|---|---|
| Sexo | 0.36 | 0.299 |
| Raza | 0.435 | 0.221 |

Esto confirma el riesgo descrito en la bibliografía del curso: un modelo
entrenado sin controles de equidad no solo reproduce el sesgo de sus
datos de entrenamiento, sino que puede **amplificarlo**, especialmente
en el atributo de raza (el impacto dispar casi se redujo a la mitad).

## 2. Acciones correctivas

Se aplicó una técnica de **mitigación por post-procesamiento** utilizando
la librería Fairlearn (`ThresholdOptimizer`, restricción
`demographic_parity`). El método ajusta el umbral de decisión de forma
independiente para cada grupo de sexo, sin modificar el modelo entrenado
ni los datos originales, buscando igualar la tasa de predicciones
positivas entre grupos.

| Métrica | Modelo base | Modelo mitigado |
|---|---|---|
| Accuracy | 84.6% | 83.2% |
| Índice de impacto dispar (sexo) | 0.299 | **0.935** |

**Costo de la mitigación:** 1.4 puntos porcentuales de accuracy.
**Beneficio:** el impacto dispar pasó de una zona de alto riesgo (0.299)
a estar dentro del umbral aceptable (0.935, muy cercano a 1.0 — equidad
perfecta).

Esta relación costo/beneficio respalda la recomendación de aplicar la
mitigación en un eventual despliegue: la pérdida de desempeño es
marginal frente a la reducción sustancial del riesgo de discriminación.

*Nota de alcance: la mitigación aplicada corrige la disparidad por sexo.
El atributo de raza (impacto dispar 0.221 en el modelo base) queda
identificado como pendiente de mitigación en una iteración futura del
proyecto.*

## 3. Controles preventivos (gobernanza, alineados a ISO 42001 Anexo A)

- **A.7.2 — Calidad de datos:** documentar en cada versión del dataset
  los índices de impacto dispar de origen, antes de cualquier
  entrenamiento, como parte del proceso de ingesta.
- **A.8.1 — Datos diversos:** evaluar en futuras iteraciones si ampliar
  o rebalancear las fuentes de datos reduce el sesgo de origen (0.36 /
  0.435), en lugar de corregir únicamente en la etapa de predicción.
- **A.8.5 — Equidad:** incorporar el cálculo del índice de impacto
  dispar como métrica obligatoria de evaluación en cada reentrenamiento
  del modelo (no solo accuracy/F1), con un umbral mínimo aceptable de
  0.8.
- **A.9 — Transparencia:** dejar registrado en el repositorio (este
  documento) el trade-off accuracy/equidad de cada versión del modelo,
  para que cualquier decisión de despliegue sea informada.
- **Monitoreo continuo:** en producción, medir periódicamente el
  impacto dispar sobre datos nuevos (deriva de datos / *data drift*),
  ya que la disparidad podría reaparecer si la población de entrada
  cambia con el tiempo.

## 4. Conclusión

El caso demuestra de forma medible que (1) el sesgo en los datos de
entrenamiento se traslada y puede amplificarse en las predicciones de un
modelo sin controles, y (2) que existen técnicas de mitigación (post
-procesamiento con Fairlearn) capaces de corregir gran parte de esa
disparidad con un costo de desempeño bajo. Se recomienda extender el
análisis al atributo de raza y evaluar mitigación en la etapa de datos
(pre-procesamiento) como complemento.
