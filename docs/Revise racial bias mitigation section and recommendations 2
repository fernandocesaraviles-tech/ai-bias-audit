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
El atributo de raza (impacto dispar 0.221 en el modelo base) se abordó
en una segunda iteración, documentada en la sección 2.1.*

### 2.1 Segunda iteración: mitigación del sesgo por raza

Se repitió el mismo procedimiento (`ThresholdOptimizer`, demographic
parity) usando `race` como atributo sensible. A diferencia de sexo, raza
tiene 5 categorías, varias de ellas con pocos casos en el dataset.

| Métrica | Modelo base | Modelo mitigado (raza) |
|---|---|---|
| Accuracy | 84.6% | 84.2% |
| Índice de impacto dispar (raza) | 0.221 | **0.72** |

**Resultado parcial:** la mejora es considerable (0.221 → 0.72) con un
costo de accuracy mínimo (0.4 puntos), pero el valor final queda **por
debajo del umbral de 0.8**, a diferencia de la mitigación por sexo que
sí lo superó (0.935).

**Hallazgo relevante — sobrecorrección en grupos pequeños:** el grupo
que tenía la tasa más baja en el modelo base (`Amer-Indian-Eskimo`,
0.089) pasó a tener la tasa más alta tras la mitigación (0.278),
invirtiendo el orden respecto a `Asian-Pac-Islander`, que era el grupo
mejor posicionado. Esto sugiere que el ajuste de umbral es menos estable
cuando el grupo tiene pocos casos en los datos de entrenamiento: con
poca muestra, el optimizador puede sobrecorregir. Esto es consistente
con lo visto en el módulo de ética sobre la relación entre
representación de datos y calidad de la mitigación (a más categorías y
menor tamaño de grupo, mayor riesgo de un ajuste ruidoso).

**Recomendación:** antes de considerar este resultado listo para
producción, evaluar técnicas de mitigación en la etapa de datos (por
ejemplo, sobremuestreo de los grupos minoritarios) en lugar de depender
únicamente del post-procesamiento, y repetir la medición con más datos
o con validación cruzada para confirmar que el resultado es estable y
no un artefacto de la partición de datos usada.

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
- **A.7.2 — Representación de subgrupos:** para atributos con múltiples
  categorías y grupos pequeños (como raza en este dataset), evaluar el
  tamaño mínimo de muestra por grupo antes de aplicar mitigación por
  post-procesamiento, dado el riesgo observado de sobrecorrección.

## 4. Conclusión

El caso demuestra de forma medible que (1) el sesgo en los datos de
entrenamiento se traslada y puede amplificarse en las predicciones de un
modelo sin controles, (2) que existen técnicas de mitigación (post
-procesamiento con Fairlearn) capaces de corregir gran parte de esa
disparidad con un costo de desempeño bajo, y (3) que la efectividad de
esa mitigación depende del tamaño de los grupos involucrados: funcionó
casi a la perfección con sexo (2 grupos grandes, 0.299 → 0.935) pero de
forma parcial e inestable con raza (5 grupos, algunos pequeños,
0.221 → 0.72, con sobrecorrección en el grupo más chico). Se recomienda
complementar con mitigación en la etapa de datos para los grupos
minoritarios antes de un eventual despliegue.
