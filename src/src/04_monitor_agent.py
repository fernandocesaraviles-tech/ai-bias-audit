"""
04_monitor_agent.py

Paso 6: agente simple de monitoreo del modelo en producción.

Un agente inteligente, en su forma más simple (agente reactivo basado en
reglas), es un sistema que:
    1. Percibe su entorno (en este caso, nuevos lotes de datos que llegan)
    2. Evalúa esa percepción contra reglas o umbrales
    3. Decide y ejecuta una acción (en este caso, emitir una alerta)

Este agente automatiza el "Monitoreo continuo" descrito en el informe de
mitigación (docs/informe_mitigacion.md) y en el módulo de ISO 42001:
revisa si el modelo sigue funcionando bien (accuracy) y si sigue siendo
equitativo (índice de impacto dispar), y decide si hace falta
reentrenar.

Resultado de la corrida de referencia (4 lotes simulados): ningún lote
disparó alerta, con accuracy entre 0.819-0.836 e impacto dispar por sexo
entre 0.887-0.982 — el modelo mitigado se mantiene estable.

IMPORTANTE: en Colab, antes de correr este script, ejecutá en una celda
aparte:
    !pip install fairlearn
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from fairlearn.postprocessing import ThresholdOptimizer

pd.set_option("display.max_columns", None)

UMBRAL_ACCURACY_MINIMO = 0.80
UMBRAL_IMPACTO_DISPAR_MINIMO = 0.80


def load_data() -> pd.DataFrame:
    adult = fetch_openml(name="adult", version=2, as_frame=True)
    df = adult.frame.copy()
    if "class" in df.columns and "income" not in df.columns:
        df = df.rename(columns={"class": "income"})
    df = df.dropna()
    return df


def preparar_datos(df: pd.DataFrame):
    y = (df["income"].astype(str).str.contains(">50K")).astype(int)
    X = df.drop(columns=["income"])
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def indice_impacto_dispar(df_batch: pd.DataFrame, y_pred, atributo: str) -> float:
    temp = df_batch.copy()
    temp["prediccion"] = y_pred
    tasas = temp.groupby(atributo, observed=True)["prediccion"].mean()
    return round(tasas.min() / tasas.max(), 3)


class AgenteMonitoreo:
    """
    Agente reactivo simple: percibe un lote (batch) de datos nuevos,
    evalúa el desempeño y la equidad del modelo sobre ese lote, y decide
    una acción según reglas fijas (umbrales).
    """

    def __init__(self, mitigador, scaler):
        self.mitigador = mitigador
        self.scaler = scaler
        self.historial = []

    def percibir_y_evaluar(self, X_batch, y_batch, df_batch, id_lote):
        X_batch_scaled = self.scaler.transform(X_batch)
        y_pred = self.mitigador.predict(X_batch_scaled, sensitive_features=df_batch["sex"])

        resultado = {
            "lote": id_lote,
            "accuracy": accuracy_score(y_batch, y_pred),
            "impacto_dispar_sexo": indice_impacto_dispar(df_batch, y_pred, "sex"),
        }
        self.historial.append(resultado)
        return resultado

    def decidir_accion(self, resultado) -> str:
        alertas = []
        if resultado["accuracy"] < UMBRAL_ACCURACY_MINIMO:
            alertas.append(
                f"accuracy {resultado['accuracy']:.3f} por debajo del mínimo "
                f"({UMBRAL_ACCURACY_MINIMO})"
            )
        if resultado["impacto_dispar_sexo"] < UMBRAL_IMPACTO_DISPAR_MINIMO:
            alertas.append(
                f"impacto dispar por sexo {resultado['impacto_dispar_sexo']} "
                f"por debajo del mínimo ({UMBRAL_IMPACTO_DISPAR_MINIMO})"
            )

        if alertas:
            return "🚨 ALERTA: revisar/reentrenar el modelo -> " + "; ".join(alertas)
        return "✅ OK: el modelo sigue dentro de los umbrales aceptables"

    def reporte_final(self):
        print("\n" + "=" * 70)
        print("RESUMEN DEL MONITOREO (historial del agente)")
        print("=" * 70)
        print(pd.DataFrame(self.historial))


def main():
    df = load_data()
    X, y = preparar_datos(df)

    X_train, X_resto, y_train, y_resto, df_train, df_resto = train_test_split(
        X, y, df, test_size=0.4, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    modelo_base = LogisticRegression(max_iter=1000)
    modelo_base.fit(X_train_scaled, y_train)

    mitigador = ThresholdOptimizer(
        estimator=modelo_base,
        constraints="demographic_parity",
        predict_method="predict_proba",
        prefit=True,
    )
    mitigador.fit(X_train_scaled, y_train, sensitive_features=df_train["sex"])

    agente = AgenteMonitoreo(mitigador, scaler)

    n_lotes = 4
    indices = np.array_split(df_resto.index, n_lotes)

    print("=" * 70)
    print("AGENTE DE MONITOREO — simulación de 4 lotes de datos nuevos")
    print("=" * 70)

    for i, idx in enumerate(indices, start=1):
        X_lote = X_resto.loc[idx]
        y_lote = y_resto.loc[idx]
        df_lote = df_resto.loc[idx]

        resultado = agente.percibir_y_evaluar(X_lote, y_lote, df_lote, id_lote=i)
        accion = agente.decidir_accion(resultado)

        print(f"\nLote {i}:")
        print(f"  accuracy = {resultado['accuracy']:.3f}")
        print(f"  impacto dispar (sexo) = {resultado['impacto_dispar_sexo']}")
        print(f"  Decisión del agente: {accion}")

    agente.reporte_final()


if __name__ == "__main__":
    main()
