"""
03_fairness_audit.py

Paso 3: mitigar el sesgo detectado en el modelo base usando Fairlearn,
aplicando post-procesamiento sobre las predicciones para igualar la tasa
de selección entre hombres y mujeres (constraint: demographic_parity).

IMPORTANTE: en Colab, antes de correr este script, ejecutá en una celda
aparte:
    !pip install fairlearn
"""

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from fairlearn.postprocessing import ThresholdOptimizer

pd.set_option("display.max_columns", None)


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


def tasa_positiva_prediccion(df_test: pd.DataFrame, y_pred, atributo: str) -> pd.Series:
    temp = df_test.copy()
    temp["prediccion"] = y_pred
    return temp.groupby(atributo, observed=True)["prediccion"].mean().sort_values(ascending=False)


def indice_impacto_dispar(tasas: pd.Series) -> float:
    return round(tasas.min() / tasas.max(), 3)


def main():
    df = load_data()
    X, y = preparar_datos(df)

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Modelo base (igual al paso 2) ---
    modelo_base = LogisticRegression(max_iter=1000)
    modelo_base.fit(X_train_scaled, y_train)
    y_pred_base = modelo_base.predict(X_test_scaled)

    # --- Mitigación con Fairlearn ---
    # Ajustamos el umbral de decisión por separado para cada grupo de sexo,
    # buscando que la proporción de predicciones ">50K" sea similar entre
    # hombres y mujeres (demographic parity), sin re-entrenar el modelo.
    mitigador = ThresholdOptimizer(
        estimator=modelo_base,
        constraints="demographic_parity",
        predict_method="predict_proba",
        prefit=True,
    )
    mitigador.fit(X_train_scaled, y_train, sensitive_features=df_train["sex"])
    y_pred_mitigado = mitigador.predict(X_test_scaled, sensitive_features=df_test["sex"])

    print("=" * 70)
    print("ACCURACY: MODELO BASE vs MODELO MITIGADO")
    print("=" * 70)
    print(f"Accuracy modelo base:     {accuracy_score(y_test, y_pred_base):.3f}")
    print(f"Accuracy modelo mitigado: {accuracy_score(y_test, y_pred_mitigado):.3f}")
    print("(es normal y esperable que la accuracy baje un poco: es el")
    print(" costo de reducir el sesgo — la 'tensión precisión vs equidad')")

    tasas_base = tasa_positiva_prediccion(df_test, y_pred_base, "sex")
    tasas_mitigado = tasa_positiva_prediccion(df_test, y_pred_mitigado, "sex")

    print("\n" + "=" * 70)
    print("ÍNDICE DE IMPACTO DISPAR POR SEXO — ANTES vs DESPUÉS DE MITIGAR")
    print("=" * 70)
    print("Datos crudos (paso 1):     0.36")
    print(f"Modelo base (paso 2):      {indice_impacto_dispar(tasas_base)}")
    print(f"Modelo mitigado (paso 3):  {indice_impacto_dispar(tasas_mitigado)}")

    print("\nTasas de predicción '>50K' por sexo, modelo mitigado:")
    print(tasas_mitigado.round(3))


if __name__ == "__main__":
    main()
