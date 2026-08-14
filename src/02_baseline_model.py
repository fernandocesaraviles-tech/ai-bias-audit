"""
02_baseline_model.py

Paso 2: entrenar un modelo base (regresión logística) para predecir el
ingreso, y comparar el sesgo en las PREDICCIONES del modelo contra el
sesgo que ya vimos en los datos crudos (paso 1).
"""

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", None)


def load_data() -> pd.DataFrame:
    adult = fetch_openml(name="adult", version=2, as_frame=True)
    df = adult.frame.copy()
    if "class" in df.columns and "income" not in df.columns:
        df = df.rename(columns={"class": "income"})
    df = df.dropna()  # simplificación para el modelo base: descartamos filas con nulos
    return df


def preparar_datos(df: pd.DataFrame):
    y = (df["income"].astype(str).str.contains(">50K")).astype(int)
    X = df.drop(columns=["income"])
    # convertimos columnas categóricas (sex, race, workclass, etc.) a
    # variables numéricas mediante one-hot encoding
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

    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train_scaled, y_train)

    y_pred = modelo.predict(X_test_scaled)

    print("=" * 70)
    print("DESEMPEÑO DEL MODELO BASE")
    print("=" * 70)
    print(f"Precisión (accuracy): {accuracy_score(y_test, y_pred):.3f}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))

    print("\n" + "=" * 70)
    print("SESGO EN LAS PREDICCIONES DEL MODELO (no en los datos crudos)")
    print("=" * 70)

    tasas_sexo_pred = tasa_positiva_prediccion(df_test, y_pred, "sex")
    print("\nTasa de predicción '>50K' por sexo:")
    print(tasas_sexo_pred.round(3))
    print(f"Índice de impacto dispar (sexo, predicciones): {indice_impacto_dispar(tasas_sexo_pred)}")

    tasas_raza_pred = tasa_positiva_prediccion(df_test, y_pred, "race")
    print("\nTasa de predicción '>50K' por raza:")
    print(tasas_raza_pred.round(3))
    print(f"Índice de impacto dispar (raza, predicciones): {indice_impacto_dispar(tasas_raza_pred)}")

    print("\n" + "=" * 70)
    print("COMPARACIÓN: DATOS CRUDOS (paso 1) vs PREDICCIONES DEL MODELO")
    print("=" * 70)
    print("Datos crudos        -> sexo: 0.36   | raza: 0.435")
    print(f"Predicciones modelo -> sexo: {indice_impacto_dispar(tasas_sexo_pred)}  | raza: {indice_impacto_dispar(tasas_raza_pred)}")


if __name__ == "__main__":
    main()
