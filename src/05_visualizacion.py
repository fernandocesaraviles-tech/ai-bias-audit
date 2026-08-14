"""
05_visualizacion.py

Paso 5: gráfico comparativo final del proyecto. Resume en una sola
imagen los índices de impacto dispar por sexo y por raza, en las tres
etapas medidas: datos crudos, modelo base y modelo mitigado.

Los valores están "hardcodeados" (escritos directamente) porque ya los
obtuvimos y verificamos en los pasos 1 a 4 — no hace falta re-entrenar
nada para graficar.
"""

import matplotlib.pyplot as plt
import numpy as np

etapas = ["Datos crudos", "Modelo base\n(sin mitigar)", "Modelo mitigado"]

# valores obtenidos en los pasos 1, 2, 3 y 4
impacto_sexo = [0.36, 0.299, 0.935]
impacto_raza = [0.435, 0.221, 0.72]

umbral_aceptable = 0.8

x = np.arange(len(etapas))
ancho = 0.35

fig, ax = plt.subplots(figsize=(9, 5.5))

barras_sexo = ax.bar(x - ancho / 2, impacto_sexo, ancho, label="Sexo", color="#4C72B0")
barras_raza = ax.bar(x + ancho / 2, impacto_raza, ancho, label="Raza", color="#DD8452")

ax.axhline(
    y=umbral_aceptable,
    color="green",
    linestyle="--",
    linewidth=1.5,
    label=f"Umbral aceptable ({umbral_aceptable})",
)

ax.set_ylabel("Índice de impacto dispar\n(más cerca de 1.0 = más equitativo)")
ax.set_title("Evolución del sesgo algorítmico a lo largo del proyecto")
ax.set_xticks(x)
ax.set_xticklabels(etapas)
ax.set_ylim(0, 1.05)
ax.legend()

# etiquetas con el valor numérico arriba de cada barra
for barras in (barras_sexo, barras_raza):
    for barra in barras:
        altura = barra.get_height()
        ax.annotate(
            f"{altura:.2f}",
            xy=(barra.get_x() + barra.get_width() / 2, altura),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

plt.tight_layout()
plt.savefig("docs/grafico_impacto_dispar.png", dpi=150)
plt.show()

print("Gráfico guardado en docs/grafico_impacto_dispar.png")
