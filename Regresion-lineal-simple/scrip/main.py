from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Paso 1: Cargar el archivo CSV
try:
    base_dir = Path(__file__).resolve().parent.parent
except NameError:
    base_dir = Path.cwd()

ruta_csv = base_dir / "data" / "datos_cancer_purgados.csv"
datos = pd.read_csv(ruta_csv)

# Paso 2: Definición de variables predictora (X) y objetivo (y)
X = datos[["radius_mean"]]
y = datos["area_mean"]

# Paso 3: Creación y entrenamiento del modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Paso 4: Extracción de métricas
coeficiente = modelo.coef_[0]
intercepto = modelo.intercept_
r2 = modelo.score(X, y)

print("=" * 45)
print("        RESULTADOS DEL MODELO DE REGRESIÓN       ")
print("=" * 45)
print(f"Coeficiente (pendiente): {coeficiente:.4f}")
print(f"Intercepto:              {intercepto:.4f}")
print(f"Coeficiente R²:          {r2:.4f}")
print(f"Ecuación: area_mean = {coeficiente:.4f} * radius_mean + ({intercepto:.4f})")
print("=" * 45)

# Paso 5: Interpretación de los resultados
print("\nINTERPRETACIÓN DE RESULTADOS:")
print(f"1. Coeficiente: Por cada incremento de 1 unidad en 'radius_mean', "
      f"el 'area_mean' aumenta en promedios {coeficiente:.2f} unidades.")
print(f"2. Calidad de Ajuste (R²): El {r2 * 100:.2f}% de la variabilidad del área "
      f"es explicada linealmente por el radio.")
print("3. Observación Teórica: Dado que el área geométrica sigue A = π*r², la relación real es cuadrática.")
print("   La regresión lineal funciona bien localmente en este rango, pero el intercepto negativo es un artefacto")
print("   de la linealización.")

# Paso 6: Generación de la gráfica
x_recta = pd.DataFrame({
    "radius_mean": [X["radius_mean"].min(), X["radius_mean"].max()]
})
y_recta = modelo.predict(x_recta)

plt.figure(figsize=(9, 5.5))

plt.scatter(
    X["radius_mean"],
    y,
    color="steelblue",
    alpha=0.6,
    edgecolors="none",
    label="Datos reales"
)

plt.plot(
    x_recta["radius_mean"],
    y_recta,
    color="firebrick",
    linewidth=2.5,
    label=f"Recta: y = {coeficiente:.2f}x + ({intercepto:.2f})"
)

plt.title("Regresión Lineal Simple: Radio vs Área del Tumor", fontsize=12, fontweight="bold")
plt.xlabel("Radio Promedio (radius_mean)", fontsize=10)
plt.ylabel("Área Promedio (area_mean)", fontsize=10)
plt.legend(loc="upper left")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

plt.show()