from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1 Cargar el data set
archivo = Path(__file__).with_name('IrisFlowers.csv')
df = pd.read_csv(archivo)

print("Primeras filas del dataset:")
print(df.head())
print(f"Registros encontrados en el archivo: {len(df)}")

# 2 Preparar los datos para obtener las caracteristicas y el objetivo.
caracteristicas = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

X = df[caracteristicas]
y = df["species"]

print("caracteristicas:")
print(X.head())
print("objetivo:")
print(y.head())

# 3 Separamos los datos en una proporción de 70 - 30 para entrenamiento y pruebas.
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify = y,
)

# 4 Creamos el modelo de KNN con un pipeline que incluye las transformaciones necesarias.
modelo = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=3)
)

# 5 Entrenamos el modelo con los datos de entrenamiento.
modelo.fit(X_entrenamiento, y_entrenamiento)

# 6 Realizamos predicciones con los datos de prueba.
predicciones = modelo.predict(X_prueba)

exactitud = accuracy_score(y_prueba, predicciones)

print(f"Exactitud del modelo: {exactitud:.2f}")
print("Reporte de clasificación:")
print(classification_report(y_prueba, predicciones))

# 7 Intentaremos clasificar una nueva flor con características específicas.
nueva_flor = [[5.1, 3.5, 1.4, 0.2]]
nueva_flor = pd.DataFrame(
    data=nueva_flor,
    columns=caracteristicas,
)

prediccion = modelo.predict(nueva_flor)[0]
probabilidades = modelo.predict_proba(nueva_flor)[0]
clases = modelo.named_steps['kneighborsclassifier'].classes_
confianza = probabilidades.max()

# mostrar el resultado de la predicción
print("Datos de la nueva flor:")
print(nueva_flor)
print(f"Predicción de la especie: {prediccion}")
print("Probabilidades de cada especie:")
for especie, probabilidade in zip(clases, probabilidades):
    print(f"{especie}: {probabilidade:.2f}")
print(f"Confianza en la predicción: {confianza:.2f}")

# 8 Visualizamos los datos y la predicción en un gráfico de dispersión.
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Gráfico 1: Matriz de confusión
ConfusionMatrixDisplay.from_predictions(
    y_prueba,
    predicciones,
    ax=axes[0],
    cmap='Blues',
)

axes[0].set_title("Matriz de Confusión")

# Gráfico 2: datos del dataset y nueva flor
colores = {
    "setosa": "blue",
    "versicolor": "orange",
    "virginica": "green",
}

for especie, color in colores.items():
    datos_especie = df[df["species"] == especie]

    axes[1].scatter(
        datos_especie["petal_length"],
        datos_especie["petal_width"],
        label=especie,
        color=color,
        alpha=0.7,
    )

axes[1].scatter(
    nueva_flor["petal_length"],
    nueva_flor["petal_width"],
    marker="*",
    s=350,
    color="red",
    edgecolors="black",
    label=f"Nueva flor: {prediccion}",
)

axes[1].set_title("Clasificación de la nueva flor")
axes[1].set_xlabel("Longitud del pétalo")
axes[1].set_ylabel("Ancho del pétalo")
axes[1].legend()
axes[1].grid(alpha=0.3)


# Gráfico 3: probabilidades de la nueva flor
axes[2].bar(
    clases,
    probabilidades,
    color=[colores[especie] for especie in clases],
)

axes[2].set_title("Probabilidades de clasificación")
axes[2].set_xlabel("Especie")
axes[2].set_ylabel("Probabilidad")
axes[2].set_ylim(0, 1)
axes[2].grid(axis="y", alpha=0.3)

for indice, probabilidad in enumerate(probabilidades):
    axes[2].text(
        indice,
        probabilidad + 0.03,
        f"{probabilidad:.2f}",
        ha="center",
    )


plt.tight_layout()
plt.show()
