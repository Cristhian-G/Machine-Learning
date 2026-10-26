from pathlib import Path
# matplotlib se utiliza para mostrar la matriz de confusión.
import matplotlib.pyplot as plt
# pandas permite leer y manipular el archivo CSV.
import pandas as pd
# Estas clases organizan las transformaciones de las columnas.
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
# Clasificador que calcula la probabilidad de supervivencia.
from sklearn.linear_model import LogisticRegression
# Métricas y gráfico para evaluar las predicciones.
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
# Divide los datos en grupos de entrenamiento y prueba.
from sklearn.model_selection import train_test_split
# Pipeline encadena varios pasos y evita aplicar transformaciones incorrectas.
from sklearn.pipeline import Pipeline
# Convierte categorías a números y normaliza las variables numéricas.
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Cargar el archivo desde la misma carpeta que este script.
archivo = Path(__file__).with_name("Titanic-Dataset.csv")
df = pd.read_csv(archivo)

# 2. Separar la variable que queremos predecir (Survived) de las
# variables que usará el modelo para hacer la predicción.
objetivo = "Survived"
X = df.drop(columns=objetivo)
y = df[objetivo]

# Mostrar un resumen inicial de los datos: 1 significa que sobrevivió
# y 0 significa que no sobrevivió.
print("Personas en el dataset:", len(df))
print("Personas que sobrevivieron:", int(y.sum()))
print("Personas que no sobrevivieron:", int((y == 0).sum()))

# Indicar qué columnas contienen números y cuáles contienen texto.
# Age puede tener valores vacíos, por eso se tratará más adelante.
columnas_numericas = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
columnas_categoricas = ["Sex", "Embarked"]

# 3. Preparar las columnas numéricas:
# - Reemplazar edades faltantes por la mediana.
# - Estandarizar los valores para que tengan una escala comparable.
transformador_numerico = Pipeline(
    steps=[
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ]
)

# 4. Preparar las columnas categóricas:
# - Reemplazar textos faltantes por la categoría más frecuente.
# - Crear una columna numérica para cada categoría (one-hot encoding).
transformador_categorico = Pipeline(
    steps=[
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Aplicar cada transformador únicamente a las columnas que le corresponden.
# Esto reúne toda la preparación de los datos en un solo objeto.
preprocesamiento = ColumnTransformer(
    transformers=[
        ("numericas", transformador_numerico, columnas_numericas),
        ("categoricas", transformador_categorico, columnas_categoricas),
    ]
)

# 5. Crear el pipeline final: primero prepara los datos y después
# entrena la regresión logística. max_iter evita detener el entrenamiento
# antes de que el algoritmo encuentre una solución.
modelo = Pipeline(
    steps=[
        ("preprocesamiento", preprocesamiento),
        ("regresion_logistica", LogisticRegression(max_iter=1000)),
    ]
)

# 6. Dividir los datos:
# - 70 % se usa para aprender.
# - 30 % se reserva para evaluar el modelo con datos no vistos.
# random_state hace que la división sea reproducible.
# stratify conserva una proporción similar de supervivientes en ambos grupos.
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Entrenar el modelo con las variables y respuestas conocidas.
modelo.fit(X_entrenamiento, y_entrenamiento)

# Predecir si las personas del conjunto de prueba sobrevivieron.
y_prediccion = modelo.predict(X_prueba)

# 7. Evaluar el rendimiento del modelo.
# accuracy_score calcula el porcentaje total de predicciones correctas.
print("\nResultados del modelo")
print("Exactitud:", f"{accuracy_score(y_prueba, y_prediccion):.2%}")

# classification_report muestra precision, recall y F1-score para
# las clases "no sobrevivió" y "sobrevivió".
print("\nReporte de clasificación:")
print(
    classification_report(
        y_prueba,
        y_prediccion,
        target_names=["No sobrevivió", "Sobrevivió"],
        zero_division=0,
    )
)

# Contar cuántas predicciones del conjunto de prueba indican supervivencia.
print("Predicciones de sobrevivientes en el conjunto de prueba:", int(y_prediccion.sum()))

# 8. Crear una matriz de confusión para comparar las respuestas reales
# con las predicciones del modelo.
ConfusionMatrixDisplay.from_predictions(
    y_prueba,
    y_prediccion,
    display_labels=["No sobrevivió", "Sobrevivió"],
    cmap="Blues",
)
# Ajustar el espacio de la figura y mostrar el gráfico.
plt.title("Matriz de confusión - Regresión logística")
plt.tight_layout()
plt.show()
