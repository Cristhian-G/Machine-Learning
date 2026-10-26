"""
Practica de Support Vector Machines (SVM) con el dataset Iris.

Se utilizan las caracteristicas petal_length y petal_width, y las clases
setosa y versicolor para poder representar el problema en dos dimensiones.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Cargar Iris desde scikit-learn y conservar solamente dos clases y dos
# caracteristicas (largo y ancho del petalo).
iris = load_iris()
mascara_clases = iris.target < 2
X = iris.data[mascara_clases][:, [2, 3]]
y = iris.target[mascara_clases]
nombre_caracteristicas = ["largo del petalo", "ancho del petalo"]
nombre_clases = iris.target_names[:2]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

# El escalador se ajusta unicamente con entrenamiento para evitar fuga de
# informacion del conjunto de prueba.
modelo_lineal = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear", C=1.0),
)
modelo_lineal.fit(X_train, y_train)
y_pred = modelo_lineal.predict(X_test)

print("=== SVM lineal (C=1.0) ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nReporte de clasificacion:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=nombre_clases,
        zero_division=0,
    )
)
print("Matriz de confusion:")
print(confusion_matrix(y_test, y_pred))
print(
    "Vectores de soporte por clase:",
    modelo_lineal.named_steps["svc"].n_support_,
)
print(
    "Total de vectores de soporte:",
    len(modelo_lineal.named_steps["svc"].support_),
)


def dibujar_regiones_decision(modelo, titulo):
    """Dibuja las regiones de decision en las unidades originales."""
    margen = 0.3
    x_min, x_max = X[:, 0].min() - margen, X[:, 0].max() + margen
    y_min, y_max = X[:, 1].min() - margen, X[:, 1].max() + margen
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )
    puntos = np.c_[xx.ravel(), yy.ravel()]
    Z = modelo.predict(puntos).reshape(xx.shape)

    plt.figure(figsize=(9, 6))
    plt.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
    for clase, color in zip([0, 1], ["royalblue", "darkorange"]):
        seleccion = y == clase
        plt.scatter(
            X[seleccion, 0],
            X[seleccion, 1],
            c=color,
            label=nombre_clases[clase],
            edgecolors="k",
            s=45,
        )

    svc = modelo.named_steps["svc"]
    escalador = modelo.named_steps["standardscaler"]
    vectores_soporte = escalador.inverse_transform(svc.support_vectors_)
    plt.scatter(
        vectores_soporte[:, 0],
        vectores_soporte[:, 1],
        s=150,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="Vectores de soporte",
    )
    plt.xlabel(nombre_caracteristicas[0])
    plt.ylabel(nombre_caracteristicas[1])
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()


dibujar_regiones_decision(modelo_lineal, "SVM lineal: regiones de decision")


# Comparacion solicitada: varios valores de C para cada kernel. Para RBF se
# prueban ademas todos los valores de gamma indicados.
valores_C = [0.01, 0.1, 1, 10, 100]
valores_gamma = [0.01, 0.1, 1, 10]
resultados = []

for kernel in ["linear", "rbf", "poly"]:
    gammas = [None] if kernel != "rbf" else valores_gamma
    for C in valores_C:
        for gamma in gammas:
            parametros = {"kernel": kernel, "C": C}
            if gamma is not None:
                parametros["gamma"] = gamma
            modelo = make_pipeline(StandardScaler(), SVC(**parametros))
            modelo.fit(X_train, y_train)
            predicciones = modelo.predict(X_test)
            svc = modelo.named_steps["svc"]
            resultados.append(
                {
                    "kernel": kernel,
                    "C": C,
                    "gamma": gamma if gamma is not None else "-",
                    "accuracy": accuracy_score(y_test, predicciones),
                    "vectores_soporte": len(svc.support_),
                }
            )

print("\n=== Comparacion de hiperparametros ===")
print("kernel\tC\tgamma\taccuracy\tvectores_soporte")
for resultado in resultados:
    print(
        f"{resultado['kernel']}\t{resultado['C']}\t"
        f"{resultado['gamma']}\t{resultado['accuracy']:.4f}\t"
        f"{resultado['vectores_soporte']}"
    )

mejor_resultado = max(resultados, key=lambda resultado: resultado["accuracy"])
print("\nMejor configuracion en el conjunto de prueba:")
print(mejor_resultado)

plt.show()


# ---------------------------------------------------------------------------
# RESPUESTAS (comentadas)
# ---------------------------------------------------------------------------
# ¿Que es un vector de soporte?
# Es una muestra de entrenamiento cercana a la frontera de decision. Estas
# muestras determinan la posicion del hiperplano y el margen de la SVM.
#
# ¿Que hace el parametro C?
# C controla la penalizacion por clasificar mal una muestra. Un C pequeno
# permite mas errores y produce un margen mas amplio; un C grande penaliza
# mas los errores y procura clasificar correctamente el entrenamiento, aunque
# puede producir un margen mas estrecho y favorecer el sobreajuste.
#
# ¿Que hace gamma?
# Gamma (principalmente en RBF) controla el alcance de influencia de cada
# muestra. Un gamma pequeno produce una frontera mas suave; un gamma grande
# crea una frontera mas local y compleja, con mayor riesgo de sobreajuste.
#
# ¿Por que escalar los datos?
# Porque SVM usa distancias y productos internos. Si una caracteristica tiene
# una escala numerica mayor, dominaria el calculo de la frontera. StandardScaler
# deja las caracteristicas en una escala comparable y se ajusta solo con train.
#
# ¿Que kernel funciono mejor y por que?
# En esta ejecucion hubo empate: linear, RBF (con varios valores de gamma) y
# polynomial alcanzaron accuracy de 1.00 en prueba. Por eso se elige linear
# como mejor opcion practica: logra la misma precision con una frontera simple,
# facil de interpretar y sin depender de gamma. RBF y polynomial son utiles si
# los datos no pueden separarse bien con una frontera lineal.