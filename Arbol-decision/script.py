"""
Árbol de decisión CART implementado desde cero para clasificar flores Iris.

El programa realiza el flujo de aprendizaje supervisado:
1. Lee el archivo CSV y convierte las clases de texto a números.
2. Divide los datos en entrenamiento y prueba.
3. Cons truye un árbol usando el índice de impureza Gini.
4. Predice las clases de las muestras de prueba.
5. Calcula métricas y genera gráficos para interpretar el resultado.

No se utiliza un clasificador de scikit-learn; la construcción del árbol y
las métricas se implementan manualmente con NumPy y pandas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================================================================
# 1. CARGA Y PREPROCESAMIENTO
# ==============================================================================
def cargar_y_preparar_datos(ruta_archivo='IrisFlowers.csv'):
    """Lee el CSV y separa las características de la clase que se predecirá.

    Args:
        ruta_archivo: Ruta del archivo CSV. La columna ``species`` se usa como
            objetivo si existe; de lo contrario, se toma la última columna.

    Returns:
        Una tupla con:
        - ``X``: matriz NumPy con las medidas de las flores.
        - ``y``: clases convertidas a enteros (0, 1, 2, ...).
        - nombres de las columnas usadas como características.
        - nombres originales de las clases, en el mismo orden que la
          codificación numérica.
    """
    df = pd.read_csv(ruta_archivo)

    # La variable objetivo (target) es aquello que el árbol debe aprender
    # a predecir: en este caso, la especie de la flor.
    columna_target = 'species' if 'species' in df.columns else df.columns[-1]

    X_df = df.drop(columns=[columna_target])
    y_df = df[columna_target]

    # np.unique devuelve las clases ordenadas y, mediante return_inverse,
    # el número de clase correspondiente a cada fila.
    clases_unicas, y_encoded = np.unique(y_df, return_inverse=True)

    X = X_df.to_numpy()
    y = y_encoded

    return X, y, list(X_df.columns), clases_unicas


# ==============================================================================
# 2. ÍNDICE DE GINI Y MEJOR SPLIT
# ==============================================================================
def calcular_gini(y):
    """Calcula la impureza Gini de un conjunto de etiquetas.

    Gini vale 0 cuando todas las muestras pertenecen a una sola clase.
    Cuanto más mezcladas estén las clases, mayor será su valor. La fórmula es
    ``1 - suma(probabilidad_de_clase^2)``.
    """
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilidades = counts / len(y)
    return 1.0 - np.sum(probabilidades ** 2)


def mejor_split(X, y):
    """Busca la división que deja los grupos hijos menos mezclados.

    Para cada característica se prueban umbrales entre valores consecutivos.
    Cada umbral divide las muestras en:
    - izquierda: valor de la característica <= umbral;
    - derecha: valor de la característica > umbral.

    La división elegida es la que minimiza el Gini ponderado por el número de
    muestras de cada grupo.
    """
    n_samples, n_features = X.shape
    if n_samples <= 1:
        return None, None, 0.0

    mejor_gini_ponderado = float('inf')
    mejor_feature = None
    mejor_umbral = None

    for feature_idx in range(n_features):
        valores = X[:, feature_idx]
        umbrales_unicos = np.unique(valores)

        # Un punto medio evita colocar el umbral exactamente sobre una
        # observación y es suficiente para separar valores consecutivos.
        if len(umbrales_unicos) > 1:
            umbrales_candidatos = (umbrales_unicos[:-1] + umbrales_unicos[1:]) / 2.0
        else:
            umbrales_candidatos = umbrales_unicos

        for umbral in umbrales_candidatos:
            left_mask = valores <= umbral
            right_mask = ~left_mask

            y_left, y_right = y[left_mask], y[right_mask]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            n_left, n_right = len(y_left), len(y_right)
            gini_left = calcular_gini(y_left)
            gini_right = calcular_gini(y_right)

            # Un grupo grande debe influir más en la calidad total del split
            # que un grupo pequeño.
            gini_ponderado = (n_left / n_samples) * gini_left + (n_right / n_samples) * gini_right

            if gini_ponderado < mejor_gini_ponderado:
                mejor_gini_ponderado = gini_ponderado
                mejor_feature = feature_idx
                mejor_umbral = umbral

    return mejor_feature, mejor_umbral, mejor_gini_ponderado


# ==============================================================================
# 3. NODO Y CONSTRUCCIÓN RECURSIVA DEL ÁRBOL
# ==============================================================================
class Nodo:
    """Representa un nodo de decisión o una hoja del árbol.

    En un nodo interno se guardan ``caracteristica`` y ``umbral`` para decidir
    por qué rama continuar. En una hoja, ``valor`` contiene la clase predicha.
    """

    def __init__(self, caracteristica=None, umbral=None, izquierdo=None, derecho=None, *, valor=None, gini=None,
                 n_samples=0):
        self.caracteristica = caracteristica
        self.umbral = umbral
        self.izquierdo = izquierdo
        self.derecho = derecho
        self.valor = valor
        self.gini = gini
        self.n_samples = n_samples

    def es_hoja(self):
        """Indica si este nodo termina una rama y ya contiene una predicción."""
        return self.valor is not None


def construir_arbol(X, y, profundidad=0, max_profundidad=3, min_muestras_split=2):
    """Construye recursivamente un árbol de decisión.

    El proceso se repite así:
    1. Si se alcanza una condición de parada, se crea una hoja con la clase
       mayoritaria.
    2. Si todavía se puede dividir, se busca el mejor ``feature_idx`` y
       umbral.
    3. Se crean los hijos izquierdo y derecho y se repite el proceso.

    ``max_profundidad`` limita la complejidad del árbol. Esto ayuda a evitar
    que memorice demasiado los datos de entrenamiento (sobreajuste).
    """
    n_samples = len(y)
    n_clases = len(np.unique(y))
    gini_nodo = calcular_gini(y)

    # Una hoja se crea cuando ya no conviene o no se permite seguir dividiendo.
    if (profundidad >= max_profundidad or n_clases == 1 or n_samples < min_muestras_split):
        valores_unicos, conteos = np.unique(y, return_counts=True)
        clase_mayoritaria = valores_unicos[np.argmax(conteos)]
        return Nodo(valor=clase_mayoritaria, gini=gini_nodo, n_samples=n_samples)

    feature_idx, umbral, _ = mejor_split(X, y)

    if feature_idx is None:
        valores_unicos, conteos = np.unique(y, return_counts=True)
        clase_mayoritaria = valores_unicos[np.argmax(conteos)]
        return Nodo(valor=clase_mayoritaria, gini=gini_nodo, n_samples=n_samples)

    # Las máscaras booleanas seleccionan las filas que irán a cada hijo.
    left_mask = X[:, feature_idx] <= umbral
    right_mask = ~left_mask

    hijo_izq = construir_arbol(X[left_mask], y[left_mask], profundidad + 1, max_profundidad, min_muestras_split)
    hijo_der = construir_arbol(X[right_mask], y[right_mask], profundidad + 1, max_profundidad, min_muestras_split)

    return Nodo(
        caracteristica=feature_idx,
        umbral=umbral,
        izquierdo=hijo_izq,
        derecho=hijo_der,
        gini=gini_nodo,
        n_samples=n_samples
    )


# ==============================================================================
# 4. PREDICCIÓN Y MÉTRICAS
# ==============================================================================
def predecir_muestra(nodo, x):
    """Predice una sola muestra recorriendo el árbol desde su raíz."""
    if nodo.es_hoja():
        return nodo.valor

    # La misma regla usada para construir el árbol decide el camino durante
    # la predicción: izquierda si cumple, derecha si no cumple.
    if x[nodo.caracteristica] <= nodo.umbral:
        return predecir_muestra(nodo.izquierdo, x)
    return predecir_muestra(nodo.derecho, x)


def predecir(raiz, X):
    """Aplica ``predecir_muestra`` a todas las filas de una matriz."""
    return np.array([predecir_muestra(raiz, x) for x in X])


def calcular_matriz_confusion(y_true, y_pred, num_clases):
    """Construye una tabla de aciertos y errores por clase.

    Las filas representan la clase real y las columnas la clase predicha.
    Por eso, la diagonal contiene los aciertos; los valores fuera de ella
    muestran qué clases se confundieron.
    """
    cm = np.zeros((num_clases, num_clases), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def calcular_metricas(cm, nombres_clases):
    """Calcula accuracy, precisión, recall y F1 para cada clase.

    Para una clase concreta:
    - TP: muestras de esa clase clasificadas correctamente.
    - FP: muestras de otras clases predichas como esta clase.
    - FN: muestras de esta clase predichas como otra.
    """
    exactitud = np.trace(cm) / np.sum(cm)
    num_clases = cm.shape[0]

    precisiones, recalls, f1_scores = [], [], []
    for i in range(num_clases):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisiones.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

    df_metricas = pd.DataFrame({
        'Clase': nombres_clases,
        'Precisión': precisiones,
        'Recall': recalls,
        'F1-Score': f1_scores
    })

    return exactitud, df_metricas


# ==============================================================================
# 5. GRAFICACIÓN DEL ÁRBOL DE DECISIÓN
# ==============================================================================
def graficar_arbol_cart(raiz, nombres_caracteristicas, nombres_clases):
    """
    Dibuja recursivamente la estructura jerárquica del árbol de decisión.

    Los nodos azules representan decisiones. Las hojas usan un color
    diferente según la clase predicha. La rama izquierda significa que la
    condición se cumple y la derecha que no se cumple.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    colores_clases = {0: '#ffb3b3', 1: '#b3ffb3', 2: '#b3b3ff'}
    color_decision = '#e6f2ff'

    def dibujar_nodo(nodo, x, y, dx):
        # Cada nodo se dibuja y, si no es hoja, se llama a sí mismo para
        # representar visualmente sus dos descendientes.
        if nodo.es_hoja():
            texto = f"Clase: {nombres_clases[nodo.valor]}\nGini: {nodo.gini:.2f}\nMuestras: {nodo.n_samples}"
            color_fondo = colores_clases[nodo.valor]
            bbox = dict(boxstyle="round,pad=0.5", fc=color_fondo, ec="#333333", lw=1.5)
            ax.text(x, y, texto, ha='center', va='center', bbox=bbox, fontsize=9, fontweight='bold')
            return

        texto = f"{nombres_caracteristicas[nodo.caracteristica]} <= {nodo.umbral:.2f}\nGini: {nodo.gini:.2f}\nMuestras: {nodo.n_samples}"
        bbox = dict(boxstyle="round,pad=0.4", fc=color_decision, ec="#2b5c8f", lw=1.5)
        ax.text(x, y, texto, ha='center', va='center', bbox=bbox, fontsize=9)

        y_hijo = y - 0.22
        x_izq = x - dx / 2
        x_der = x + dx / 2

        # Flecha izquierda (Cumple condición)
        ax.annotate('', xy=(x_izq, y_hijo + 0.05), xytext=(x, y - 0.05),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color='#555555'))
        ax.text((x + x_izq) / 2 - 0.02, (y + y_hijo) / 2, "Sí", ha='right', va='center', fontsize=9, color='green',
                fontweight='bold')

        # Flecha derecha (No cumple condición)
        ax.annotate('', xy=(x_der, y_hijo + 0.05), xytext=(x, y - 0.05),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color='#555555'))
        ax.text((x + x_der) / 2 + 0.02, (y + y_hijo) / 2, "No", ha='left', va='center', fontsize=9, color='red',
                fontweight='bold')

        dibujar_nodo(nodo.izquierdo, x_izq, y_hijo, dx / 2)
        dibujar_nodo(nodo.derecho, x_der, y_hijo, dx / 2)

    dibujar_nodo(raiz, x=0.5, y=0.9, dx=0.45)
    plt.title("Estructura del Árbol de Decisión CART (Dataset Iris)", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.show()


def graficar_evaluacion(cm, df_metricas, nombres_clases):
    """Muestra la matriz de confusión y las métricas de cada clase."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=nombres_clases, yticklabels=nombres_clases, cbar=False)
    axes[0].set_title('Matriz de Confusión', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Clase Predicha')
    axes[0].set_ylabel('Clase Real')

    df_plot = df_metricas.melt(id_vars='Clase', var_name='Métrica', value_name='Valor')
    sns.barplot(data=df_plot, x='Clase', y='Valor', hue='Métrica', palette='Set2', ax=axes[1])
    axes[1].set_title('Métricas por Clase', fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 1.15)

    for p in axes[1].patches:
        if p.get_height() > 0:
            axes[1].annotate(f'{p.get_height():.2f}',
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center', xytext=(0, 5),
                             textcoords='offset points', fontsize=9)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 6. EJECUCIÓN DEL FLUJO COMPLETO
# ==============================================================================
if __name__ == "__main__":
    # Esta condición evita ejecutar el flujo automáticamente si el archivo
    # se importa desde otro programa.
    X, y, nombres_caracteristicas, nombres_clases = cargar_y_preparar_datos('IrisFlowers.csv')

    # La semilla hace que la partición aleatoria sea reproducible: al ejecutar
    # el script otra vez se obtienen las mismas muestras de entrenamiento y
    # prueba.
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    tam_test = int(len(X) * 0.2)
    indices_test, indices_train = indices[:tam_test], indices[tam_test:]

    X_train, y_train = X[indices_train], y[indices_train]
    X_test, y_test = X[indices_test], y[indices_test]

    # Entrenar el árbol CART usando únicamente el conjunto de entrenamiento.
    # Las muestras de prueba se reservan para medir su capacidad de generalizar.
    arbol_raiz = construir_arbol(X_train, y_train, max_profundidad=3, min_muestras_split=2)

    # Evaluación en el conjunto que el árbol no vio durante el entrenamiento.
    y_pred = predecir(arbol_raiz, X_test)
    cm = calcular_matriz_confusion(y_test, y_pred, num_clases=len(nombres_clases))
    exactitud, df_metricas = calcular_metricas(cm, nombres_clases)

    print(f"Exactitud Global (Accuracy): {exactitud * 100:.2f}%\n")
    print(df_metricas.to_string(index=False))

    # 1. Graficar el Árbol Resultante
    graficar_arbol_cart(arbol_raiz, nombres_caracteristicas, nombres_clases)

    # 2. Graficar la Matriz de Confusión y Métricas
    graficar_evaluacion(cm, df_metricas, nombres_clases)