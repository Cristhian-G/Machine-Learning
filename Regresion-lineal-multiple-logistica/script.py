import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# PASO 1: CARGA DE DATOS
# ==========================================
df = pd.read_csv('datos_cancer_purgados.csv')

# ==========================================
# PASO 2: PREPROCESAMIENTO DE VARIABLES
# ==========================================
# Convertir 'diagnosis' a binario: Benigno = 0, Maligno = 1
df['diagnosis_num'] = df['diagnosis'].map({'Benigno': 0, 'Maligno': 1})

print("Distribución de la variable objetivo (diagnosis):")
print(df['diagnosis_num'].value_counts())
print("\n" + "="*50 + "\n")

# ==========================================
# TAREA 1: REGRESIÓN LOGÍSTICA (CLASIFICACIÓN)
# ==========================================
# Variables predictoras (5 numéricas) y objetivo (diagnosis_num)
X_clas = df[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']]
y_clas = df['diagnosis_num']

# División en Train / Test (80% / 20%)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clas, y_clas, test_size=0.2, random_state=42
)

# Escalado de características (Recomendado para Regresión Logística)
scaler = StandardScaler()
X_train_c_scaled = scaler.fit_transform(X_train_c)
X_test_c_scaled = scaler.transform(X_test_c)

# Entrenamiento del modelo
modelo_logistico = LogisticRegression()
modelo_logistico.fit(X_train_c_scaled, y_train_c)

# Predicción y Evaluación
y_pred_c = modelo_logistico.predict(X_test_c_scaled)

matriz = confusion_matrix(y_test_c, y_pred_c)

ConfusionMatrixDisplay.from_predictions(
    y_test_c,
    y_pred_c,
    display_labels=['Benigno', 'Maligno'],
    cmap=plt.cm.Blues,
    values_format='d'
)

plt.title("Matriz de Confusión - Regresión Logística")
plt.show()

print("--- RESULTADOS TAREA 1: REGRESIÓN LOGÍSTICA ---")
print(f"Accuracy (Exactitud): {accuracy_score(y_test_c, y_pred_c):.4f}")
print("\nReporte de Clasificación Detallado:\n")
print(classification_report(y_test_c, y_pred_c, target_names=['Benigno', 'Maligno']))
print("="*50 + "\n")

# ==========================================
# TAREA 2: REGRESIÓN LINEAL MÚLTIPLEprint("Matriz de confusión:")
print(pd.DataFrame(
    matriz,
    index=['Real Benigno', 'Real Maligno'],
    columns=['Predicho Benigno', 'Predicho Maligno']
))
# ==========================================
# Variables predictoras (4 numéricas) y objetivo (area_mean)
X_reg = df[['radius_mean', 'texture_mean', 'perimeter_mean', 'smoothness_mean']]
y_reg = df['area_mean']

# División en Train / Test (80% / 20%)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Entrenamiento del modelo
modelo_lineal = LinearRegression()
modelo_lineal.fit(X_train_r, y_train_r)

# Predicción y Evaluación
y_pred_r = modelo_lineal.predict(X_test_r)

print("--- RESULTADOS TAREA 2: REGRESIÓN LINEAL MÚLTIPLE ---")
print(f"Coeficiente de Determinación R^2: {r2_score(y_test_r, y_pred_r):.4f}")
print(f"Error Cuadrático Medio (MSE): {mean_squared_error(y_test_r, y_pred_r):.4f}")

print("\nCoeficientes del Modelo:")
for feature, coef in zip(X_reg.columns, modelo_lineal.coef_):
    print(f"{feature}: {coef:.4f}")

print(f"Intercepto: {modelo_lineal.intercept_:.4f}")