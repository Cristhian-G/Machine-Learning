# 1. Cargar librerías requeridas
library(readr)
library(dplyr)
library(ggplot2)

# 2. Cargar el dataset sucio
datos_sucios <- read_csv("datos_cancer_sucios.csv")

# Función auxiliar para tratar outliers usando el rango intercuartílico (IQR)
limpiar_outliers_iqr <- function(x){
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr_val <- q3 - q1
  limite_inferior <- q1 - 1.5 * iqr_val
  limite_superior <- q3 + 1.5 * iqr_val
  
  x[x < limite_inferior | x > limite_superior] <- NA
  return(x)
}

# 3. Pipeline de limpieza y transformación
datos_purgados <- datos_sucios %>%
  # paso 1: eliminar columna basura
  select(-columna_basura) %>%
  
  # paso 2: Quitar filas duplicadas
  distinct() %>%
  
  # paso 3: Normalizar diagnosis (quitar espacios y pasar a mayúsculas)
  mutate(diagnosis = toupper(trimws(diagnosis))) %>%
  
  # paso 4: Eliminar filas con NA en la variable clave (diagnosis)
  filter(!is.na(diagnosis) & diagnosis %in% c("B","M")) %>%
  
  # paso 5: Convertir columnas cuantitativas a numéricas (valores no convertibles se vuelven NA)
  mutate(across(c(radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean), as.numeric)) %>%
  
  # Paso 6: Convertir valores negativos imposibles a NA
  mutate(across(c(radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean), 
                ~ ifelse(.x < 0, NA, .x))) %>%
  
  # Paso 7: Identificar y reemplazar outliers extremos por NA usando IQR
  mutate(across(c(radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean), 
                limpiar_outliers_iqr)) %>%
  
  # Paso 8: Imputar los NA restantes en las variables cuantitativas con la mediana
  mutate(across(c(radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean), 
                ~ ifelse(is.na(.x), median(.x, na.rm = TRUE), .x))) %>%
  
  # Paso 9: Convertir diagnosis a factor con etiquetas descriptivas
  mutate(diagnosis = factor(diagnosis, levels = c("B", "M"), labels = c("Benigno", "Maligno")))

# 4. Exportar datasets
write_csv(datos_sucios, "datos_cancer_sucios.csv")
write_csv(datos_purgados, "datos_cancer_purgados.csv")
