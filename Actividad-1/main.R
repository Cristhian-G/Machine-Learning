library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)

data <- read_csv("data(1).csv")

datos_cancer <- data |>
  select(-id, -...33)
str(datos_cancer)
head(datos_cancer, 10)
summary(datos_cancer)

levels(datos_cancer$diagnosis)
levels(datos_cancer$diagnosis) <- c("Benigno", "Maligno")

# verificar valores NA
sum(is.na(datos_cancer))
colSums(is.na(datos_cancer))

estadisticas_grup <- datos_cancer |>
  group_by(diagnosis) |>
  summarise(
    n = n(),
    radio_promedio = mean(radius_mean, na.rm = TRUE),
    radio_sd = sd(radius_mean, na.rm = TRUE),
    textura_promedio = mean(texture_mean, na.rm = TRUE),
    area_promedio = mean(area_mean, na.rm = TRUE),
    .group = "drop"
  )


tabla_diagnostico <- table(datos_cancer$diagnosis)
print(tabla_diagnostico)

prop.table(tabla_diagnostico)

df_resumen <- data.frame(
  Diagnostico = names(tabla_diagnostico),
  Conteo = as.vector(tabla_diagnostico),
  Porcentaje = as.vector(prop.table(tabla_diagnostico) * 100)
)

print(df_resumen)