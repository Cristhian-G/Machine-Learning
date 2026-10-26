<div align="center">

# Agrupamiento jerárquico en Python

Ejemplo didáctico de aprendizaje no supervisado con `SciPy`, `NumPy` y `Matplotlib`.

Cristhian Germán Ramírez Ruiz - [GitHub]("https://github.com/Cristhian-G")

Machine Learning 2026B 

</div>

## Descripción

Este proyecto muestra cómo aplicar el **agrupamiento jerárquico aglomerativo** a un conjunto de datos bidimensional. El algoritmo comienza considerando cada observación como un grupo independiente y fusiona progresivamente los grupos más similares hasta construir una jerarquía completa.

El resultado se representa mediante un **dendrograma**, un diagrama en forma de árbol que permite analizar las fusiones entre grupos y decidir cuántos clusters utilizar.

El notebook genera datos simulados distribuidos en tres grupos y muestra:

- Los datos originales sin etiquetas.
- La construcción de la jerarquía con el método de Ward.
- El dendrograma y un posible punto de corte.
- La asignación de cada observación a un cluster.
- Una visualización final de los grupos encontrados.

> **Nota:** el algoritmo no utiliza las etiquetas originales de los grupos simulados. Aprende la estructura únicamente a partir de las características de los datos.

## Fundamentos del algoritmo

El agrupamiento jerárquico puede ser:

- **Aglomerativo:** cada punto comienza como un cluster y los grupos se van fusionando.
- **Divisivo:** todos los puntos comienzan en un solo grupo y este se divide progresivamente.

En este ejemplo se utiliza el enfoque aglomerativo:

```python
Z = linkage(X, method="ward")
etiquetas = fcluster(Z, t=3, criterion="maxclust")
```

El método `ward` selecciona las fusiones que producen el menor incremento posible de la variabilidad interna. La función `fcluster` corta la jerarquía para generar las etiquetas finales; en este caso, se solicitan tres clusters.

## Requisitos

- Python 3.9 o superior
- Jupyter Notebook, JupyterLab o un editor compatible con notebooks
- Las bibliotecas indicadas en `requirements.txt`

## Instalación

1. Clona o descarga este proyecto y entra en su carpeta:

   ```bash
   cd Agrupamiento-jerarquico
   ```

2. Crea y activa un entorno virtual (recomendado):

   **Linux/macOS**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Windows PowerShell**

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Instala las dependencias del análisis:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Instala Jupyter si todavía no está disponible:

   ```bash
   python -m pip install notebook
   ```

## Ejecución del notebook

Desde la carpeta del proyecto, ejecuta:

```bash
jupyter notebook
```

Se abrirá una página en el navegador. Selecciona `sample.ipynb` y ejecuta las celdas en orden usando **Run** o presionando `Shift + Enter`.

También puedes utilizar JupyterLab:

```bash
jupyter lab
```

Otra opción es abrir directamente `sample.ipynb` con Visual Studio Code y seleccionar un kernel de Python que tenga instaladas las dependencias.

## Estructura del proyecto

```text
Agrupamiento-jerarquico/
├── sample.ipynb       # Notebook con la explicación y el código
├── requirements.txt   # Dependencias de Python
└── README.md          # Documentación del proyecto
```

## Interpretación de los resultados

En el dendrograma, la altura de cada unión representa la distancia entre los grupos fusionados. Un salto grande en altura suele indicar una separación importante entre grupos y puede ayudar a elegir el punto de corte.

La gráfica final utiliza un color diferente para cada cluster. Como los datos de este ejemplo tienen dos características, es posible visualizarlos directamente en un plano; con más características sería necesario utilizar técnicas de reducción de dimensionalidad o gráficos alternativos.

## Consideraciones

- El agrupamiento jerárquico no necesita fijar el número de clusters antes de construir el dendrograma.
- Puede ser costoso para conjuntos de datos muy grandes.
- La escala de las características puede influir en las distancias; en datos reales conviene considerar una normalización.
- Los valores atípicos pueden modificar la estructura del dendrograma.

## Licencia

Este proyecto tiene fines educativos y puede utilizarse libremente para estudiar los fundamentos del agrupamiento jerárquico.