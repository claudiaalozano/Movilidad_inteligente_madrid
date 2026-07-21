# Movilidad inteligente Madrid

Este repositorio contiene el desarrollo de un sistema de análisis de movilidad urbana en Madrid a partir de datos de puntos medidores de tráfico, red viaria de OpenStreetMap y estimaciones de velocidad.

El objetivo principal es construir un grafo de puntos medidores conectado mediante caminos reales sobre la red OSM, calcular tiempos estimados de recorrido y validar rutas representativas frente a tiempos de referencia externos.

## Fuentes de datos

El proyecto utiliza las siguientes fuentes de datos:

- Datos de puntos medidores de tráfico del Ayuntamiento de Madrid.
- Datos históricos de tráfico asociados a los medidores.
- Red viaria de Madrid descargada desde OpenStreetMap mediante OSMnx.
- Capa de velocidades estimadas para Madrid.
- Tiempos de referencia obtenidos manualmente desde Google Maps para la validación de rutas.

Los datos pesados no se almacenan necesariamente en GitHub. En algunos casos se utilizan rutas locales o Google Drive para trabajar con archivos de mayor tamaño.

## Estructura del repositorio

```text
Movilidad_inteligente_madrid/
│
├── ETA Holistico/
│   ├── ETA.py
│   ├── Informe.pdf
│   └── Presentación.pdf
│
├── automatizacion_modelo/
│   └── pipeline_reentrenamiento.ipynb
│
├── data/
│   └── processed/
│       ├── analisis_componentes.csv
│       ├── analisis_componentes_reparado.csv
│       ├── aristas_reparacion_componentes.csv
│       ├── mapa_componentes.html
│       ├── resumen_conectividad_reparacion.csv
│       └── revision_snapping_medidores.csv
│
├── grafo/
│   └── grafo_puntos_medidores.ipynb
│
├── mapa_velocidades/
│   └── velocidades_madrid_final.geojson
│
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## Orden de ejecución

### 1. Construcción del grafo de medidores

Notebook principal:

```text
grafo/grafo_puntos_medidores.ipynb
```

Este notebook realiza los siguientes pasos:

1. Carga de datos de medidores.
2. Descarga o carga de la red viaria de Madrid desde OpenStreetMap.
3. Asociación de cada medidor al nodo OSM más cercano.
4. Revisión del snapping de medidores a la red OSM.
5. Generación de pares candidatos entre medidores.
6. Cálculo de caminos reales sobre OSM.
7. Integración de velocidades y tiempos estimados.
8. Construcción del grafo de puntos medidores.
9. Análisis de conectividad.
10. Reparación de componentes desconectadas cercanas al componente principal.
11. Revisión de aristas añadidas.
12. Validación de rutas frente a tiempos de referencia externos.

### 2. Automatización del reentrenamiento

Notebook de automatización:

```text
automatizacion_modelo/pipeline_reentrenamiento.ipynb
```

Este notebook plantea una estructura inicial para automatizar el proceso de actualización del modelo de predicción de tráfico. Actualmente está en fase de diseño y no ejecuta todavía un entrenamiento real completo, ya que depende de datos pesados y de la integración definitiva del modelo predictivo.

## Archivos generados

El notebook del grafo genera varios archivos de salida en `data/processed/`.

### `analisis_componentes.csv`

Contiene el análisis actualizado de componentes conexas del grafo.

### `analisis_componentes_reparado.csv`

Contiene una copia explícita del análisis de componentes después de aplicar la reparación de conectividad.

### `aristas_reparacion_componentes.csv`

Contiene las aristas añadidas para conectar componentes cercanas al componente principal.

### `resumen_conectividad_reparacion.csv`

Resume la conectividad del grafo antes y después de aplicar la reparación.

### `revision_snapping_medidores.csv`

Contiene la revisión de distancias entre los medidores y los nodos OSM a los que han sido asignados.

### `mapa_componentes.html`

Mapa generado para visualizar la distribución de componentes del grafo.

## Validación de rutas

La validación se realiza comparando los tiempos estimados por el grafo con tiempos de referencia externos obtenidos manualmente desde Google Maps.

La ruta inicial Puerta del Sol → Plaza de Castilla se mantiene únicamente como validación exploratoria. Para la validación final se utiliza Sevilla → Plaza de Castilla, ya que Sevilla es un punto cercano al entorno de Sol pero situado en una zona con circulación rodada.

Las rutas de validación final son:

1. Sevilla → Plaza de Castilla
2. Atocha → Chamartín
3. Moncloa → Cibeles
4. Nuevos Ministerios → Legazpi
5. Plaza de España → Ventas
6. Ventas → Atocha
7. Cibeles → Atocha
8. Príncipe Pío → Avenida de América
9. Plaza de Castilla → Méndez Álvaro
10. Cibeles → Plaza Elíptica

## Automatización del reentrenamiento

La carpeta `automatizacion_modelo/` contiene el diseño inicial del pipeline de automatización para el reentrenamiento del modelo de predicción de tráfico.

El objetivo de esta parte es preparar un flujo que permita:

- detectar nuevos datos;
- incorporar esos datos al histórico;
- ejecutar un proceso de entrenamiento o reentrenamiento;
- registrar métricas y resultados;
- versionar modelos;
- documentar logs de ejecución.

Actualmente esta parte está en fase inicial y funciona como diseño del flujo. El entrenamiento real queda pendiente de integrar con el modelo predictivo definitivo y con los datos históricos completos.

## Requisitos

Las dependencias principales del proyecto están recogidas en:

```text
requirements.txt
```

Entre las librerías principales utilizadas se encuentran:

- pandas
- geopandas
- networkx
- osmnx
- numpy
- scikit-learn
- folium
- shapely

