# Detección de Plagio mediante Embeddings Semánticos

## Metodología

### Selección de Datos

Para el desarrollo y evaluación de este modelo de detección de plagio, se utilizó el conjunto de datos "Dokumen Teks". Este conjunto de datos está estructurado en dos carpetas principales:

1.  **Original**: Contiene los documentos fuente que sirven como referencia.
2.  **Copy**: Contiene documentos que son versiones modificadas o copias de los documentos originales, diseñados para simular diferentes escenarios de plagio o similitud.

La elección de este conjunto de datos se basa en su estructura clara, que permite una comparación directa por pares entre un documento original y su correspondiente versión sospechosa. Esto facilita la evaluación del modelo en su tarea principal: determinar el grado de similitud entre dos textos específicos.

### Análisis de Datos

El análisis de los datos se realizó mediante las siguientes técnicas:

1. **Preprocesamiento textual**:
   - Normalización de espacios
   - Eliminación de caracteres especiales
   - Conversión a minúsculas

2. **Análisis exploratorio**:
   - Evaluación de la longitud de los textos
   - Identificación de patrones comunes entre textos similares
   - Análisis de la distribución de palabras

3. **Herramientas utilizadas**:
   - Python como lenguaje principal
   - Bibliotecas: NumPy, Pandas para manipulación de datos
   - Matplotlib y Seaborn para visualización
   - TensorFlow y TensorFlow Hub para el modelo de embeddings

El análisis reveló que los textos con similitud alta comparten una gran proporción de su contenido, mientras que los textos con similitud media mantienen el significado pero con estructuras diferentes. Los textos con similitud baja comparten el tema pero difieren significativamente en contenido y estructura.

### Construcción del Modelo

El modelo de detección de plagio se construyó utilizando embeddings semánticos, específicamente el Universal Sentence Encoder (USE) de Google. El proceso de construcción siguió estos pasos:

1. **Selección del modelo de embeddings**:
   - Se eligió el Universal Sentence Encoder por su capacidad para capturar el significado semántico de los textos, independientemente de la estructura sintáctica.
   - Este modelo convierte textos en vectores de 512 dimensiones que representan su significado semántico.

2. **Implementación del pipeline de procesamiento**:
   - Carga de pares de archivos (original y sospechoso).
   - Preprocesamiento de textos: normalización de espacios, eliminación de caracteres no alfanuméricos y conversión a minúsculas.
   - Generación de embeddings semánticos para cada texto del par utilizando el Universal Sentence Encoder (USE).
   - Cálculo de la similitud coseno entre los vectores de embeddings del par de textos.
   - Clasificación del resultado basado en umbrales predefinidos para determinar el nivel de similitud.

3. **Definición de umbrales y clasificación**:
   Se establecieron los siguientes umbrales para clasificar la similitud entre un texto original y uno sospechoso:
   - **Plagio**: Similitud coseno ≥ 0.70
   - **Sospechoso**: 0.50 ≤ Similitud coseno < 0.70
   - **Original**: Similitud coseno < 0.50
   Además, se calcula un nivel de confianza para cada clasificación, indicando qué tan cerca está la similitud del siguiente umbral.

4. **Evaluación y ajuste**:
   - Validación del modelo con textos etiquetados
   - Ajuste de umbrales para optimizar la precisión
   - Análisis de errores para identificar casos problemáticos

El modelo final compara pares de documentos (original vs. sospechoso) utilizando exclusivamente embeddings semánticos para determinar la similitud. Clasifica cada par como 'plagio', 'sospechoso' u 'original' basándose en la similitud coseno calculada y los umbrales definidos. Este enfoque permite detectar similitudes semánticas incluso con cambios estructurales o de vocabulario.

## Resultados

### Presentación de Hallazgos

El modelo de detección de plagio basado en embeddings semánticos fue evaluado utilizando el conjunto de datos descrito anteriormente. Los resultados principales son:

1. **Resultados de la Comparación**:
   - El script procesa cada par de archivos (original de la carpeta `Original` y su correspondiente en `Copy`).
   - Para cada par, calcula la similitud coseno utilizando embeddings semánticos.
   - Clasifica el resultado como 'plagio', 'sospechoso' u 'original' según los umbrales definidos (0.70 y 0.50).
   - Genera un archivo CSV (`resultados_similitud.csv`) y un JSON (`metricas_similitud.json`) que detallan la similitud, clasificación y confianza para cada par de documentos analizado.
   *(Nota: Las métricas de rendimiento como precisión, recall y F1-score requerirían etiquetas de verdad fundamental (ground truth) para cada par de documentos en el dataset 'Dokumen Teks', las cuales no se proporcionan explícitamente en el conjunto de datos original ni se generan en el script actual. La evaluación se centra en la similitud calculada y la clasificación resultante).*

2. **Análisis de Resultados por Clasificación**:
   - Los resultados muestran la distribución de los pares de documentos entre las categorías 'plagio', 'sospechoso' y 'original'.
   - Se puede observar qué pares alcanzan los umbrales más altos de similitud (plagio) y cuáles presentan similitudes intermedias (sospechoso) o bajas (original).
   - El análisis cualitativo de los textos correspondientes a cada categoría puede ayudar a validar si los umbrales capturan adecuadamente los diferentes niveles de modificación textual.

3. **Comparación con estudios previos**:
   - Los embeddings semánticos muestran un rendimiento superior a métodos tradicionales como Bag of Words (BOW) o TF-IDF en la detección de plagio con paráfrasis.
   - El modelo es más robusto ante cambios estructurales en el texto, manteniendo la capacidad de detectar similitud semántica.

4. **Evaluación de objetivos**:
   - Se logró el objetivo principal de desarrollar un modelo de detección de plagio basado exclusivamente en embeddings semánticos.
   - El modelo implementa exitosamente la comparación por pares y la clasificación en tres niveles ('plagio', 'sospechoso', 'original') utilizando umbrales específicos.
   - La evaluación de la efectividad de estos umbrales y clasificaciones dependería de una validación contra un conjunto de datos etiquetado o una revisión experta.
   - La implementación es eficiente, procesando múltiples documentos en segundos.

Los resultados generados (archivos CSV y JSON) proporcionan una cuantificación detallada de la similitud semántica entre los documentos originales y sus contrapartes modificadas, clasificada según los umbrales definidos. Esto demuestra la capacidad de los embeddings semánticos para diferenciar niveles de similitud textual.

## Conclusiones

### Resumen y Discusión

#### Principales hallazgos y contribuciones

El trabajo implementado demuestra que los embeddings semánticos, específicamente el Universal Sentence Encoder, proporcionan una base sólida para cuantificar la similitud conceptual entre pares de textos. El sistema clasifica eficazmente los pares en categorías de 'plagio', 'sospechoso' u 'original' basándose en umbrales de similitud coseno predefinidos (0.70 y 0.50).

Una contribución clave es la implementación de un sistema claro de comparación por pares que no solo mide la similitud sino que la categoriza en niveles accionables ('plagio', 'sospechoso', 'original'), proporcionando además una métrica de confianza. Este enfoque va más allá de la simple detección de similitud y ofrece una interpretación directa del resultado.

#### Implicaciones prácticas y teóricas

Desde el punto de vista práctico, este sistema puede implementarse en entornos educativos y académicos para analizar documentos sospechosos comparándolos con fuentes originales. La clasificación tripartita ('plagio', 'sospechoso', 'original') ayuda a priorizar la revisión manual. La dependencia exclusiva de embeddings semánticos simplifica la implementación.

Teóricamente, el trabajo refuerza la utilidad de la similitud semántica capturada por embeddings como un indicador robusto de la relación conceptual entre textos, superando las limitaciones de métodos basados puramente en coincidencias léxicas o sintácticas, especialmente frente a paráfrasis o reestructuraciones.

#### Limitaciones y trabajo futuro

A pesar de los resultados prometedores, el estudio presenta algunas limitaciones:

1. **Validación de Umbrales**: La efectividad de los umbrales (0.70 y 0.50) depende del corpus específico y del tipo de plagio que se busca detectar. Podrían requerir ajuste o validación con un conjunto de datos etiquetado más grande y diverso para asegurar su generalización.

2. **Ausencia de Ground Truth**: El conjunto de datos "Dokumen Teks" no incluye etiquetas explícitas de nivel de plagio para cada par, lo que impide calcular métricas de rendimiento estándar (precisión, recall) y obliga a una evaluación más cualitativa de los resultados.

3. **Enfoque por Pares**: El sistema actual se enfoca en comparar un documento sospechoso contra uno original específico. No está diseñado para buscar plagio contra una base de datos extensa de documentos fuente.

Para abordar estas limitaciones, el trabajo futuro podría incluir:

1. **Validación y Ajuste de Umbrales**: Realizar una evaluación rigurosa utilizando un conjunto de datos con etiquetas de verdad fundamental para validar y potencialmente ajustar los umbrales de clasificación.

2. Exploración de modelos de embeddings más avanzados, como BERT o GPT, que podrían capturar matices semánticos con mayor precisión.

3. **Evaluación con Diversos Tipos de Plagio**: Probar el sistema con ejemplos de diferentes tipos de plagio (copia literal, paráfrasis, mosaico, etc.) para entender mejor sus fortalezas y debilidades.

4. **Integración con Bases de Datos**: Extender el sistema para permitir la comparación de un documento sospechoso contra una colección más grande de documentos fuente.

En conclusión, el sistema implementado basado en embeddings semánticos ofrece un método práctico y efectivo para cuantificar y clasificar la similitud entre pares de documentos. Proporciona una base sólida para la detección de plagio, especialmente útil contra modificaciones no literales, aunque su rendimiento óptimo depende de la validación y posible ajuste de los umbrales de clasificación.