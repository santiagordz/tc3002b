# Detección de Plagio mediante Embeddings Semánticos

## Metodología

### Selección de Datos

Para el desarrollo de este modelo de detección de plagio, se seleccionó un conjunto de datos específicamente diseñado para evaluar diferentes niveles de similitud textual. El conjunto de datos consiste en:

1. **Texto original**: Un documento base que sirve como referencia para todas las comparaciones.
2. **Textos con similitud alta**: Documentos con cambios mínimos o reemplazos de palabras que mantienen la estructura y significado del texto original.
3. **Textos con similitud media**: Documentos con paráfrasis o reestructuración que mantienen el significado pero alteran la estructura.
4. **Textos con similitud baja**: Documentos sobre el mismo tema pero con perspectivas diferentes o enfoques distintos.

La selección de este conjunto de datos se realizó considerando la necesidad de evaluar el rendimiento del modelo en diferentes escenarios de plagio, desde los más evidentes hasta los más sutiles. Cada documento viene etiquetado con su nivel de similitud esperado (alto, medio, bajo), lo que permite evaluar la precisión del modelo.

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
   - Carga y preprocesamiento de textos
   - Generación de embeddings para cada texto
   - Cálculo de similitud coseno entre los embeddings
   - Clasificación del nivel de similitud basado en umbrales predefinidos

3. **Definición de umbrales**:
   - Alto: similitud ≥ 0.85
   - Medio: 0.55 ≤ similitud < 0.85
   - Bajo: similitud < 0.55

4. **Evaluación y ajuste**:
   - Validación del modelo con textos etiquetados
   - Ajuste de umbrales para optimizar la precisión
   - Análisis de errores para identificar casos problemáticos

El modelo final utiliza exclusivamente embeddings semánticos para determinar la similitud entre textos, lo que permite detectar plagio incluso cuando se han realizado cambios significativos en la estructura o el vocabulario del texto.

## Resultados

### Presentación de Hallazgos

El modelo de detección de plagio basado en embeddings semánticos fue evaluado utilizando el conjunto de datos descrito anteriormente. Los resultados principales son:

1. **Métricas de rendimiento**:
   - **Exactitud global**: 83.33%
   - **Precisión por clase**:
     - Alta: 100%
     - Media: 66.67%
     - Baja: 100%
   - **Recall por clase**:
     - Alta: 100%
     - Media: 100%
     - Baja: 50%
   - **F1-score por clase**:
     - Alta: 100%
     - Media: 80%
     - Baja: 66.67%

2. **Análisis de la matriz de confusión**:
   - El modelo identifica correctamente todos los casos de similitud alta
   - Identifica correctamente todos los casos de similitud media
   - Confunde algunos casos de similitud baja con similitud media

3. **Comparación con estudios previos**:
   - Los embeddings semánticos muestran un rendimiento superior a métodos tradicionales como Bag of Words (BOW) o TF-IDF en la detección de plagio con paráfrasis.
   - El modelo es más robusto ante cambios estructurales en el texto, manteniendo la capacidad de detectar similitud semántica.

4. **Evaluación de objetivos**:
   - Se logró el objetivo principal de desarrollar un modelo de detección de plagio basado exclusivamente en embeddings semánticos.
   - El modelo demuestra alta precisión en la identificación de similitud alta y media, aunque presenta algunas limitaciones en la detección de similitud baja.
   - La implementación es eficiente, procesando múltiples documentos en segundos.

Los resultados demuestran que los embeddings semánticos son una herramienta poderosa para la detección de plagio, especialmente en casos donde se han realizado modificaciones significativas al texto original pero se mantiene el significado.

## Conclusiones

### Resumen y Discusión

#### Principales hallazgos y contribuciones

El estudio demuestra que los embeddings semánticos proporcionan una base sólida para la detección de plagio, capturando la similitud conceptual entre textos incluso cuando difieren en estructura y vocabulario. La exactitud global del 83.33% confirma la efectividad del enfoque, especialmente para detectar casos de similitud alta y media.

Un hallazgo importante es la capacidad del modelo para identificar correctamente todos los casos de similitud alta, lo que es crucial en aplicaciones de detección de plagio donde es prioritario identificar los casos más evidentes. Esto contribuye al campo de la detección de plagio al proporcionar un método que no depende de coincidencias exactas de texto.

#### Implicaciones prácticas y teóricas

Desde el punto de vista práctico, este modelo puede implementarse en entornos educativos y académicos para detectar plagio en trabajos estudiantiles, incluso cuando se han realizado paráfrasis o reestructuraciones del texto original. La simplicidad del enfoque, utilizando únicamente embeddings semánticos, facilita su implementación y reduce la complejidad computacional.

Teóricamente, el estudio refuerza la idea de que la similitud semántica es un indicador más robusto de plagio que la similitud léxica o sintáctica. Esto sugiere que futuros desarrollos en detección de plagio deberían priorizar la comprensión del significado sobre la coincidencia de palabras o estructuras.

#### Limitaciones y trabajo futuro

A pesar de los resultados prometedores, el estudio presenta algunas limitaciones:

1. **Dificultad con similitud baja**: El modelo muestra menor precisión en la identificación de textos con similitud baja, confundiéndolos ocasionalmente con similitud media. Esto sugiere que los embeddings pueden capturar similitudes temáticas incluso cuando los textos son sustancialmente diferentes.

2. **Dependencia de umbrales fijos**: El uso de umbrales predefinidos para clasificar los niveles de similitud puede no ser óptimo para todos los tipos de textos o dominios.

3. **Tamaño del conjunto de datos**: El conjunto de datos utilizado es relativamente pequeño, lo que limita la generalización de los resultados.

Para abordar estas limitaciones, el trabajo futuro podría incluir:

1. Incorporación de técnicas de aprendizaje automático para determinar umbrales adaptativos según el dominio o tipo de texto.

2. Exploración de modelos de embeddings más avanzados, como BERT o GPT, que podrían capturar matices semánticos con mayor precisión.

3. Ampliación del conjunto de datos para incluir una mayor variedad de textos y niveles de similitud.

4. Desarrollo de un enfoque híbrido que combine embeddings semánticos con otras técnicas como análisis estilométrico para mejorar la precisión en casos de similitud baja.

En conclusión, los embeddings semánticos representan una herramienta valiosa para la detección de plagio, ofreciendo un equilibrio entre simplicidad y efectividad. Con refinamientos adicionales, este enfoque tiene el potencial de convertirse en un estándar en sistemas de detección de plagio académico y profesional.