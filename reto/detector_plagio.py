import os
import numpy as np
import pandas as pd
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import tensorflow as tf
import tensorflow_hub as hub
import json


# Configuración global
CONFIG = {
    'umbrales': {
        'plagio': 0.70,    # Antes 'alto'
        'sospechoso': 0.50  # Antes 'medio'
    },
    'pesos': {
        'bow': 0.15,
        'tfidf': 0.20,
        'semantico': 0.10,
        'ngrama_palabra': 0.15,
        'ngrama_caracter': 0.20,
        'markov': 0.10,
        'estilo': 0.05,
        'estructura': 0.05,
    }
}

# Cache para modelos de embeddings
EMBEDDINGS_MODEL = None

# Preprocesamiento del texto
def preprocess_text(texto):
    """Preprocesamiento básico del texto con optimizaciones.
    
    Args:
        texto (str): Texto a preprocesar
        
    Returns:
        str: Texto preprocesado
    """
    # Verificar entrada
    if texto is None or not isinstance(texto, str):
        return ''
    
    try:
        # Limitar longitud para mejorar rendimiento
        if len(texto) > 1000000:  # Limitar a 1M caracteres
            texto = texto[:1000000]
            print("Texto truncado a 1M caracteres para optimizar rendimiento")
            
        # Normalizar espacios (más eficiente)
        texto = ' '.join(texto.split())
        
        # Conservar sólo caracteres alfanuméricos y espacios para análisis general
        texto_procesado = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚüÜñÑ0-9\s]', '', texto)
        
        return texto_procesado.lower().strip()
    except Exception as e:
        print(f"Error en preprocesamiento de texto: {e}")
        return ''


# Carga de modelo de embeddings semánticos
def cargar_modelo_embeddings():
    """Carga el modelo de embeddings semánticos (Universal Sentence Encoder)."""
    global EMBEDDINGS_MODEL
    
    if EMBEDDINGS_MODEL is None:
        print("Cargando modelo de embeddings semánticos...")
        
        try:
            # Intentar usar modelo previamente descargado
            if os.path.exists('modelos/use_model'):
                EMBEDDINGS_MODEL = hub.load('modelos/use_model')
            else:
                # Descargar y guardar modelo
                EMBEDDINGS_MODEL = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
                # Crear directorio si no existe
                if not os.path.exists('modelos'):
                    os.makedirs('modelos')
                # Guardar para uso futuro
                tf.saved_model.save(EMBEDDINGS_MODEL, 'modelos/use_model')
            
            print("Modelo de embeddings cargado con éxito.")
        except Exception as e:
            print(f"Error al cargar modelo de embeddings: {e}")
            print("Usando un modelo alternativo simplificado...")
            # Modelo alternativo simple (fallback)
            EMBEDDINGS_MODEL = "FALLBACK"
    
    return EMBEDDINGS_MODEL

# Cache para embeddings
EMBEDDINGS_CACHE = {}

# Obtener embeddings semánticos
def obtener_embeddings_semanticos(textos):
    """Obtiene embeddings semánticos para los textos proporcionados con caché."""
    global EMBEDDINGS_CACHE
    modelo = cargar_modelo_embeddings()
    
    # Crear claves de caché para los textos
    cache_keys = [hash(texto[:1000]) for texto in textos]  # Usar solo los primeros 1000 caracteres para la clave
    
    # Verificar caché para cada texto
    resultados = []
    textos_a_procesar = []
    indices_a_procesar = []
    
    for i, (texto, key) in enumerate(zip(textos, cache_keys)):
        if key in EMBEDDINGS_CACHE:
            resultados.append(EMBEDDINGS_CACHE[key])
        else:
            textos_a_procesar.append(texto)
            indices_a_procesar.append(i)
    
    # Si no hay textos para procesar, devolver resultados de caché
    if not textos_a_procesar:
        return np.array(resultados)
    
    # Procesar textos que no están en caché
    if modelo == "FALLBACK":
        # Modelo fallback simple basado en TF-IDF
        vectorizer = TfidfVectorizer(max_features=512)
        nuevos_embeddings = vectorizer.fit_transform(textos_a_procesar).toarray()
    else:
        try:
            # Asegurarse de que los textos no estén vacíos
            textos_a_procesar = [t if t.strip() else "texto vacío" for t in textos_a_procesar]
            
            # Truncar textos muy largos para evitar problemas de memoria
            textos_a_procesar = [t[:100000] for t in textos_a_procesar]
            
            # Obtener embeddings
            nuevos_embeddings = modelo(textos_a_procesar).numpy()
        except Exception as e:
            print(f"Error al obtener embeddings: {e}")
            # Fallback a TF-IDF simple
            vectorizer = TfidfVectorizer(max_features=512)
            nuevos_embeddings = vectorizer.fit_transform(textos_a_procesar).toarray()
    
    # Guardar nuevos embeddings en caché
    for i, idx in enumerate(indices_a_procesar):
        EMBEDDINGS_CACHE[cache_keys[idx]] = nuevos_embeddings[i]
        resultados.insert(idx, nuevos_embeddings[i])
    
    # Limitar tamaño de caché (mantener solo los últimos 100 embeddings)
    if len(EMBEDDINGS_CACHE) > 100:
        # Eliminar las claves más antiguas
        oldest_keys = list(EMBEDDINGS_CACHE.keys())[:-100]
        for key in oldest_keys:
            del EMBEDDINGS_CACHE[key]
    
    return np.array(resultados)

# Matriz de transición de Markov
def crear_matriz_markov_mejorada(texto, n=2):
    """Crea matriz de transición de n-gramas de caracteres."""
    # Preprocesar el texto para Markov
    texto = ''.join(c for c in texto.lower() if c.isalpha() or c.isspace())
    
    if len(texto) <= n:
        return np.zeros((27, 27))  # 26 letras + espacio
    
    # Mapear caracteres a índices (26 para espacio)
    char_a_idx = lambda c: 26 if c == ' ' else (ord(c) - ord('a') if 'a' <= c <= 'z' else 0)
    
    matriz = np.zeros((27, 27))
    for i in range(len(texto) - 1):
        a, b = char_a_idx(texto[i]), char_a_idx(texto[i + 1])
        if 0 <= a < 27 and 0 <= b < 27:
            matriz[a][b] += 1
            
    # Normalizar
    sumas_filas = matriz.sum(axis=1, keepdims=True)
    matriz = np.divide(matriz, sumas_filas, out=np.zeros_like(matriz), where=sumas_filas != 0)
    return matriz

# Función de similitud coseno para matrices densas
def cosine_sim_dense(v1, v2):
    """Calcula similitud coseno entre matrices densas."""
    if np.sum(v1) == 0 or np.sum(v2) == 0:
        return 0.0
    return np.dot(v1.flatten(), v2.flatten()) / (np.linalg.norm(v1.flatten()) * np.linalg.norm(v2.flatten()))

# Calcular similitud de n-gramas
def calcular_similitud_ngrama(texto1, texto2, rango_n=(2, 3), analizador='word'):
    """Calcula similitud basada en n-gramas."""
    # Usar n-gramas de caracteres o palabras
    vectorizador = CountVectorizer(analyzer=analizador, ngram_range=rango_n)
    
    try:
        vectores = vectorizador.fit_transform([texto1, texto2])
        if vectores[0].nnz == 0 or vectores[1].nnz == 0:  # Verificar si hay elementos no cero
            return 0.0
        similitud = cosine_similarity(vectores[0], vectores[1])[0, 0]
        return similitud
    except Exception as e:
        print(f"Error en cálculo de n-gramas: {e}")
        return 0.0

# Extraer características estilísticas
def extraer_caracteristicas_estilo(texto):
    """Extrae características estilísticas del texto."""
    if not texto or len(texto) < 10:
        return {
            'longitud_promedio_palabra': 0,
            'longitud_promedio_oracion': 0,
            'ratio_puntuacion': 0,
            'ratio_mayusculas': 0
        }
    
    # Recuperar texto original con puntuación para este análisis
    texto_original = texto
    
    # Calcular características
    palabras = [w for w in texto.split() if w]
    oraciones = [s for s in re.split(r'[.!?]', texto_original) if s.strip()]
    
    try:
        caracteristicas = {
            'longitud_promedio_palabra': np.mean([len(w) for w in palabras]) if palabras else 0,
            'longitud_promedio_oracion': np.mean([len(s.split()) for s in oraciones]) if oraciones else 0,
            'ratio_puntuacion': len(re.findall(r'[.,;:!?]', texto_original)) / max(1, len(texto_original)),
            'ratio_mayusculas': sum(1 for c in texto_original if c.isupper()) / max(1, len(texto_original))
        }
    except Exception as e:
        print(f"Error al extraer características de estilo: {e}")
        caracteristicas = {
            'longitud_promedio_palabra': 0,
            'longitud_promedio_oracion': 0,
            'ratio_puntuacion': 0,
            'ratio_mayusculas': 0
        }
    
    return caracteristicas

# Clasificar la similitud
def clasificar_similitud(sim):
    """Clasifica el nivel de similitud basado en umbrales.
    
    Retorna:
        str: 'plagio', 'sospechoso', o 'original' según el nivel de similitud.
    """
    if sim >= CONFIG['umbrales']['plagio']:
        return 'plagio'
    elif sim >= CONFIG['umbrales']['sospechoso']:
        return 'sospechoso'
    else:
        return 'original'

# Calcular similitud combinada
def calcular_similitud_combinada(texto_original, texto_similar, pesos=None):
    """Calcula la similitud combinada usando múltiples métricas con pesos."""
    if pesos is None:
        pesos = CONFIG['pesos']
    
    resultados = {}
    
    # Calcular similitudes individuales
    
    # Vectorización BOW
    vectorizer_bow = CountVectorizer()
    try:
        vectores_bow = vectorizer_bow.fit_transform([texto_original, texto_similar])
        cos_bow = cosine_similarity(vectores_bow[0], vectores_bow[1])[0, 0]
    except:
        cos_bow = 0
    resultados['cos_BOW'] = cos_bow
    
    # Vectorización TF-IDF
    vectorizer_tfidf = TfidfVectorizer()
    try:
        vectores_tfidf = vectorizer_tfidf.fit_transform([texto_original, texto_similar])
        cos_tfidf = cosine_similarity(vectores_tfidf[0], vectores_tfidf[1])[0, 0]
    except:
        cos_tfidf = 0
    resultados['cos_TFIDF'] = cos_tfidf
    
    # Similitud semántica
    try:
        embeddings_semanticos = obtener_embeddings_semanticos([texto_original, texto_similar])
        cos_semantico = cosine_similarity([embeddings_semanticos[0]], [embeddings_semanticos[1]])[0][0]
    except:
        cos_semantico = 0
    resultados['cos_SEMANTICO'] = cos_semantico
    
    # Similitudes de N-gramas
    cos_ngrama_palabra = calcular_similitud_ngrama(texto_original, texto_similar, rango_n=(2, 3), analizador='word')
    resultados['cos_NGRAMA_PALABRA'] = cos_ngrama_palabra
    
    cos_ngrama_caracter = calcular_similitud_ngrama(texto_original, texto_similar, rango_n=(3, 5), analizador='char')
    resultados['cos_NGRAMA_CARACTER'] = cos_ngrama_caracter
    
    # Similitudes de estilo
    markov_orig = crear_matriz_markov_mejorada(texto_original)
    markov_sim = crear_matriz_markov_mejorada(texto_similar)
    cos_markov = cosine_sim_dense(markov_orig, markov_sim)
    resultados['cos_MARKOV'] = cos_markov
    
    # Extraer características de estilo
    estilo_orig = extraer_caracteristicas_estilo(texto_original)
    estilo_sim = extraer_caracteristicas_estilo(texto_similar)
    
    # Calcular similitud de estilo (enfoque simple)
    puntuacion_sim_estilo = 0
    if estilo_orig and estilo_sim:
        caracteristicas = list(estilo_orig.keys())
        difs_caracteristicas = [abs(estilo_orig[f] - estilo_sim[f]) / max(estilo_orig[f], 0.001) for f in caracteristicas]
        puntuacion_sim_estilo = 1 - min(1, sum(difs_caracteristicas) / len(caracteristicas))
    resultados['cos_ESTILO'] = puntuacion_sim_estilo
    
    # Combinar todas las similitudes con pesos
    puntuacion_combinada = (
        pesos['bow'] * cos_bow +
        pesos['tfidf'] * cos_tfidf +
        pesos['semantico'] * cos_semantico +
        pesos['ngrama_palabra'] * cos_ngrama_palabra +
        pesos['ngrama_caracter'] * cos_ngrama_caracter +
        pesos['markov'] * cos_markov +
        pesos['estilo'] * puntuacion_sim_estilo
    )
    resultados['cos_COMBINADO'] = puntuacion_combinada
    
    # Clasificar todas las similitudes
    for clave in list(resultados):
        metodo = clave.replace('cos_', '')
        resultados[f'pred_{metodo}'] = clasificar_similitud(resultados[clave])
    
    return resultados

# Evaluar modelo
def evaluar_modelo(df_resultados):
    """Evalúa el modelo con múltiples métricas."""
    metricas = {}
    
    # Evaluar cada método por separado
    metodos = ['BOW', 'TFIDF', 'MARKOV', 'SEMANTICO', 'NGRAMA_PALABRA', 'NGRAMA_CARACTER', 'COMBINADO']
    
    # Nuevas etiquetas de clasificación
    etiquetas = ['plagio', 'sospechoso', 'original']
    
    for metodo in metodos:
        if f"pred_{metodo}" in df_resultados.columns:
            y_true = df_resultados["precargado"]
            y_pred = df_resultados[f"pred_{metodo}"]
            
            # Calcular precisión, recall, f1 para cada clase
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average=None, labels=etiquetas
            )
            
            # Calcular exactitud
            exactitud = (y_true == y_pred).mean()
            
            metricas[metodo] = {
                'exactitud': exactitud,
                'precision': dict(zip(etiquetas, precision)),
                'recall': dict(zip(etiquetas, recall)),
                'f1': dict(zip(etiquetas, f1)),
                'matriz_confusion': confusion_matrix(
                    y_true, y_pred, labels=etiquetas
                ).tolist()  # Convertir a lista para poder serializarlo
            }
    
    return metricas

# Procesar archivos
def procesar_archivo(nombre_archivo, ruta_carpeta, texto_original):
    """Procesa un archivo de texto y calcula similitudes con el original."""
    
    archivo_similar = os.path.join(ruta_carpeta, nombre_archivo)
    
    try:
        with open(archivo_similar, 'r', encoding='utf-8') as f:
            texto_similar = preprocess_text(f.read())
        
        # Extraer etiqueta precargada del nombre del archivo
        if nombre_archivo.startswith("plagio"):
            precargado = "plagio"
        elif nombre_archivo.startswith("sospechoso"):
            precargado = "sospechoso"
        elif nombre_archivo.startswith("original") and nombre_archivo != "original.txt":
            precargado = "original"
        # Mantener compatibilidad con el formato anterior
        elif nombre_archivo.startswith("high"):
            precargado = "plagio"
        elif nombre_archivo.startswith("moderate") or nombre_archivo.startswith("medium"):
            precargado = "sospechoso"
        elif nombre_archivo.startswith("low"):
            precargado = "original"
        else:
            precargado = "unknown"
        
        # Calcular todas las similitudes
        resultados_similitud = calcular_similitud_combinada(texto_original, texto_similar)
        
        # Preparar resultado
        resultado = {
            "original": "original.txt",
            "similar": nombre_archivo,
            "precargado": precargado,
        }
        
        # Añadir todas las métricas calculadas
        resultado.update(resultados_similitud)
        
        # Añadir aciertos
        for metodo in ['BOW', 'TFIDF', 'MARKOV', 'SEMANTICO', 'NGRAMA_PALABRA', 'NGRAMA_CARACTER', 'COMBINADO']:
            if f"pred_{metodo}" in resultados_similitud:
                resultado[f"acierto_{metodo}"] = resultado[f"pred_{metodo}"] == precargado
        
        return resultado
    
    except Exception as e:
        print(f"Error al procesar el archivo {nombre_archivo}: {e}")
        return {
            "original": "original.txt",
            "similar": nombre_archivo,
            "error": str(e)
        }

def crear_visualizaciones_no_etiquetadas(df_resultados, ruta_carpeta):
    """Crea visualizaciones adecuadas para datos no etiquetados."""
    try:
        # 1. Distribución de similitudes
        plt.figure(figsize=(10, 6))
        plt.hist(df_resultados['similitud'], bins=20, alpha=0.7, color='blue')
        plt.title('Distribución de Puntuaciones de Similitud')
        plt.xlabel('Puntuación de Similitud')
        plt.ylabel('Frecuencia')
        plt.grid(True, alpha=0.3)
        
        # Añadir líneas de umbral
        plt.axvline(x=CONFIG['umbrales']['plagio'], color='red', linestyle='--', 
                  label=f'Umbral de Plagio ({CONFIG["umbrales"]["plagio"]})')
        plt.axvline(x=CONFIG['umbrales']['sospechoso'], color='orange', linestyle='--', 
                  label=f'Umbral Sospechoso ({CONFIG["umbrales"]["sospechoso"]})')
        
        plt.legend()
        plt.tight_layout()
        
        ruta_dist = os.path.join(ruta_carpeta, 'distribucion_similitudes.png')
        plt.savefig(ruta_dist)
        plt.close()
        
        # 2. Mapa de calor de similitudes entre documentos
        if len(df_resultados) <= 100:  # Limitar para evitar mapas de calor enormes
            # Crear una matriz de similitud entre todos los pares
            originales = df_resultados['original'].unique()
            copias = df_resultados['similar'].unique()
            
            # Crear una matriz para el mapa de calor
            matriz_similitud = np.zeros((len(originales), len(copias)))
            
            # Llenar la matriz
            for i, orig in enumerate(originales):
                for j, cop in enumerate(copias):
                    fila = df_resultados[(df_resultados['original'] == orig) & 
                                        (df_resultados['similar'] == cop)]
                    if not fila.empty:
                        matriz_similitud[i, j] = fila['similitud'].values[0]
            
            # Crear el mapa de calor
            plt.figure(figsize=(12, 8))
            sns.heatmap(matriz_similitud, cmap='YlOrRd', 
                       xticklabels=copias, yticklabels=originales)
            plt.title('Mapa de Calor de Similitud entre Documentos')
            plt.xlabel('Documentos de Copy')
            plt.ylabel('Documentos de Original')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            ruta_heatmap = os.path.join(ruta_carpeta, 'mapa_calor_similitud.png')
            plt.savefig(ruta_heatmap)
            plt.close()
        
        # 3. Gráfico de comparación de métodos
        plt.figure(figsize=(14, 8))
        
        metodos = ['BOW', 'TFIDF', 'SEMANTICO', 'NGRAMA_PALABRA', 'NGRAMA_CARACTER', 'MARKOV', 'COMBINADO']
        colores = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'black']
        
        # Crear identificador para cada comparación
        df_resultados['id_comparacion'] = df_resultados.apply(
            lambda row: f"{row['original']} vs {row['similar']}", axis=1)
        
        # Ordenar por similitud combinada para mejor visualización
        df_ordenado = df_resultados.sort_values(by='similitud', ascending=False)
        
        for i, metodo in enumerate(metodos):
            col = f"cos_{metodo}"
            if col in df_ordenado.columns:
                plt.scatter(
                    range(len(df_ordenado)), 
                    df_ordenado[col],
                    alpha=0.7,
                    label=metodo,
                    color=colores[i % len(colores)]
                )
        
        plt.title('Comparación de Similitudes por Método')
        plt.xlabel('Comparación (ordenado por similitud)')
        plt.ylabel('Puntuación de Similitud')
        plt.xticks([])  # Ocultar etiquetas del eje x
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        ruta_comparacion = os.path.join(ruta_carpeta, 'grafico_comparacion_metodos.png')
        plt.savefig(ruta_comparacion)
        plt.close()
        
        # 4. Tops documentos más similares
        tops = df_resultados.sort_values(by='similitud', ascending=False).head(10)
        
        plt.figure(figsize=(12, 6))
        plt.bar(tops['id_comparacion'], tops['similitud'], color='royalblue')
        plt.title('Top 10 Documentos con Mayor Similitud')
        plt.xlabel('Par de Documentos')
        plt.ylabel('Puntuación de Similitud')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        ruta_tops = os.path.join(ruta_carpeta, 'top_documentos_similares.png')
        plt.savefig(ruta_tops)
        plt.close()
        
    except Exception as e:
        print(f"Error al crear visualizaciones para datos no etiquetados: {e}")

def realizar_clustering(df_resultados, ruta_carpeta):
    """Realiza análisis de clustering para agrupar documentos por similitud."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Preparar características para clustering
        columnas_metricas = [col for col in df_resultados.columns if col.startswith('cos_')]
        
        if not columnas_metricas:
            print("No se encontraron métricas para realizar clustering")
            return
        
        # Crear matriz de características
        X = df_resultados[columnas_metricas].values
        
        # Normalizar características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determinar número óptimo de clusters (entre 2 y 5)
        from sklearn.metrics import silhouette_score
        
        silhouette_scores = []
        k_range = range(2, min(6, len(df_resultados) // 5 + 1))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Seleccionar mejor k
        # if silhouette_scores:
        #     mejor_k = k_range[np.argmax(silhouette_scores)]
        # else:
        mejor_k = 3  # Valor predeterminado
        
        # Crear modelo final
        kmeans = KMeans(n_clusters=mejor_k, random_state=42, n_init=10)
        df_resultados['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Visualizar clusters en 2D usando PCA
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(X_scaled)
        
        # Crear DataFrame para visualización
        df_pca = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
        df_pca['cluster'] = df_resultados['cluster']
        df_pca['similitud'] = df_resultados['similitud']
        df_pca['id_comparacion'] = df_resultados['id_comparacion'] if 'id_comparacion' in df_resultados.columns else df_resultados.index
        
        # Graficar clusters
        plt.figure(figsize=(10, 8))
        
        # Colores para clusters
        colores = plt.cm.tab10(np.linspace(0, 1, mejor_k))
        
        # Graficar cada cluster
        for i in range(mejor_k):
            plt.scatter(
                df_pca[df_pca['cluster'] == i]['PC1'], 
                df_pca[df_pca['cluster'] == i]['PC2'],
                s=50, c=[colores[i]], 
                label=f'Cluster {i+1}'
            )
        
        # Marcar centroides
        centroides = pca.transform(kmeans.cluster_centers_)
        plt.scatter(
            centroides[:, 0], centroides[:, 1],
            s=200, marker='X', c='black', alpha=0.7,
            label='Centroides'
        )
        
        plt.title(f'Clustering de Documentos por Similitud (k={mejor_k})')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        ruta_clusters = os.path.join(ruta_carpeta, 'clustering_documentos.png')
        plt.savefig(ruta_clusters)
        plt.close()
        
        # Guardar resultados con clusters
        df_resultados.to_csv(os.path.join(ruta_carpeta, "resultados_clustering.csv"), index=False)
        
        # Análisis por cluster
        resumen_clusters = df_resultados.groupby('cluster').agg({
            'similitud': ['mean', 'min', 'max', 'count']
        })
        
        # Crear informe por cluster
        plt.figure(figsize=(10, 6))
        
        # Graficar estadísticas por cluster
        for i in range(mejor_k):
            if i in resumen_clusters.index:
                plt.bar(
                    f"Cluster {i+1}",
                    resumen_clusters.loc[i, ('similitud', 'mean')],
                    yerr=[[0], [resumen_clusters.loc[i, ('similitud', 'max')] - resumen_clusters.loc[i, ('similitud', 'mean')]]],
                    color=colores[i],
                    alpha=0.7,
                    capsize=10
                )
        
        plt.title('Similitud Media por Cluster')
        plt.ylabel('Similitud Media')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        ruta_resumen = os.path.join(ruta_carpeta, 'resumen_clusters.png')
        plt.savefig(ruta_resumen)
        plt.close()
        
    except Exception as e:
        print(f"Error al realizar clustering: {e}")

# Modificar ejecutar_analisis para trabajar con datos no etiquetados
def ejecutar_analisis(ruta_carpeta, max_comparaciones=25):
    """Ejecuta el análisis completo de similitud entre textos sin depender de etiquetas."""
    tiempo_inicio = time.time()
    
    # Verificar que existe la carpeta
    if not os.path.exists(ruta_carpeta):
        return {
            "error": f"La carpeta {ruta_carpeta} no existe"
        }
    
    # Verificar que existen las subcarpetas 'Original' y 'Copy'
    ruta_original = os.path.join(ruta_carpeta, "Original")
    ruta_copy = os.path.join(ruta_carpeta, "Copy")
    
    if not os.path.exists(ruta_original) or not os.path.exists(ruta_copy):
        return {
            "error": f"No se encontraron las carpetas 'Original' y 'Copy' en {ruta_carpeta}"
        }
    
    # Listar archivos originales y copias
    archivos_originales = [f for f in os.listdir(ruta_original) if f.endswith('.txt')]
    archivos_copia = [f for f in os.listdir(ruta_copy) if f.endswith('.txt')]
    
    if not archivos_originales or not archivos_copia:
        return {
            "error": f"No se encontraron archivos .txt en las carpetas 'Original' o 'Copy'"
        }
    
    # Limitar el número de comparaciones seleccionando un subconjunto de archivos
    if len(archivos_originales) * len(archivos_copia) > max_comparaciones:
        # Seleccionar un número equilibrado de archivos
        import random
        random.shuffle(archivos_originales)
        random.shuffle(archivos_copia)
        
        # Calcular cuántos archivos de cada tipo necesitamos para acercarnos al máximo
        from math import sqrt
        n_originales = min(len(archivos_originales), int(sqrt(max_comparaciones)))
        n_copias = min(len(archivos_copia), max_comparaciones // n_originales)
        
        archivos_originales_seleccionados = archivos_originales[:n_originales]
        archivos_copia_seleccionados = archivos_copia[:n_copias]
    else:
        archivos_originales_seleccionados = archivos_originales
        archivos_copia_seleccionados = archivos_copia
    
    total_comparaciones = len(archivos_originales_seleccionados) * len(archivos_copia_seleccionados)
    print(f"Realizando {total_comparaciones} comparaciones (de un total posible de {len(archivos_originales) * len(archivos_copia)})")
    
    # Procesar cada combinación de archivo original con archivos de copia
    resultados = []
    
    # Determinar número óptimo de workers
    max_workers = min(os.cpu_count(), 4)  # Limitar a 4 workers
    
    comparaciones_realizadas = 0
    
    try:
        # Procesamiento de comparaciones
        for archivo_original in archivos_originales_seleccionados:
            ruta_completa_original = os.path.join(ruta_original, archivo_original)
            
            # Leer texto original
            try:
                with open(ruta_completa_original, 'r', encoding='utf-8') as f:
                    texto_original = preprocess_text(f.read())
            except Exception as e:
                print(f"Error al leer el archivo original {archivo_original}: {e}")
                continue
            
            # Procesar cada archivo de copia con este original
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Función para procesar una comparación
                def procesar_comparacion(archivo_copia):
                    ruta_completa_copia = os.path.join(ruta_copy, archivo_copia)
                    
                    try:
                        with open(ruta_completa_copia, 'r', encoding='utf-8') as f:
                            texto_copia = preprocess_text(f.read())
                        
                        # Calcular similitudes
                        resultados_similitud = calcular_similitud_combinada(texto_original, texto_copia)
                        
                        # Preparar resultado
                        resultado = {
                            "original": archivo_original,
                            "similar": archivo_copia,
                            "similitud": resultados_similitud['cos_COMBINADO'],
                            "nivel": resultados_similitud['pred_COMBINADO']
                        }
                        
                        # Añadir todas las métricas calculadas
                        resultado.update(resultados_similitud)
                        
                        return resultado
                    
                    except Exception as e:
                        print(f"Error al procesar el archivo {archivo_copia}: {e}")
                        return {
                            "original": archivo_original,
                            "similar": archivo_copia,
                            "error": str(e)
                        }
                
                # Enviar todas las comparaciones para procesamiento
                futures = [executor.submit(procesar_comparacion, archivo_copia) for archivo_copia in archivos_copia_seleccionados]
                
                # Recolectar resultados
                for future in futures:
                    resultado = future.result()
                    resultados.append(resultado)
                    comparaciones_realizadas += 1
                    
                    # Mostrar progreso
                    if comparaciones_realizadas % 5 == 0 or comparaciones_realizadas == total_comparaciones:
                        print(f"Progreso: {comparaciones_realizadas}/{total_comparaciones} comparaciones ({(comparaciones_realizadas/total_comparaciones)*100:.1f}%)")
    
    except Exception as e:
        # Fallback a procesamiento secuencial
        print(f"Error en procesamiento paralelo: {e}. Usando procesamiento secuencial...")
        for archivo_original in archivos_originales_seleccionados:
            ruta_completa_original = os.path.join(ruta_original, archivo_original)
            
            try:
                with open(ruta_completa_original, 'r', encoding='utf-8') as f:
                    texto_original = preprocess_text(f.read())
                    
                for archivo_copia in archivos_copia_seleccionados:
                    ruta_completa_copia = os.path.join(ruta_copy, archivo_copia)
                    
                    try:
                        with open(ruta_completa_copia, 'r', encoding='utf-8') as f:
                            texto_copia = preprocess_text(f.read())
                        
                        resultados_similitud = calcular_similitud_combinada(texto_original, texto_copia)
                        
                        resultado = {
                            "original": archivo_original,
                            "similar": archivo_copia,
                            "similitud": resultados_similitud['cos_COMBINADO'],
                            "nivel": resultados_similitud['pred_COMBINADO']
                        }
                        
                        resultado.update(resultados_similitud)
                        resultados.append(resultado)
                        
                        comparaciones_realizadas += 1
                        if comparaciones_realizadas % 5 == 0 or comparaciones_realizadas == total_comparaciones:
                            print(f"Progreso: {comparaciones_realizadas}/{total_comparaciones} comparaciones ({(comparaciones_realizadas/total_comparaciones)*100:.1f}%)")
                    
                    except Exception as e:
                        print(f"Error al procesar el archivo {archivo_copia}: {e}")
                        resultados.append({
                            "original": archivo_original,
                            "similar": archivo_copia,
                            "error": str(e)
                        })
            
            except Exception as e:
                print(f"Error al leer el archivo original {archivo_original}: {e}")
                continue
    
    # Crear DataFrame con resultados
    try:
        df_resultados = pd.DataFrame(resultados)
        
        # Guardar resultados a CSV
        ruta_salida = os.path.join(ruta_carpeta, "resultados_similitud.csv")
        df_resultados.to_csv(ruta_salida, index=False)
        
        tiempo_total = time.time() - tiempo_inicio
        
        # Crear visualizaciones para datos no etiquetados
        crear_visualizaciones_no_etiquetadas(df_resultados, ruta_carpeta)
        
        # Realizar análisis de clustering si hay suficientes datos
        if len(df_resultados) >= 10:
            realizar_clustering(df_resultados, ruta_carpeta)
        
        return {
            "tiempo_ejecucion": tiempo_total,
            "total_documentos": len(resultados),
            "ruta_resultados": ruta_salida,
            "archivos_procesados": [f"{o} vs {c}" for o in archivos_originales_seleccionados for c in archivos_copia_seleccionados],
            "df_resultados": df_resultados,
        }
        
    except Exception as e:
        return {
            "error": f"Error al crear resultados: {e}",
            "resultados_parciales": resultados
        }
        
# Actualizar la función crear_visualizaciones para manejar el nuevo formato de datos
def crear_visualizaciones(df_resultados, ruta_carpeta):
    """Crea visualizaciones de resultados."""
    try:
        # 1. Gráfico de comparación de similitudes por método
        plt.figure(figsize=(14, 8))
        
        metodos = ['BOW', 'TFIDF', 'SEMANTICO', 'NGRAMA_PALABRA', 'NGRAMA_CARACTER', 'MARKOV', 'COMBINADO']
        colores = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'black']
        
        # Crear un identificador único para cada comparación
        df_resultados['id_comparacion'] = df_resultados.apply(lambda row: f"{row['original']} vs {row['similar']}", axis=1)
        
        for i, metodo in enumerate(metodos):
            col = f"cos_{metodo}"
            if col in df_resultados.columns:
                plt.scatter(
                    range(len(df_resultados)), 
                    df_resultados[col],
                    alpha=0.7,
                    label=metodo,
                    color=colores[i % len(colores)]
                )
        
        plt.title('Comparación de Similitudes por Método')
        plt.xlabel('Comparación (Documento Original vs Copia)')
        plt.ylabel('Puntuación de Similitud')
        plt.xticks([])  # Ocultar etiquetas del eje x porque pueden ser muchas
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        ruta_comparacion = os.path.join(ruta_carpeta, 'grafico_comparacion_metodos.png')
        plt.savefig(ruta_comparacion)
        plt.close()
        
        # 2. Matriz de confusión para método combinado (solo si hay etiquetas válidas)
        if ('pred_COMBINADO' in df_resultados.columns and 
            'precargado' in df_resultados.columns and 
            df_resultados['precargado'].nunique() > 1):
            
            etiquetas = sorted(df_resultados['precargado'].unique())
            
            # Filtrar filas con precargado desconocido
            df_filtrado = df_resultados[df_resultados['precargado'] != 'unknown']
            
            if not df_filtrado.empty:
                cm = confusion_matrix(
                    df_filtrado['precargado'], 
                    df_filtrado['pred_COMBINADO'],
                    labels=etiquetas
                )
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=etiquetas,
                           yticklabels=etiquetas)
                plt.title('Matriz de Confusión - Método Combinado')
                plt.xlabel('Predicción')
                plt.ylabel('Real')
                plt.tight_layout()
                
                ruta_matriz = os.path.join(ruta_carpeta, 'matriz_confusion_combinado.png')
                plt.savefig(ruta_matriz)
                plt.close()
        
    except Exception as e:
        print(f"Error al crear visualizaciones: {e}")

# Función para comparar dos textos directamente
def comparar_textos(texto1, texto2):
    """Compara dos textos directamente y devuelve la similitud."""
    texto1_procesado = preprocess_text(texto1)
    texto2_procesado = preprocess_text(texto2)
    
    resultados = calcular_similitud_combinada(texto1_procesado, texto2_procesado)
    
    # Filtrar solo los resultados relevantes
    resultados_filtrados = {
        'similitud_bow': resultados['cos_BOW'],
        'similitud_tfidf': resultados['cos_TFIDF'],
        'similitud_semantica': resultados['cos_SEMANTICO'],
        'similitud_ngrama_palabra': resultados['cos_NGRAMA_PALABRA'],
        'similitud_ngrama_caracter': resultados['cos_NGRAMA_CARACTER'],
        'similitud_markov': resultados['cos_MARKOV'],
        'similitud_estilo': resultados['cos_ESTILO'],
        'similitud_combinada': resultados['cos_COMBINADO'],
        'nivel_similitud': resultados['pred_COMBINADO']
    }
    
    return resultados_filtrados

# Función para guardar configuración
def guardar_configuracion(config):
    """Guarda la configuración actual a un archivo."""
    try:
        # Crear directorio si no existe
        if not os.path.exists('config'):
            os.makedirs('config')
        
        # Guardar configuración
        with open('config/config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        
        # Actualizar configuración global
        global CONFIG
        CONFIG = config
        
        return True
    except Exception as e:
        print(f"Error al guardar configuración: {e}")
        return False

# Función para cargar configuración
def cargar_configuracion():
    """Carga la configuración desde un archivo."""
    try:
        # Verificar si existe el archivo
        if os.path.exists('config/config.json'):
            with open('config/config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # Verificar y actualizar etiquetas antiguas si es necesario
            if 'umbrales' in config:
                # Convertir de formato antiguo a nuevo si es necesario
                if 'alto' in config['umbrales'] and 'plagio' not in config['umbrales']:
                    config['umbrales']['plagio'] = config['umbrales'].pop('alto')
                if 'medio' in config['umbrales'] and 'sospechoso' not in config['umbrales']:
                    config['umbrales']['sospechoso'] = config['umbrales'].pop('medio')
            
            return config
        else:
            return CONFIG
    except Exception as e:
        print(f"Error al cargar configuración: {e}")
        return CONFIG

# Inicializar configuración al cargar el módulo
CONFIG = cargar_configuracion()

# Punto de entrada si se ejecuta como script
if __name__ == "__main__":
    print("Detector de Plagio - Sistema de Análisis de Similitud Textual")
    print("Para usar la interfaz de usuario, ejecute el archivo 'app.py'")