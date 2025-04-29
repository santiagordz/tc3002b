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
        'alto': 0.85,
        'medio': 0.55
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
    """Preprocesamiento básico del texto."""
    if texto is None:
        return ''
    # Normalizar espacios
    texto = re.sub(r'\s+', ' ', texto)
    # Conservar sólo caracteres alfanuméricos y espacios para análisis general
    texto_procesado = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚüÜñÑ0-9\s]', '', texto)
    return texto_procesado.lower().strip()

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

# Obtener embeddings semánticos
def obtener_embeddings_semanticos(textos):
    """Obtiene embeddings semánticos para los textos proporcionados."""
    modelo = cargar_modelo_embeddings()
    
    if modelo == "FALLBACK":
        # Modelo fallback simple basado en TF-IDF
        vectorizer = TfidfVectorizer(max_features=512)
        embeddings = vectorizer.fit_transform(textos).toarray()
        return embeddings
    
    try:
        # Asegurarse de que los textos no estén vacíos
        textos = [t if t.strip() else "texto vacío" for t in textos]
        
        # Truncar textos muy largos para evitar problemas de memoria
        textos = [t[:100000] for t in textos]
        
        # Obtener embeddings
        embeddings = modelo(textos).numpy()
        return embeddings
    
    except Exception as e:
        print(f"Error al obtener embeddings: {e}")
        # Fallback a TF-IDF simple
        vectorizer = TfidfVectorizer(max_features=512)
        embeddings = vectorizer.fit_transform(textos).toarray()
        return embeddings

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
    """Clasifica el nivel de similitud basado en umbrales."""
    if sim >= CONFIG['umbrales']['alto']:
        return 'high'
    elif sim >= CONFIG['umbrales']['medio']:
        return 'medium'
    else:
        return 'low'

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
    
    for metodo in metodos:
        if f"pred_{metodo}" in df_resultados.columns:
            y_true = df_resultados["precargado"]
            y_pred = df_resultados[f"pred_{metodo}"]
            
            # Calcular precisión, recall, f1 para cada clase
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average=None, labels=['high', 'medium', 'low']
            )
            
            # Calcular exactitud
            exactitud = (y_true == y_pred).mean()
            
            metricas[metodo] = {
                'exactitud': exactitud,
                'precision': dict(zip(['high', 'medium', 'low'], precision)),
                'recall': dict(zip(['high', 'medium', 'low'], recall)),
                'f1': dict(zip(['high', 'medium', 'low'], f1)),
                'matriz_confusion': confusion_matrix(
                    y_true, y_pred, labels=['high', 'medium', 'low']
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
        if nombre_archivo.startswith("high"):
            precargado = "high"
        elif nombre_archivo.startswith("moderate") or nombre_archivo.startswith("medium"):
            precargado = "medium"
        elif nombre_archivo.startswith("low"):
            precargado = "low"
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

# Ejecutar análisis completo
def ejecutar_analisis(ruta_carpeta):
    """Ejecuta el análisis completo de similitud entre textos."""
    tiempo_inicio = time.time()
    
    # Verificar que existe la carpeta
    if not os.path.exists(ruta_carpeta):
        return {
            "error": f"La carpeta {ruta_carpeta} no existe"
        }
    
    # Verificar que existe el archivo original
    archivo_original = os.path.join(ruta_carpeta, "original.txt")
    if not os.path.exists(archivo_original):
        return {
            "error": f"No se encontró el archivo original.txt en {ruta_carpeta}"
        }
    
    # Leer texto original
    try:
        with open(archivo_original, 'r', encoding='utf-8') as f:
            texto_original = preprocess_text(f.read())
    except Exception as e:
        return {
            "error": f"Error al leer el archivo original: {e}"
        }
    
    # Preparar archivos a procesar
    archivos = []
    for nombre_archivo in os.listdir(ruta_carpeta):
        if nombre_archivo == 'original.txt' or not nombre_archivo.endswith('.txt'):
            continue
        archivos.append(nombre_archivo)
    
    # Si no hay archivos para comparar
    if not archivos:
        return {
            "error": "No se encontraron archivos para comparar con el original"
        }
    
    # Procesar archivos (secuencial o paralelo)
    resultados = []
    
    try:
        # Procesamiento paralelo
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(procesar_archivo, nombre, ruta_carpeta, texto_original) 
                      for nombre in archivos]
            for future in futures:
                resultado = future.result()
                resultados.append(resultado)
    except Exception as e:
        # Fallback a procesamiento secuencial si falla el paralelo
        print(f"Error en procesamiento paralelo: {e}. Usando procesamiento secuencial...")
        for nombre in archivos:
            resultado = procesar_archivo(nombre, ruta_carpeta, texto_original)
            resultados.append(resultado)
    
    # Crear DataFrame con resultados
    try:
        df_resultados = pd.DataFrame(resultados)
        
        # Contar aciertos por método
        aciertos = {}
        for metodo in ['BOW', 'TFIDF', 'MARKOV', 'SEMANTICO', 'NGRAMA_PALABRA', 'NGRAMA_CARACTER', 'COMBINADO']:
            col_acierto = f"acierto_{metodo}"
            if col_acierto in df_resultados.columns:
                aciertos[metodo] = df_resultados[col_acierto].sum()
        
        # Evaluar modelo
        metricas = evaluar_modelo(df_resultados)
        
        # Guardar resultados a CSV
        ruta_salida = os.path.join(ruta_carpeta, "resultados_similitud.csv")
        df_resultados.to_csv(ruta_salida, index=False)
        
        # Guardar métricas
        ruta_metricas = os.path.join(ruta_carpeta, "metricas_evaluacion.json")
        with open(ruta_metricas, 'w', encoding='utf-8') as f:
            json.dump(metricas, f, indent=4)
        
        tiempo_total = time.time() - tiempo_inicio
        
        # Crear resultados visuales
        crear_visualizaciones(df_resultados, ruta_carpeta)
        
        return {
            "tiempo_ejecucion": tiempo_total,
            "total_documentos": len(resultados),
            "aciertos": aciertos,
            "metricas": metricas,
            "ruta_resultados": ruta_salida,
            "archivos_procesados": archivos,
            "df_resultados": df_resultados,
        }
        
    except Exception as e:
        return {
            "error": f"Error al crear resultados: {e}",
            "resultados_parciales": resultados
        }

# Crear visualizaciones
def crear_visualizaciones(df_resultados, ruta_carpeta):
    """Crea visualizaciones de resultados."""
    try:
        # 1. Gráfico de comparación de similitudes por método
        plt.figure(figsize=(14, 8))
        
        metodos = ['BOW', 'TFIDF', 'SEMANTICO', 'NGRAMA_PALABRA', 'NGRAMA_CARACTER', 'MARKOV', 'COMBINADO']
        colores = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'black']
        
        for i, metodo in enumerate(metodos):
            col = f"cos_{metodo}"
            if col in df_resultados.columns:
                plt.scatter(
                    df_resultados.index, 
                    df_resultados[col],
                    alpha=0.7,
                    label=metodo,
                    color=colores[i % len(colores)]
                )
        
        plt.title('Comparación de Similitudes por Método')
        plt.xlabel('Documento')
        plt.ylabel('Puntuación de Similitud')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        ruta_comparacion = os.path.join(ruta_carpeta, 'grafico_comparacion_metodos.png')
        plt.savefig(ruta_comparacion)
        plt.close()
        
        # 2. Matriz de confusión para método combinado
        if 'pred_COMBINADO' in df_resultados.columns and 'precargado' in df_resultados.columns:
            cm = confusion_matrix(
                df_resultados['precargado'], 
                df_resultados['pred_COMBINADO'],
                labels=['high', 'medium', 'low']
            )
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['high', 'medium', 'low'],
                       yticklabels=['high', 'medium', 'low'])
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
        with open('config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        print(f"Error al guardar configuración: {e}")
        return False

# Función para cargar configuración
def cargar_configuracion():
    """Carga la configuración desde un archivo."""
    try:
        if os.path.exists('config.json'):
            with open('config.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        return CONFIG  # Usar configuración por defecto
    except Exception as e:
        print(f"Error al cargar configuración: {e}")
        return CONFIG  # Usar configuración por defecto

# Inicializar configuración al cargar el módulo
CONFIG = cargar_configuracion()

# Punto de entrada si se ejecuta como script
if __name__ == "__main__":
    print("Detector de Plagio - Sistema de Análisis de Similitud Textual")
    print("Para usar la interfaz de usuario, ejecute el archivo 'app.py'")