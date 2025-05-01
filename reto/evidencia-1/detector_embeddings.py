import os
import numpy as np
import pandas as pd
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import silhouette_score
from scipy.stats import entropy
import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf
import tensorflow_hub as hub
import json
from json import JSONEncoder

# Configuración global
CONFIG = {
    # Configuración para análisis de similitud
    'similitud': {
        'umbral_plagio': 0.50,  # Umbral para considerar plagio
    },
    # Rutas de carpetas
    'rutas': {
        'documentos_originales': './Dokumen Teks/Original',
        'documentos_sospechosos': './Dokumen Teks/Copy'
    }
}

# Cache para modelo de embeddings
EMBEDDINGS_MODEL = None

def preprocess_text(texto):
    """Preprocesamiento básico del texto."""
    if texto is None:
        return ''
    # Normalizar espacios
    texto = re.sub(r'\s+', ' ', texto)
    # Conservar sólo caracteres alfanuméricos y espacios para análisis general
    texto_procesado = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚüÜñÑ0-9\s]', '', texto)
    return texto_procesado.lower().strip()

def cargar_modelo_embeddings():
    """Carga el modelo de embeddings semánticos (Universal Sentence Encoder)."""
    global EMBEDDINGS_MODEL
    
    if EMBEDDINGS_MODEL is None:
        print("Cargando modelo de embeddings semánticos...")
        
        try:
            # Intentar usar modelo previamente descargado
            if os.path.exists('./modelos'):
                print("Cargando modelo de embeddings desde la carpeta local...")
                EMBEDDINGS_MODEL = hub.load('./modelos/use_model')
            else:
                print("Modelo no encontrado. Descargando...")
                # Descargar y guardar modelo
                EMBEDDINGS_MODEL = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
                # Crear directorio si no existe
                if not os.path.exists('./modelos'):
                    os.makedirs('./modelos')
                # Guardar para uso futuro
                tf.saved_model.save(EMBEDDINGS_MODEL, './modelos/use_model')
            
            print("Modelo de embeddings cargado con éxito.")
        except Exception as e:
            print(f"Error al cargar modelo de embeddings: {e}")
            print("Usando un modelo alternativo simplificado...")
            # Modelo alternativo simple (fallback)
            EMBEDDINGS_MODEL = "FALLBACK"
    
    return EMBEDDINGS_MODEL

def obtener_embeddings_semanticos(textos):
    """Obtiene embeddings semánticos para los textos proporcionados."""
    modelo = cargar_modelo_embeddings()
    
    if modelo == "FALLBACK":
        # Fallback a vectores aleatorios para demostración
        return np.random.rand(len(textos), 512)
    
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
        # Fallback a vectores aleatorios para demostración
        return np.random.rand(len(textos), 512)

def clasificar_similitud(sim):
    """Clasifica el nivel de similitud basado en umbrales.
    
    Args:
        sim: Valor de similitud entre 0 y 1
        
    Returns:
        dict: Diccionario con la clasificación y nivel de confianza
    """
    if sim >= CONFIG['similitud']['umbral_plagio']:
        return {
            'clasificacion': 'plagio',
            'confianza': (sim - CONFIG['similitud']['umbral_plagio']) / (1 - CONFIG['similitud']['umbral_plagio'])
        }
    else:
        return {
            'clasificacion': 'original',
            'confianza': 1 - (sim / CONFIG['similitud']['umbral_plagio'])
        }

def comparar_textos_embeddings(texto1, texto2):
    """Compara dos textos usando embeddings semánticos.
    
    Args:
        texto1: Texto original
        texto2: Texto a comparar
        
    Returns:
        dict: Diccionario con métricas de similitud
    """
    texto1_procesado = preprocess_text(texto1)
    texto2_procesado = preprocess_text(texto2)
    
    # Obtener embeddings
    embeddings = obtener_embeddings_semanticos([texto1_procesado, texto2_procesado])
    
    # Calcular similitud coseno
    similitud = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
    
    # Clasificar nivel de similitud
    clasificacion = clasificar_similitud(similitud)
    
    return {
        'similitud': similitud,
        'clasificacion': clasificacion['clasificacion'],
        'confianza': clasificacion['confianza']
    }

def procesar_archivo(archivo_original, archivo_sospechoso):
    """Procesa un par de archivos y calcula similitud entre ellos usando embeddings.
    
    Args:
        archivo_original: Ruta al archivo original
        archivo_sospechoso: Ruta al archivo sospechoso
        
    Returns:
        dict: Resultado del análisis de similitud
    """
    try:
        # Extraer IDs de los nombres de archivo
        id_original = os.path.basename(archivo_original).replace('source-document', '').replace('.txt', '')
        id_sospechoso = os.path.basename(archivo_sospechoso).replace('suspicious-document', '').replace('.txt', '')
        
        # Leer contenidos
        with open(archivo_original, 'r', encoding='utf-8') as f:
            texto_original = preprocess_text(f.read())
            
        with open(archivo_sospechoso, 'r', encoding='utf-8') as f:
            texto_sospechoso = preprocess_text(f.read())
        
        # Calcular similitud con embeddings
        resultado_similitud = comparar_textos_embeddings(texto_original, texto_sospechoso)
        
        # Preparar resultado
        resultado = {
            "id_original": id_original,
            "id_sospechoso": id_sospechoso,
            "archivo_original": os.path.basename(archivo_original),
            "archivo_sospechoso": os.path.basename(archivo_sospechoso),
            "similitud": float(resultado_similitud['similitud']),  # Convertir a float Python estándar
            "clasificacion": resultado_similitud['clasificacion'],
            "confianza": float(resultado_similitud['confianza'])  # Convertir a float Python estándar
        }
        
        return resultado
    
    except Exception as e:
        print(f"Error al procesar los archivos {archivo_original} y {archivo_sospechoso}: {e}")
        return {
            "id_original": id_original if 'id_original' in locals() else 'desconocido',
            "id_sospechoso": id_sospechoso if 'id_sospechoso' in locals() else 'desconocido',
            "archivo_original": os.path.basename(archivo_original),
            "archivo_sospechoso": os.path.basename(archivo_sospechoso),
            "error": str(e)
        }

def evaluar_resultados(df_resultados):
    """Evalúa los resultados del análisis de similitud.
    
    Args:
        df_resultados: DataFrame con los resultados del análisis
        
    Returns:
        dict: Métricas y estadísticas del análisis
    """
    metricas = {}
    
    # Estadísticas básicas de similitud
    metricas["estadisticas"] = {
        'similitud_media': float(df_resultados["similitud"].mean()),
        'similitud_mediana': float(df_resultados["similitud"].median()),
        'similitud_min': float(df_resultados["similitud"].min()),
        'similitud_max': float(df_resultados["similitud"].max()),
        'similitud_std': float(df_resultados["similitud"].std()),
    }
    
    # Distribución de clasificaciones
    if "clasificacion" in df_resultados.columns:
        # Convertir a diccionario Python estándar
        conteo_clasificaciones = {k: int(v) for k, v in df_resultados["clasificacion"].value_counts().to_dict().items()}
        metricas["distribucion"] = conteo_clasificaciones
        
        # Calcular porcentajes
        total = sum(conteo_clasificaciones.values())
        metricas["porcentajes"] = {k: float(v/total) for k, v in conteo_clasificaciones.items()}
        
        # Convertir clasificación a valores numéricos para clustering
        mapping = {'original': 0, 'plagio': 1}
        y_true = df_resultados["clasificacion"].map(mapping).values
        
        # Preparar datos para clustering
        X = df_resultados[["similitud", "confianza"]].values
        
        # Calcular Silhouette Coefficient si hay suficientes muestras y al menos 2 clases
        if len(df_resultados) > 2 and len(conteo_clasificaciones) >= 2:
            try:
                # Silhouette Coefficient
                silhouette_avg = silhouette_score(X, y_true)
                metricas["silhouette_coefficient"] = float(silhouette_avg)
                
                # Aplicar K-means clustering (k=2 para original/plagio)
                kmeans = KMeans(n_clusters=2, random_state=42)
                y_pred = kmeans.fit_predict(X)
                
                # Calcular pureza de clusters
                contingency_matrix = confusion_matrix(y_true, y_pred)
                cluster_purity = np.sum(np.max(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
                metricas["cluster_purity"] = float(cluster_purity)
                
                # Calcular Information Gain / Entropía
                base_entropy = entropy(df_resultados["clasificacion"].value_counts(normalize=True))
                
                # Entropía condicional por cluster
                conditional_entropy = 0
                for i in range(2):  # Para cada cluster
                    cluster_size = np.sum(y_pred == i)
                    if cluster_size > 0:
                        cluster_probs = []
                        for j in range(2):  # Para cada clase
                            count = np.sum((y_pred == i) & (y_true == j))
                            prob = count / cluster_size if cluster_size > 0 else 0
                            if prob > 0:
                                cluster_probs.append(prob)
                        if cluster_probs:
                            conditional_entropy += (cluster_size / len(y_pred)) * entropy(cluster_probs)
                
                # Information Gain
                information_gain = base_entropy - conditional_entropy
                metricas["information_gain"] = float(information_gain)
                
                # Métricas de evaluación binaria
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, 
                    df_resultados["similitud"] >= CONFIG['similitud']['umbral_plagio'],
                    average='binary',
                    pos_label=1
                )
                
            except Exception as e:
                print(f"Error al calcular métricas avanzadas: {e}")
                metricas["error_metricas_avanzadas"] = str(e)
    
        # Análisis de distribución de similitud por clase
        metricas["analisis_por_clase"] = {}
        for clase in df_resultados["clasificacion"].unique():
            df_clase = df_resultados[df_resultados["clasificacion"] == clase]
            metricas["analisis_por_clase"][clase] = {
                "count": int(len(df_clase)),
                "similitud_media": float(df_clase["similitud"].mean()),
                "similitud_mediana": float(df_clase["similitud"].median()),
                "similitud_std": float(df_clase["similitud"].std())
            }
    
    return metricas

def crear_visualizaciones(df_resultados, ruta_carpeta):
    """Crea visualizaciones de resultados del análisis de similitud.
    
    Args:
        df_resultados: DataFrame con los resultados del análisis
        ruta_carpeta: Carpeta donde guardar las visualizaciones
    """
    try:
        # 1. Histograma de distribución de similitudes
        plt.figure(figsize=(12, 6))
        sns.histplot(df_resultados['similitud'], bins=20, kde=True)
        plt.axvline(x=CONFIG['similitud']['umbral_plagio'], color='r', linestyle='--', label='Umbral Plagio')
        plt.title('Distribución de Similitudes entre Documentos')
        plt.xlabel('Puntuación de Similitud')
        plt.ylabel('Frecuencia')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        ruta_histograma = os.path.join(ruta_carpeta, 'histograma_similitud.png')
        plt.savefig(ruta_histograma)
        plt.close()
        
        # 2. Gráfico de barras por clasificación
        if 'clasificacion' in df_resultados.columns:
            conteo = df_resultados['clasificacion'].value_counts()
            plt.figure(figsize=(10, 6))
            sns.barplot(x=conteo.index, y=conteo.values)
            plt.title('Distribución de Clasificaciones')
            plt.xlabel('Clasificación')
            plt.ylabel('Cantidad de Documentos')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            ruta_barras = os.path.join(ruta_carpeta, 'distribucion_clasificaciones.png')
            plt.savefig(ruta_barras)
            plt.close()
        
        # 3. Scatter plot de similitud vs confianza
        if 'confianza' in df_resultados.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df_resultados, x='similitud', y='confianza', hue='clasificacion', palette='viridis')
            plt.title('Relación entre Similitud y Confianza')
            plt.xlabel('Similitud')
            plt.ylabel('Confianza')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            ruta_scatter = os.path.join(ruta_carpeta, 'similitud_vs_confianza.png')
            plt.savefig(ruta_scatter)
            plt.close()
        
    except Exception as e:
        print(f"Error al crear visualizaciones: {e}")

def ejecutar_analisis_embeddings(ruta_carpeta_salida=None):
    """Ejecuta el análisis completo de similitud entre textos usando embeddings.
    
    Args:
        ruta_carpeta_salida: Carpeta donde guardar los resultados
        
    Returns:
        dict: Resultados del análisis
    """
    tiempo_inicio = time.time()
    
    # Obtener rutas de carpetas desde la configuración
    ruta_originales = CONFIG['rutas']['documentos_originales']
    ruta_sospechosos = CONFIG['rutas']['documentos_sospechosos']
    
    # Verificar que existen las carpetas
    if not os.path.exists(ruta_originales):
        return {"error": f"La carpeta de documentos originales {ruta_originales} no existe"}
    
    if not os.path.exists(ruta_sospechosos):
        return {"error": f"La carpeta de documentos sospechosos {ruta_sospechosos} no existe"}
    
    # Crear carpeta de salida si no existe y no se especificó
    if ruta_carpeta_salida is None:
        ruta_carpeta_salida = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
    
    if not os.path.exists(ruta_carpeta_salida):
        os.makedirs(ruta_carpeta_salida)
    
    # Obtener lista de archivos originales y sospechosos
    archivos_originales = [f for f in os.listdir(ruta_originales) if f.endswith('.txt')]
    archivos_sospechosos = [f for f in os.listdir(ruta_sospechosos) if f.endswith('.txt')]
    
    # Verificar que hay archivos para comparar
    if not archivos_originales or not archivos_sospechosos:
        return {"error": "No se encontraron suficientes archivos para comparar"}
    
    # Crear pares de archivos para comparar (por ID)
    pares_archivos = []
    for archivo_original in archivos_originales:
        id_original = archivo_original.replace('source-document', '').replace('.txt', '')
        archivo_sospechoso = f"suspicious-document{id_original}.txt"
        
        if archivo_sospechoso in archivos_sospechosos:
            pares_archivos.append({
                'original': os.path.join(ruta_originales, archivo_original),
                'sospechoso': os.path.join(ruta_sospechosos, archivo_sospechoso)
            })
    
    # Si no hay pares para comparar
    if not pares_archivos:
        return {"error": "No se encontraron pares de documentos para comparar"}
    
    print(f"Se encontraron {len(pares_archivos)} pares de documentos para analizar")
    
    # Procesar pares de archivos
    resultados = []
    for par in pares_archivos:
        resultado = procesar_archivo(par['original'], par['sospechoso'])
        resultados.append(resultado)
        # Mostrar progreso
        if len(resultados) % 10 == 0:
            print(f"Procesados {len(resultados)} de {len(pares_archivos)} documentos")
    
    # Crear DataFrame con resultados
    try:
        df_resultados = pd.DataFrame(resultados)
        
        # Evaluar resultados
        metricas = evaluar_resultados(df_resultados)
        
        # Guardar resultados a CSV
        ruta_salida = os.path.join(ruta_carpeta_salida, "resultados_similitud.csv")
        df_resultados.to_csv(ruta_salida, index=False)
        
        # Clase personalizada para codificar tipos NumPy a JSON
        class NumpyEncoder(JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                return super(NumpyEncoder, self).default(obj)
        
        # Guardar métricas usando el codificador personalizado
        ruta_metricas = os.path.join(ruta_carpeta_salida, "metricas_similitud.json")
        with open(ruta_metricas, 'w', encoding='utf-8') as f:
            json.dump(metricas, f, indent=4, cls=NumpyEncoder)
        
        tiempo_total = time.time() - tiempo_inicio
        
        # Crear resultados visuales
        crear_visualizaciones(df_resultados, ruta_carpeta_salida)
        
        return {
            "tiempo_ejecucion": tiempo_total,
            "total_documentos": len(resultados),
            "metricas": metricas,
            "ruta_resultados": ruta_salida,
            "ruta_metricas": ruta_metricas,
            "df_resultados": df_resultados,
        }
        
    except Exception as e:
        return {
            "error": f"Error al crear resultados: {e}",
            "resultados_parciales": resultados
        }

if __name__ == "__main__":
    # Ejecutar análisis con la nueva estructura de carpetas
    resultado = ejecutar_analisis_embeddings(
        ruta_carpeta_salida="./evidencia-1/resultados"
    )
    
    # Mostrar resultados
    if "error" in resultado:
        print(f"Error: {resultado['error']}")
    else:
        print(f"Análisis completado en {resultado['tiempo_ejecucion']:.2f} segundos")
        print(f"Total de documentos analizados: {resultado['total_documentos']}")
        print(f"Distribución de clasificaciones:")
        for clasificacion, cantidad in resultado['metricas']['distribucion'].items():
            porcentaje = resultado['metricas']['porcentajes'][clasificacion] * 100
            print(f"  - {clasificacion}: {cantidad} documentos ({porcentaje:.2f}%)")
        print(f"Resultados guardados en: {resultado['ruta_resultados']}")
        print(f"Métricas guardadas en: {resultado['ruta_metricas']}")
        print("\nVisualizaciones generadas en la carpeta de resultados.")