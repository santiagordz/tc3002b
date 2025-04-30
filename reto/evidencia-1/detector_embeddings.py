import os
import numpy as np
import pandas as pd
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import tensorflow as tf
import tensorflow_hub as hub
import json

# Configuración global
CONFIG = {
    'umbrales': {
        'alto': 0.90,
        'medio': 0.60
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
            if os.path.exists('../modelos/use_model'):
                EMBEDDINGS_MODEL = hub.load('../modelos/use_model')
            else:
                # Descargar y guardar modelo
                EMBEDDINGS_MODEL = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
                # Crear directorio si no existe
                if not os.path.exists('../modelos'):
                    os.makedirs('../modelos')
                # Guardar para uso futuro
                tf.saved_model.save(EMBEDDINGS_MODEL, '../modelos/use_model')
            
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
    """Clasifica el nivel de similitud basado en umbrales."""
    if sim >= CONFIG['umbrales']['alto']:
        return 'high'
    elif sim >= CONFIG['umbrales']['medio']:
        return 'medium'
    else:
        return 'low'

def comparar_textos_embeddings(texto1, texto2):
    """Compara dos textos usando embeddings semánticos."""
    texto1_procesado = preprocess_text(texto1)
    texto2_procesado = preprocess_text(texto2)
    
    # Obtener embeddings
    embeddings = obtener_embeddings_semanticos([texto1_procesado, texto2_procesado])
    
    # Calcular similitud coseno
    similitud = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    # Clasificar nivel de similitud
    nivel = clasificar_similitud(similitud)
    
    return {
        'similitud': similitud,
        'nivel': nivel
    }

def procesar_archivo(nombre_archivo, ruta_carpeta, texto_original):
    """Procesa un archivo de texto y calcula similitud con el original usando embeddings."""
    
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
        
        # Calcular similitud con embeddings
        resultado_similitud = comparar_textos_embeddings(texto_original, texto_similar)
        
        # Preparar resultado
        resultado = {
            "original": "original.txt",
            "similar": nombre_archivo,
            "precargado": precargado,
            "similitud_embeddings": resultado_similitud['similitud'],
            "nivel_predicho": resultado_similitud['nivel'],
            "acierto": resultado_similitud['nivel'] == precargado
        }
        
        return resultado
    
    except Exception as e:
        print(f"Error al procesar el archivo {nombre_archivo}: {e}")
        return {
            "original": "original.txt",
            "similar": nombre_archivo,
            "error": str(e)
        }

def evaluar_modelo(df_resultados):
    """Evalúa el modelo con múltiples métricas."""
    metricas = {}
    
    # Evaluar el método de embeddings
    if "nivel_predicho" in df_resultados.columns:
        y_true = df_resultados["precargado"]
        y_pred = df_resultados["nivel_predicho"]
        
        # Calcular precisión, recall, f1 para cada clase
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=['high', 'medium', 'low']
        )
        
        # Calcular exactitud
        exactitud = (y_true == y_pred).mean()
        
        metricas["EMBEDDINGS"] = {
            'exactitud': exactitud,
            'precision': dict(zip(['high', 'medium', 'low'], precision)),
            'recall': dict(zip(['high', 'medium', 'low'], recall)),
            'f1': dict(zip(['high', 'medium', 'low'], f1)),
            'matriz_confusion': confusion_matrix(
                y_true, y_pred, labels=['high', 'medium', 'low']
            ).tolist()
        }
    
    return metricas

def crear_visualizaciones(df_resultados, ruta_carpeta):
    """Crea visualizaciones de resultados."""
    try:
        # 1. Gráfico de similitudes por documento
        plt.figure(figsize=(12, 6))
        plt.bar(df_resultados['similar'], df_resultados['similitud_embeddings'], color='blue', alpha=0.7)
        plt.axhline(y=CONFIG['umbrales']['alto'], color='r', linestyle='--', label='Umbral Alto')
        plt.axhline(y=CONFIG['umbrales']['medio'], color='orange', linestyle='--', label='Umbral Medio')
        plt.title('Similitud por Documento usando Embeddings')
        plt.xlabel('Documento')
        plt.ylabel('Puntuación de Similitud')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        ruta_grafico = os.path.join(ruta_carpeta, 'grafico_similitud_embeddings.png')
        plt.savefig(ruta_grafico)
        plt.close()
        
        # 2. Matriz de confusión
        if 'nivel_predicho' in df_resultados.columns and 'precargado' in df_resultados.columns:
            cm = confusion_matrix(
                df_resultados['precargado'], 
                df_resultados['nivel_predicho'],
                labels=['high', 'medium', 'low']
            )
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['high', 'medium', 'low'],
                       yticklabels=['high', 'medium', 'low'])
            plt.title('Matriz de Confusión - Método de Embeddings')
            plt.xlabel('Predicción')
            plt.ylabel('Real')
            plt.tight_layout()
            
            ruta_matriz = os.path.join(ruta_carpeta, 'matriz_confusion_embeddings.png')
            plt.savefig(ruta_matriz)
            plt.close()
        
    except Exception as e:
        print(f"Error al crear visualizaciones: {e}")

def ejecutar_analisis_embeddings(ruta_carpeta_textos, ruta_carpeta_salida):
    """Ejecuta el análisis completo de similitud entre textos usando embeddings."""
    tiempo_inicio = time.time()
    
    # Verificar que existe la carpeta de textos
    if not os.path.exists(ruta_carpeta_textos):
        return {
            "error": f"La carpeta {ruta_carpeta_textos} no existe"
        }
    
    # Verificar que existe el archivo original
    archivo_original = os.path.join(ruta_carpeta_textos, "original.txt")
    if not os.path.exists(archivo_original):
        return {
            "error": f"No se encontró el archivo original.txt en {ruta_carpeta_textos}"
        }
    
    # Crear carpeta de salida si no existe
    if not os.path.exists(ruta_carpeta_salida):
        os.makedirs(ruta_carpeta_salida)
    
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
    for nombre_archivo in os.listdir(ruta_carpeta_textos):
        if nombre_archivo == 'original.txt' or not nombre_archivo.endswith('.txt'):
            continue
        archivos.append(nombre_archivo)
    
    # Si no hay archivos para comparar
    if not archivos:
        return {
            "error": "No se encontraron archivos para comparar con el original"
        }
    
    # Procesar archivos
    resultados = []
    for nombre in archivos:
        resultado = procesar_archivo(nombre, ruta_carpeta_textos, texto_original)
        resultados.append(resultado)
    
    # Crear DataFrame con resultados
    try:
        df_resultados = pd.DataFrame(resultados)
        
        # Contar aciertos
        aciertos = df_resultados["acierto"].sum() if "acierto" in df_resultados.columns else 0
        
        # Evaluar modelo
        metricas = evaluar_modelo(df_resultados)
        
        # Guardar resultados a CSV
        ruta_salida = os.path.join(ruta_carpeta_salida, "resultados_embeddings.csv")
        df_resultados.to_csv(ruta_salida, index=False)
        
        # Guardar métricas
        ruta_metricas = os.path.join(ruta_carpeta_salida, "metricas_embeddings.json")
        with open(ruta_metricas, 'w', encoding='utf-8') as f:
            json.dump(metricas, f, indent=4)
        
        tiempo_total = time.time() - tiempo_inicio
        
        # Crear resultados visuales
        crear_visualizaciones(df_resultados, ruta_carpeta_salida)
        
        return {
            "tiempo_ejecucion": tiempo_total,
            "total_documentos": len(resultados),
            "aciertos": aciertos,
            "exactitud": aciertos / len(resultados) if len(resultados) > 0 else 0,
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

if __name__ == "__main__":
    # Ejecutar análisis
    resultado = ejecutar_analisis_embeddings(
        ruta_carpeta_textos="/Users/santiago/School/tc3002b/reto/textos",
        ruta_carpeta_salida="."
    )
    
    # Mostrar resultados
    if "error" in resultado:
        print(f"Error: {resultado['error']}")
    else:
        print(f"Análisis completado en {resultado['tiempo_ejecucion']:.2f} segundos")
        print(f"Total de documentos analizados: {resultado['total_documentos']}")
        print(f"Exactitud del modelo: {resultado['exactitud']:.2%}")
        print(f"Resultados guardados en: {resultado['ruta_resultados']}")