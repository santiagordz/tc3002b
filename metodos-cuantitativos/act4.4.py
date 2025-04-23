import pandas as pd
import re
import math
import os
from collections import defaultdict

def limpiar_texto(texto):
    if isinstance(texto, str):
        texto = re.sub(r'[^\w\s]', '', texto)
        return texto.lower()
    else:
        return ''

def leer_archivo(ruta_archivo):
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error al leer {ruta_archivo}: {e}")
        return None

def create_bag_of_words_vector(texto):
    palabras = texto.split()
    frecuencias = defaultdict(int)
    for palabra in palabras:
        frecuencias[palabra] += 1
    return frecuencias

def calcular_similitud_coseno_bow(texto1, texto2):
    vector1_dict = create_bag_of_words_vector(texto1)
    vector2_dict = create_bag_of_words_vector(texto2)

    todas_palabras = list(set(vector1_dict.keys()) | set(vector2_dict.keys()))

    vector1_lista = [vector1_dict.get(palabra, 0) for palabra in todas_palabras]
    vector2_lista = [vector2_dict.get(palabra, 0) for palabra in todas_palabras]

    producto_punto = sum(vector1_lista[i] * vector2_lista[i] for i in range(len(vector1_lista)))

    magnitud1 = math.sqrt(sum(valor**2 for valor in vector1_lista))
    magnitud2 = math.sqrt(sum(valor**2 for valor in vector2_lista))

    if magnitud1 == 0 or magnitud2 == 0:
        cos = 0
    else:
        cos = producto_punto / (magnitud1 * magnitud2)

    return cos

def calcular_tf_idf(texto1, texto2):
    vector1_dict = create_bag_of_words_vector(texto1)
    vector2_dict = create_bag_of_words_vector(texto2)

    todas_palabras = list(set(vector1_dict.keys()) | set(vector2_dict.keys()))
    num_documentos = 2

    # Calcular TF
    tf_vector1 = {palabra: count / len(texto1.split()) if texto1 else 0 for palabra, count in vector1_dict.items()}
    tf_vector2 = {palabra: count / len(texto2.split()) if texto2 else 0 for palabra, count in vector2_dict.items()}

    # Calcular IDF
    idf = {}
    for palabra in todas_palabras:
        count = 0
        if palabra in vector1_dict:
            count += 1
        if palabra in vector2_dict:
            count += 1
        idf[palabra] = math.log(num_documentos / (count + 1)) + 1

    # Calcular TF-IDF
    tfidf_vector1 = [tf_vector1.get(palabra, 0) * idf.get(palabra, 0) for palabra in todas_palabras]
    tfidf_vector2 = [tf_vector2.get(palabra, 0) * idf.get(palabra, 0) for palabra in todas_palabras]

    # Calcular similitud coseno
    producto_punto = sum(tfidf_vector1[i] * tfidf_vector2[i] for i in range(len(tfidf_vector1)))
    magnitud1 = math.sqrt(sum(valor**2 for valor in tfidf_vector1))
    magnitud2 = math.sqrt(sum(valor**2 for valor in tfidf_vector2))
    cos_tfidf = producto_punto / (magnitud1 * magnitud2) if magnitud1 and magnitud2 else 0

    return cos_tfidf

def crear_matriz_markov(texto):
    palabras = texto.split()
    if not palabras:
        return {}
    matriz = defaultdict(lambda: defaultdict(int))
    for i in range(len(palabras) - 1):
        actual = palabras[i]
        siguiente = palabras[i+1]
        matriz[actual][siguiente] += 1

    # Convertir a probabilidades
    for actual in matriz:
        total = sum(matriz[actual].values())
        for siguiente in matriz[actual]:
            matriz[actual][siguiente] /= total
    return matriz

def vectorizar_matriz_markov(matriz):
    vector = []
    for actual in matriz:
        for siguiente in matriz[actual]:
            vector.append(matriz[actual][siguiente])
    return vector

def calcular_similitud_coseno_markov(texto1, texto2):
    matriz1 = crear_matriz_markov(texto1)
    matriz2 = crear_matriz_markov(texto2)

    vector1 = vectorizar_matriz_markov(matriz1)
    vector2 = vectorizar_matriz_markov(matriz2)

    # Asegurar que ambos vectores tengan la misma longitud (rellenando con 0 si es necesario)
    len1 = len(vector1)
    len2 = len(vector2)
    max_len = max(len1, len2)

    if max_len == 0:  # Si ambos vectores están vacíos
        return 0

    vector1_padded = vector1 + [0] * (max_len - len1)
    vector2_padded = vector2 + [0] * (max_len - len2)

    producto_punto = sum(vector1_padded[i] * vector2_padded[i] for i in range(max_len))
    magnitud1 = math.sqrt(sum(valor**2 for valor in vector1_padded))
    magnitud2 = math.sqrt(sum(valor**2 for valor in vector2_padded))

    if magnitud1 == 0 or magnitud2 == 0:
        cos = 0
    else:
        cos = producto_punto / (magnitud1 * magnitud2)

    return cos

def clasificar_similaridad(score):
    if 0.85 <= score <= 1.00:
        return "high"
    elif 0.45 <= score < 0.85:
        return "medium"
    else:  # 0 <= score < 0.45
        return "low"
        
def extraer_nivel_similaridad(nombre_archivo):
    if "high" in nombre_archivo.lower():
        return "high"
    elif "medium" in nombre_archivo.lower() or "moderate" in nombre_archivo.lower():
        return "medium"
    elif "low" in nombre_archivo.lower():
        return "low"
    else:
        return None

def main():
    # Directorio donde están los archivos
    directorio = "./texts/"  # Ajusta según donde tengas los archivos
    
    # Leer el archivo original
    archivo_original = "original.txt"
    texto_original = leer_archivo(os.path.join(directorio, archivo_original))
    
    if texto_original is None:
        print(f"No se pudo leer el archivo original: {archivo_original}")
        return
    
    # Limpiar el texto original
    texto_original_limpio = limpiar_texto(texto_original)
    
    # Buscar archivos similares
    archivos_similares = [f for f in os.listdir(directorio) 
                         if f.endswith(".txt") and f != archivo_original]
    
    # Preparar resultados
    resultados = []
    conteo_aciertos = {"BOW": 0, "TFIDF": 0, "Markov": 0}
    
    # Procesar cada archivo similar
    for archivo_similar in archivos_similares:
        texto_similar = leer_archivo(os.path.join(directorio, archivo_similar))
        
        if texto_similar is None:
            continue
            
        # Limpiar el texto similar
        texto_similar_limpio = limpiar_texto(texto_similar)
        
        # Extraer el nivel de similaridad del nombre del archivo
        similaridad_real = extraer_nivel_similaridad(archivo_similar)
        
        # Calcular similitudes
        cos_bow = calcular_similitud_coseno_bow(texto_original_limpio, texto_similar_limpio)
        cos_tfidf = calcular_tf_idf(texto_original_limpio, texto_similar_limpio)
        cos_markov = calcular_similitud_coseno_markov(texto_original_limpio, texto_similar_limpio)
        
        # Clasificar según los valores de similitud
        bow_clase = clasificar_similaridad(cos_bow)
        tfidf_clase = clasificar_similaridad(cos_tfidf)
        markov_clase = clasificar_similaridad(cos_markov)
        
        # Determinar si cada método acertó
        bow_acerto = bow_clase == similaridad_real
        tfidf_acerto = tfidf_clase == similaridad_real
        markov_acerto = markov_clase == similaridad_real
        
        # Actualizar conteo de aciertos
        if bow_acerto:
            conteo_aciertos["BOW"] += 1
        if tfidf_acerto:
            conteo_aciertos["TFIDF"] += 1
        if markov_acerto:
            conteo_aciertos["Markov"] += 1
        
        # Guardar resultados
        resultados.append({
            "Nombre_texto_original": archivo_original,
            "Nombre_texto_similar": archivo_similar,
            "Grado_similitud_precargado": similaridad_real,
            "Coseno_BOW": cos_bow,
            "Coseno_TFIDF": cos_tfidf,
            "Coseno_Markov": cos_markov,
            "BOW_acerto": bow_acerto,
            "TFIDF_acerto": tfidf_acerto,
            "Markov_acerto": markov_acerto
        })
    
    # Crear DataFrame y guardar CSV
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv("resultados_similitud.csv", index=False)
    
    # Generar reporte
    total_archivos = len(archivos_similares)
    reporte = f"""
    REPORTE DE ANÁLISIS DE SIMILITUD DE TEXTOS
    ==========================================
    
    Total de archivos analizados: {total_archivos}
    
    CONTEO DE ACIERTOS POR MÉTODO:
    - Bag of Words (BOW): {conteo_aciertos["BOW"]} de {total_archivos} ({conteo_aciertos["BOW"]/total_archivos*100:.2f}%)
    - TF-IDF: {conteo_aciertos["TFIDF"]} de {total_archivos} ({conteo_aciertos["TFIDF"]/total_archivos*100:.2f}%)
    - Cadenas de Markov: {conteo_aciertos["Markov"]} de {total_archivos} ({conteo_aciertos["Markov"]/total_archivos*100:.2f}%)
    
    CONCLUSIONES:
    El método con mayor tasa de acierto es: {max(conteo_aciertos, key=conteo_aciertos.get)}
    
    Análisis detallado:
    - BOW: Es un método simple que cuenta la frecuencia de palabras sin considerar su orden.
    - TF-IDF: Mejora el BOW al dar más peso a palabras menos comunes y menos peso a palabras muy frecuentes.
    - Markov: Considera el orden de las palabras, capturando mejor el estilo y la estructura del texto.
    
    Para este conjunto específico de textos, los resultados sugieren que el método 
    {max(conteo_aciertos, key=conteo_aciertos.get)} es el más efectivo para detectar 
    correctamente el nivel de similitud.
    """
    
    # Guardar el reporte
    with open("reporte_similitud.txt", "w") as f:
        f.write(reporte)
    
    print("Análisis completado. CSV y reporte generados.")

if __name__ == "__main__":
    main()