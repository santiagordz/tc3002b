import pandas as pd
import re
import math
from collections import defaultdict

def leer_csv(nombre_archivo):
    df = pd.read_csv(nombre_archivo, nrows=100)
    return df

def limpiar_texto(texto):
  if isinstance(texto, str):
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto.lower()
  else:
    return ''

def limpiar_datos(df):
  df['question1'] = df['question1'].apply(limpiar_texto)
  df['question2'] = df['question2'].apply(limpiar_texto)
  return df

def create_bag_of_words_vector(texto):
    palabras = texto.split()
    frecuencias = defaultdict(int)
    for palabra in palabras:
        frecuencias[palabra] += 1
    return frecuencias

def calcular_similitud_coseno_bow(row):
  vector1_dict = create_bag_of_words_vector(row['question1'])
  vector2_dict = create_bag_of_words_vector(row['question2'])

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

  return pd.Series([vector1_lista, vector2_lista, cos])

def calcular_tf_idf(row):
  vector1_dict = create_bag_of_words_vector(row['question1'])
  vector2_dict = create_bag_of_words_vector(row['question2'])

  todas_palabras = list(set(vector1_dict.keys()) | set(vector2_dict.keys()))
  num_documentos = 2

  # Calcular TF
  tf_vector1 = {palabra: count / len(row['question1'].split()) if row['question1'] else 0 for palabra, count in vector1_dict.items()}
  tf_vector2 = {palabra: count / len(row['question2'].split()) if row['question2'] else 0 for palabra, count in vector2_dict.items()}

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

  return pd.Series([tfidf_vector1, tfidf_vector2, cos_tfidf])

def crear_matriz_markov(texto):
    palabras = texto.split()
    if not palabras:
        return []
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

def calcular_similitud_coseno_markov(row):
    matriz1 = crear_matriz_markov(row['question1'])
    matriz2 = crear_matriz_markov(row['question2'])

    vector1 = vectorizar_matriz_markov(matriz1)
    vector2 = vectorizar_matriz_markov(matriz2)

    # Asegurar que ambos vectores tengan la misma longitud (rellenando con 0 si es necesario)
    len1 = len(vector1)
    len2 = len(vector2)
    max_len = max(len1, len2)

    vector1_padded = vector1 + [0] * (max_len - len1)
    vector2_padded = vector2 + [0] * (max_len - len2)

    producto_punto = sum(vector1_padded[i] * vector2_padded[i] for i in range(max_len))
    magnitud1 = math.sqrt(sum(valor**2 for valor in vector1_padded))
    magnitud2 = math.sqrt(sum(valor**2 for valor in vector2_padded))

    if magnitud1 == 0 or magnitud2 == 0:
        cos = 0
    else:
        cos = producto_punto / (magnitud1 * magnitud2)

    return pd.Series([vector1_padded, vector2_padded, cos])

nombre_archivo = 'questions.csv'
data = leer_csv(nombre_archivo)

if data is not None:
    datos_limpios = limpiar_datos(data.copy())

    # Similitud BoW
    resultados_bow = datos_limpios.apply(calcular_similitud_coseno_bow, axis=1)
    datos_limpios[['q1_vecBoW', 'q2_vecBoW', 'cos_BOW']] = resultados_bow

    # Similitud TF-IDF
    resultados_tfidf = datos_limpios.apply(calcular_tf_idf, axis=1)
    datos_limpios[['q1_vecTFIDF', 'q2_vecTFIDF', 'cos_TFID']] = resultados_tfidf

    # Similitud Cadenas de Markov
    resultados_markov = datos_limpios.apply(calcular_similitud_coseno_markov, axis=1)
    datos_limpios[['q1_vecMark', 'q2_vecMark', 'cos_MARK']] = resultados_markov

    # Guardar resultados
    datos_limpios.to_csv('results.csv', index=False)
    print("Archivo 'resultados_similitud.csv' creado exitosamente.")
