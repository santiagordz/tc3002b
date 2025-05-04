import os
import numpy as np
import pandas as pd
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import tensorflow_hub as hub
import json
from json import JSONEncoder
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.cluster import KMeans, DBSCAN
from scipy.stats import entropy
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

nltk.download('punkt_tab')
lemmatizer = WordNetLemmatizer()

# Simplified global configuration
CONFIG = {
    # Similarity analysis configuration
    'similitud': {
        'umbral_plagio': 0.75,  # Threshold for plagiarism
    },
    # Weights for different similarity methods
    'pesos': {
        'bow': 0.15,
        'tfidf': 0.20,
        'semantico': 0.25,
        'ngrama_palabra': 0.15,
        'ngrama_caracter': 0.20,
        'markov': 0.05
    },
    # Folder paths
    'rutas': {
        'documentos_originales': './Dokumen Teks/Original',
        'documentos_sospechosos': './Dokumen Teks/Copy'
    }
}

# Cache for embeddings model
EMBEDDINGS_MODEL = None

def preprocess_text(texto):
    """
    Text preprocessing with NLTK lemmatization.
    
    Args:
        texto: Input text
        
    Returns:
        str: Preprocessed and lemmatized text
    """
    if texto is None:
        return ''
    
    try:
        # Normalize spaces
        texto = re.sub(r'\s+', ' ', texto)
        
        # Keep only alphanumeric characters and spaces for general analysis
        texto_procesado = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚüÜñÑ0-9\s]', '', texto)
        texto_procesado = texto_procesado.lower().strip()
        
        # Tokenize the text
        tokens = word_tokenize(texto_procesado)
        
        # Lemmatize each token
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back into a single string
        texto_lemmatizado = ' '.join(lemmatized_tokens)
        
        return texto_lemmatizado
        
    except Exception as e:
        print(f"Error in text preprocessing: {e}")
        # Return the basic preprocessing as fallback
        return texto.lower().strip() if texto else ''

def cargar_modelo_embeddings():
    """Loads the semantic embeddings model (Universal Sentence Encoder)."""
    global EMBEDDINGS_MODEL
    
    if EMBEDDINGS_MODEL is None:
        print("Loading semantic embeddings model...")
        
        try:
            # Try to use previously downloaded model
            if os.path.exists('./modelos/nnlm'):
                print("Loading embeddings model from local folder...")
                EMBEDDINGS_MODEL = hub.load('./modelos/use_model')
            else:
                print("Model not found. Downloading...")
                # Download and save model
                EMBEDDINGS_MODEL = hub.load("https://tfhub.dev/google/nnlm-en-dim128/2")
                # Create directory if it doesn't exist
                if not os.path.exists('./modelos'):
                    os.makedirs('./modelos')
                # Save for future use
                tf.saved_model.save(EMBEDDINGS_MODEL, './modelos/nnlm')
            
            print("Embeddings model loaded successfully.")
        except Exception as e:
            print(f"Error loading embeddings model: {e}")
            print("Using a simplified alternative model...")
            # Simple fallback model
            EMBEDDINGS_MODEL = "FALLBACK"
    
    return EMBEDDINGS_MODEL

def obtener_embeddings_semanticos(textos):
    """Gets semantic embeddings for the provided texts."""
    modelo = cargar_modelo_embeddings()
    
    if modelo == "FALLBACK":
        # Fallback to random vectors for demonstration
        return np.random.rand(len(textos), 512)
    
    try:
        # Ensure texts are not empty
        textos = [t if t.strip() else "empty text" for t in textos]
        
        # Truncate very long texts to avoid memory issues
        textos = [t[:100000] for t in textos]
        
        # Get embeddings
        embeddings = modelo(textos).numpy()
        return embeddings
    
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        # Fallback to random vectors for demonstration
        return np.random.rand(len(textos), 512)

def crear_matriz_markov(texto, n=2):
    """Creates a transition matrix from n-grams of characters."""
    # Preprocess text for Markov
    texto = ''.join(c for c in texto.lower() if c.isalpha() or c.isspace())
    
    if len(texto) <= n:
        return np.zeros((27, 27))  # 26 letters + space
    
    # Map characters to indices (26 for space)
    char_a_idx = lambda c: 26 if c == ' ' else (ord(c) - ord('a') if 'a' <= c <= 'z' else 0)
    
    matriz = np.zeros((27, 27))
    for i in range(len(texto) - 1):
        a, b = char_a_idx(texto[i]), char_a_idx(texto[i + 1])
        if 0 <= a < 27 and 0 <= b < 27:
            matriz[a][b] += 1
            
    # Normalize
    sumas_filas = matriz.sum(axis=1, keepdims=True)
    matriz = np.divide(matriz, sumas_filas, out=np.zeros_like(matriz), where=sumas_filas != 0)
    return matriz

def cosine_sim_dense(v1, v2):
    """Calculates cosine similarity between dense matrices."""
    if np.sum(v1) == 0 or np.sum(v2) == 0:
        return 0.0
    return np.dot(v1.flatten(), v2.flatten()) / (np.linalg.norm(v1.flatten()) * np.linalg.norm(v2.flatten()))

def calcular_similitud_ngrama(texto1, texto2, rango_n=(2, 3), analizador='word'):
    """Calculates n-gram based similarity."""
    # Use character or word n-grams
    vectorizador = CountVectorizer(analyzer=analizador, ngram_range=rango_n)
    
    try:
        vectores = vectorizador.fit_transform([texto1, texto2])
        if vectores[0].nnz == 0 or vectores[1].nnz == 0:  # Check if there are non-zero elements
            return 0.0
        similitud = cosine_similarity(vectores[0], vectores[1])[0, 0]
        return similitud
    except Exception as e:
        print(f"Error in n-gram calculation: {e}")
        return 0.0

def clasificar_similitud(sim):
    """Classifies the similarity level based on thresholds.
    
    Args:
        sim: Similarity value between 0 and 1
        
    Returns:
        dict: Dictionary with classification and confidence level
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

def calcular_similitud_combinada(texto_original, texto_similar, pesos=None):
    """Calculates combined similarity using multiple metrics with weights."""
    if pesos is None:
        pesos = CONFIG['pesos']
    
    resultados = {}
    
    # Calculate individual similarities
    
    # BOW vectorization
    vectorizer_bow = CountVectorizer()
    try:
        vectores_bow = vectorizer_bow.fit_transform([texto_original, texto_similar])
        cos_bow = cosine_similarity(vectores_bow[0], vectores_bow[1])[0, 0]
    except:
        cos_bow = 0
    resultados['cos_BOW'] = cos_bow
    
    # TF-IDF vectorization
    vectorizer_tfidf = TfidfVectorizer()
    try:
        vectores_tfidf = vectorizer_tfidf.fit_transform([texto_original, texto_similar])
        cos_tfidf = cosine_similarity(vectores_tfidf[0], vectores_tfidf[1])[0, 0]
    except:
        cos_tfidf = 0
    resultados['cos_TFIDF'] = cos_tfidf
    
    # Semantic similarity
    try:
        embeddings_semanticos = obtener_embeddings_semanticos([texto_original, texto_similar])
        cos_semantico = cosine_similarity([embeddings_semanticos[0]], [embeddings_semanticos[1]])[0][0]
    except:
        cos_semantico = 0
    resultados['cos_SEMANTICO'] = cos_semantico
    
    # N-gram similarities
    cos_ngrama_palabra = calcular_similitud_ngrama(texto_original, texto_similar, rango_n=(2, 3), analizador='word')
    resultados['cos_NGRAMA_PALABRA'] = cos_ngrama_palabra
    
    cos_ngrama_caracter = calcular_similitud_ngrama(texto_original, texto_similar, rango_n=(3, 5), analizador='char')
    resultados['cos_NGRAMA_CARACTER'] = cos_ngrama_caracter
    
    # Markov similarities
    markov_orig = crear_matriz_markov(texto_original)
    markov_sim = crear_matriz_markov(texto_similar)
    cos_markov = cosine_sim_dense(markov_orig, markov_sim)
    resultados['cos_MARKOV'] = cos_markov
    
    # Combine all similarities with weights
    puntuacion_combinada = (
        pesos['bow'] * cos_bow +
        pesos['tfidf'] * cos_tfidf +
        pesos['semantico'] * cos_semantico +
        pesos['ngrama_palabra'] * cos_ngrama_palabra +
        pesos['ngrama_caracter'] * cos_ngrama_caracter +
        pesos['markov'] * cos_markov
    )
    resultados['similitud'] = puntuacion_combinada
    
    # Classify the combined similarity
    clasificacion = clasificar_similitud(puntuacion_combinada)
    resultados['clasificacion'] = clasificacion['clasificacion']
    resultados['confianza'] = clasificacion['confianza']
    
    return resultados

def procesar_archivo(archivo_original, archivo_sospechoso):
    """Processes a pair of files and calculates similarity between them using multiple methods.
    
    Args:
        archivo_original: Path to original file
        archivo_sospechoso: Path to suspicious file
        
    Returns:
        dict: Result of similarity analysis
    """
    try:
        # Extract IDs from filenames
        id_original = os.path.basename(archivo_original).replace('source-document', '').replace('.txt', '')
        id_sospechoso = os.path.basename(archivo_sospechoso).replace('suspicious-document', '').replace('.txt', '')
        
        # Read contents
        with open(archivo_original, 'r', encoding='utf-8') as f:
            texto_original = preprocess_text(f.read())
            
        with open(archivo_sospechoso, 'r', encoding='utf-8') as f:
            texto_sospechoso = preprocess_text(f.read())
        
        # Calculate combined similarity with multiple methods
        resultado_similitud = calcular_similitud_combinada(texto_original, texto_sospechoso)
        
        # Prepare result
        resultado = {
            "id_original": id_original,
            "id_sospechoso": id_sospechoso,
            "archivo_original": os.path.basename(archivo_original),
            "archivo_sospechoso": os.path.basename(archivo_sospechoso),
            "similitud": float(resultado_similitud['similitud']),  # Convert to standard Python float
            "clasificacion": resultado_similitud['clasificacion'],
            "confianza": float(resultado_similitud['confianza']),  # Convert to standard Python float
            "cos_BOW": float(resultado_similitud['cos_BOW']),
            "cos_TFIDF": float(resultado_similitud['cos_TFIDF']),
            "cos_SEMANTICO": float(resultado_similitud['cos_SEMANTICO']),
            "cos_NGRAMA_PALABRA": float(resultado_similitud['cos_NGRAMA_PALABRA']),
            "cos_NGRAMA_CARACTER": float(resultado_similitud['cos_NGRAMA_CARACTER']),
            "cos_MARKOV": float(resultado_similitud['cos_MARKOV'])
        }
        
        return resultado
    
    except Exception as e:
        print(f"Error processing files {archivo_original} and {archivo_sospechoso}: {e}")
        return {
            "id_original": id_original if 'id_original' in locals() else 'unknown',
            "id_sospechoso": id_sospechoso if 'id_sospechoso' in locals() else 'unknown',
            "archivo_original": os.path.basename(archivo_original),
            "archivo_sospechoso": os.path.basename(archivo_sospechoso),
            "error": str(e)
        }

def evaluar_resultados(df_resultados):
    """Evaluates the similarity analysis results with advanced clustering metrics.
    
    Args:
        df_resultados: DataFrame with analysis results
        
    Returns:
        dict: Metrics and statistics from the analysis
    """
    metricas = {}
    
    # Basic similarity statistics
    metricas["estadisticas"] = {
        'similitud_media': float(df_resultados["similitud"].mean()),
        'similitud_mediana': float(df_resultados["similitud"].median()),
        'similitud_min': float(df_resultados["similitud"].min()),
        'similitud_max': float(df_resultados["similitud"].max()),
        'similitud_std': float(df_resultados["similitud"].std()),
    }
    
    # Classification distribution
    if "clasificacion" in df_resultados.columns:
        # Convert to standard Python dictionary
        conteo_clasificaciones = {k: int(v) for k, v in df_resultados["clasificacion"].value_counts().to_dict().items()}
        metricas["distribucion"] = conteo_clasificaciones
        
        # Calculate percentages
        total = sum(conteo_clasificaciones.values())
        metricas["porcentajes"] = {k: float(v/total) for k, v in conteo_clasificaciones.items()}
        
        # Convertir clasificación a valores numéricos para clustering
        mapping = {'original': 0, 'plagio': 1}
        y_true = df_resultados["clasificacion"].map(mapping).values
        
        # Preparar datos para clustering
        X = df_resultados[["similitud", "confianza"]].values
        
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
            
        metricas["analisis_por_clase"] = {}
        for clase in df_resultados["clasificacion"].unique():
            df_clase = df_resultados[df_resultados["clasificacion"] == clase]
            metricas["analisis_por_clase"][clase] = {
                "count": int(len(df_clase)),
                "similitud_media": float(df_clase["similitud"].mean()),
                "similitud_mediana": float(df_clase["similitud"].median()),
                "similitud_std": float(df_clase["similitud"].std())
            }
            
    # Statistics for each similarity method
    metodos = ['BOW', 'TFIDF', 'SEMANTICO', 'NGRAMA_PALABRA', 'NGRAMA_CARACTER', 'MARKOV']
    metricas["metodos"] = {}
    
    for metodo in metodos:
        col = f"cos_{metodo}"
        if col in df_resultados.columns:
            metricas["metodos"][metodo] = {
                'media': float(df_resultados[col].mean()),
                'mediana': float(df_resultados[col].median()),
                'min': float(df_resultados[col].min()),
                'max': float(df_resultados[col].max()),
                'std': float(df_resultados[col].std()),
            }
    
    # ADD ADVANCED CLUSTERING METRICS
    if "clasificacion" in df_resultados.columns and len(df_resultados) > 2:
        try:
            # Create feature matrix for clustering evaluation
            X = np.array(df_resultados[['similitud'] + [f'cos_{m}' for m in metodos if f'cos_{m}' in df_resultados.columns]])
            
            # Convert classifications to numeric labels
            mapping = {'original': 0, 'plagio': 1}
            y_true = np.array([mapping.get(c, 0) for c in df_resultados["clasificacion"]])
            
            # Apply K-means clustering (k=2 for original/plagiarism)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            y_pred_kmeans = kmeans.fit_predict(X)
            
            # Try DBSCAN for density-based clustering
            dbscan = DBSCAN(eps=0.3, min_samples=3)
            y_pred_dbscan = dbscan.fit_predict(X)
            
            # Only calculate DBSCAN metrics if we have at least 2 clusters (not all noise)
            if len(np.unique([y for y in y_pred_dbscan if y >= 0])) >= 2:
                metricas["clustering"] = {
                    "dbscan_num_clusters": int(len(np.unique([y for y in y_pred_dbscan if y >= 0]))),
                    "dbscan_noise_points": int(np.sum(y_pred_dbscan == -1))
                }
            
            # Create a 2D feature matrix for visualization (similitud + TFIDF or the most important feature)
            most_important_feature = 'cos_TFIDF' if 'cos_TFIDF' in df_resultados.columns else 'cos_SEMANTICO'
            X_2d = df_resultados[['similitud', most_important_feature]].values
            
            conf_matrix = confusion_matrix(y_true, y_pred_kmeans)
            # We may need to flip the clusters if kmeans assigned different label numbers
            if np.sum(conf_matrix.diagonal()) < np.sum(conf_matrix - np.diag(conf_matrix.diagonal())):
                y_pred_kmeans = 1 - y_pred_kmeans
                conf_matrix = confusion_matrix(y_true, y_pred_kmeans)
            
            # Store confusion matrix values in metrics
            if conf_matrix.shape == (2, 2):
                tn, fp, fn, tp = conf_matrix.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
                
            metricas["matriz_confusion_kmeans"] = {
                "verdadero_negativo": int(tn),
                "falso_positivo": int(fp),
                "falso_negativo": int(fn),
                "verdadero_positivo": int(tp)
            }
            
            # Calculate clustering accuracy
            clustering_accuracy = np.sum(y_true == y_pred_kmeans) / len(y_true)
            metricas["clustering_accuracy"] = float(clustering_accuracy)
            
            # Precision, recall, F1 score
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred_kmeans, average='binary', zero_division=0
            )
            
            metricas["metricas_clustering"] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            }
            
            # Feature importance (based on K-means centroids)
            feature_names = ['similitud'] + [f'cos_{m}' for m in metodos if f'cos_{m}' in df_resultados.columns]
            centroid_diff = np.abs(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1])
            feature_importance = dict(zip(feature_names, centroid_diff / np.sum(centroid_diff)))
            metricas["feature_importance"] = {k: float(v) for k, v in feature_importance.items()}
            
        except Exception as e:
            print(f"Error calculating advanced clustering metrics: {e}")
            metricas["error_clustering"] = str(e)
    
        # Analysis of distribution by class
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
    """Creates enhanced visualizations of similarity analysis results.
    
    Args:
        df_resultados: DataFrame with analysis results
        ruta_carpeta: Folder where to save visualizations
    """
    try:
        # 1. Similarity distribution histogram
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
        
        # 2. Classification bar chart
        if 'clasificacion' in df_resultados.columns:
            conteo = df_resultados['clasificacion'].value_counts()
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=conteo.index, y=conteo.values)
            plt.title('Distribución de Clasificaciones')
            plt.xlabel('Clasificación')
            plt.ylabel('Cantidad de Documentos')
            
            # Add percentage and count labels
            total = sum(conteo)
            for i, p in enumerate(ax.patches):
                height = p.get_height()
                percentage = height / total * 100
                ax.text(p.get_x() + p.get_width()/2., height + 0.3, 
                       f'{int(height)} ({percentage:.1f}%)', 
                       ha="center", fontsize=12)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            ruta_barras = os.path.join(ruta_carpeta, 'distribucion_clasificaciones.png')
            plt.savefig(ruta_barras)
            plt.close()
        
        # 3. Similarity vs confidence scatter plot with decision boundary
        if 'confianza' in df_resultados.columns:
            plt.figure(figsize=(10, 6))
            scatter = sns.scatterplot(data=df_resultados, x='similitud', y='confianza', s=80, alpha=0.7)
            
            # Add decision boundary line
            x_min, x_max = plt.xlim()
            umbral = CONFIG['similitud']['umbral_plagio']
            plt.axvline(x=umbral, color='r', linestyle='--', label=f'Umbral ({umbral})')
            
            plt.title('Relación entre Similitud y Confianza con Clasificación')
            plt.xlabel('Similitud')
            plt.ylabel('Confianza')
            plt.grid(True, alpha=0.3)
            plt.legend(title="Clasificación")
            plt.tight_layout()
            
            ruta_scatter = os.path.join(ruta_carpeta, 'similitud_vs_confianza.png')
            plt.savefig(ruta_scatter)
            plt.close()
            
        # 4. Comparison of methods
        plt.figure(figsize=(14, 8))
        
        metodos = ['BOW', 'TFIDF', 'SEMANTICO', 'NGRAMA_PALABRA', 'NGRAMA_CARACTER', 'MARKOV']
        colores = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
        
        # Create id for each comparison
        df_resultados['id_comparacion'] = df_resultados.apply(
            lambda row: f"{row['archivo_original']} vs {row['archivo_sospechoso']}", axis=1)
        
        # Sort by combined similarity for better visualization
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
        
        # Add combined similarity line
        plt.plot(range(len(df_ordenado)), df_ordenado['similitud'], 
                 color='black', linewidth=2, label='Similitud Combinada')
        # Add plagiarism threshold
        plt.axhline(y=CONFIG['similitud']['umbral_plagio'], color='r', 
                    linestyle='--', label=f'Umbral ({CONFIG["similitud"]["umbral_plagio"]})')
        
        plt.title('Comparación de Similitudes por Método')
        plt.xlabel('Comparación (ordenado por similitud combinada)')
        plt.ylabel('Puntuación de Similitud')
        plt.xticks([])  # Hide x-axis labels
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        
        ruta_comparacion = os.path.join(ruta_carpeta, 'grafico_comparacion_metodos.png')
        plt.savefig(ruta_comparacion)
        plt.close()
        
        # 5. NEW: Feature correlation heatmap
        cols_to_include = ['similitud'] + [f'cos_{metodo}' for metodo in metodos if f'cos_{metodo}' in df_resultados.columns]
        corr_matrix = df_resultados[cols_to_include].corr()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                   square=True, linewidths=.5, annot=True, fmt=".2f")
        plt.title('Correlación entre Métodos de Similitud')
        plt.tight_layout()
        
        ruta_correlacion = os.path.join(ruta_carpeta, 'correlacion_metodos.png')
        plt.savefig(ruta_correlacion)
        plt.close()
        
        # 6. NEW: Violin plots for each method by classification
        if 'clasificacion' in df_resultados.columns:
            plt.figure(figsize=(14, 10))
            metrics_to_plot = [col for col in df_resultados.columns if col.startswith('cos_') or col == 'similitud']
            
            melted_df = pd.melt(df_resultados, 
                                id_vars=['clasificacion'], 
                                value_vars=metrics_to_plot,
                                var_name='Método', 
                                value_name='Puntuación')
            
            # Replace method names for better display
            melted_df['Método'] = melted_df['Método'].str.replace('cos_', '').str.title()
            melted_df.loc[melted_df['Método'] == 'Similitud', 'Método'] = 'Combinada'
            
            sns.violinplot(x='Método', y='Puntuación', 
                          data=melted_df, split=True, inner='quart',
            )
            
            plt.title('Distribución de Puntuaciones por Método y Clasificación')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.legend(title='Clasificación')
            plt.tight_layout()
            
            ruta_violin = os.path.join(ruta_carpeta, 'violin_plot_metodos.png')
            plt.savefig(ruta_violin)
            plt.close()
        
        # 7. NEW: 2D Clustering visualization
        if 'clasificacion' in df_resultados.columns and len(df_resultados) > 2:
            try:
                # Choose the two most informative features for visualization
                # Typically similarity and the most effective method (TFIDF or Semantic)
                if 'cos_TFIDF' in df_resultados.columns and 'cos_SEMANTICO' in df_resultados.columns:
                    x_col = 'cos_TFIDF'
                    y_col = 'cos_SEMANTICO'
                else:
                    # Fallback to first two available methods
                    available_cols = [col for col in df_resultados.columns if col.startswith('cos_')]
                    if len(available_cols) >= 2:
                        x_col, y_col = available_cols[:2]
                    else:
                        x_col = 'similitud'
                        y_col = available_cols[0] if available_cols else 'confianza'
                
                # Create 2D feature space
                X_2d = df_resultados[[x_col, y_col]].values
                
                # True labels
                mapping = {'original': 0, 'plagio': 1}
                y_true = np.array([mapping.get(c, 0) for c in df_resultados["clasificacion"]])
                
                # Apply K-means clustering
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                y_pred = kmeans.fit_predict(X_2d)
                
                # We may need to swap cluster labels for consistency (0=original, 1=plagio)
                # Compare cluster centers and swap if needed
                if kmeans.cluster_centers_[0].mean() > kmeans.cluster_centers_[1].mean():
                    y_pred = 1 - y_pred
                
                # Create plot
                plt.figure(figsize=(12, 10))
                
                # Create a custom colormap for the background decision boundary
                cmap_light = LinearSegmentedColormap.from_list(
                    'cmap_light', ['#AAFFAA', '#FFAAAA'], N=100)
                
                # Plot decision boundary
                h = 0.02  # step size in the mesh
                x_min, x_max = X_2d[:, 0].min() - 0.1, X_2d[:, 0].max() + 0.1
                y_min, y_max = X_2d[:, 1].min() - 0.1, X_2d[:, 1].max() + 0.1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                     np.arange(y_min, y_max, h))
                Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.3)
                
                # Plot data points with true labels (outer color) and predicted clusters (inner color)
                markers = ['o', 's']  # circle for original, square for plagio
                colors_true = ['blue', 'red']  # colors for true labels
                
                for i, (label, color) in enumerate(zip(['Original', 'Plagio'], colors_true)):
                    idx = y_true == i
                    plt.scatter(X_2d[idx, 0], X_2d[idx, 1], c=color, 
                              marker=markers[i], s=80, label=f'True: {label}', 
                              edgecolor='k', alpha=0.7)
                
                # Plot cluster centers
                plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                          marker='x', s=200, linewidths=3, color='black',
                          label='Centroides K-means')
                
                # Plot decision threshold line
                plt.axvline(x=CONFIG['similitud']['umbral_plagio'], color='purple', 
                         linestyle='--', label=f'Umbral ({CONFIG["similitud"]["umbral_plagio"]})')
                
                plt.title('Clustering y Clasificación en Espacio 2D')
                plt.xlabel(x_col.replace('cos_', '').title())
                plt.ylabel(y_col.replace('cos_', '').title())
                plt.legend(loc='best')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                ruta_clustering = os.path.join(ruta_carpeta, 'clustering_2d.png')
                plt.savefig(ruta_clustering)
                plt.close()
                
                # 8. NEW: Confusion matrix visualization
                conf_matrix = confusion_matrix(y_true, y_pred)
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                         xticklabels=['Original', 'Plagio'],
                         yticklabels=['Original', 'Plagio'])
                plt.title('Matriz de Confusión: Etiquetas Reales vs Clustering')
                plt.ylabel('Etiqueta Real')
                plt.xlabel('Cluster Asignado')
                plt.tight_layout()
                
                ruta_confusion = os.path.join(ruta_carpeta, 'matriz_confusion.png')
                plt.savefig(ruta_confusion)
                plt.close()
                
            except Exception as e:
                print(f"Error creating clustering visualizations: {e}")
        
        # 9. NEW: Feature importance visualization based on clustering
        try:
            if 'clasificacion' in df_resultados.columns:
                # Choose features for importance analysis
                feature_cols = ['similitud'] + [col for col in df_resultados.columns if col.startswith('cos_')]
                X_features = df_resultados[feature_cols].values
                
                # Apply K-means clustering
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                kmeans.fit(X_features)
                
                # Calculate feature importance from cluster centroids
                centroid_diff = np.abs(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1])
                importance = centroid_diff / np.sum(centroid_diff)
                
                # Create importance DataFrame
                feature_names = [col.replace('cos_', '').title() for col in feature_cols]
                feature_names = [name if name != 'Similitud' else 'Combinada' for name in feature_names]
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                })
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importance_df)
                plt.title('Importancia de Características para Detección de Plagio')
                plt.xlabel('Importance Score')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                ruta_importancia = os.path.join(ruta_carpeta, 'feature_importance.png')
                plt.savefig(ruta_importancia)
                plt.close()
                
            # 10. NEW: Silhouette plot
            if 'clasificacion' in df_resultados.columns and len(df_resultados) > 2:
                from sklearn.metrics import silhouette_samples
                
                # Use similarity and one or two additional features
                if 'cos_TFIDF' in df_resultados.columns and 'cos_SEMANTICO' in df_resultados.columns:
                    X_sil = df_resultados[['similitud', 'cos_TFIDF', 'cos_SEMANTICO']].values
                else:
                    X_sil = df_resultados[['similitud'] + 
                                        [col for col in df_resultados.columns if col.startswith('cos_')][:2]].values
                
                # Convert classifications to numeric 
                mapping = {'original': 0, 'plagio': 1}
                cluster_labels = np.array([mapping.get(c, 0) for c in df_resultados["clasificacion"]])
                
                # Calculate silhouette scores for each sample
                silhouette_vals = silhouette_samples(X_sil, cluster_labels)

                # Create silhouette plot
                plt.figure(figsize=(10, 8))
                y_lower, y_upper = 0, 0
                
                for i, cluster in enumerate([0, 1]):
                    cluster_silhouette_vals = silhouette_vals[cluster_labels == cluster]
                    cluster_silhouette_vals.sort()
                    
                    size_cluster_i = cluster_silhouette_vals.shape[0]
                    y_upper = y_lower + size_cluster_i
                    
                    color = plt.cm.nipy_spectral(float(i) / 2)
                    plt.fill_betweenx(np.arange(y_lower, y_upper),
                                     0, cluster_silhouette_vals,
                                     facecolor=color, edgecolor=color, alpha=0.7)
                    
                    # Label the silhouette plots with cluster numbers
                    label = 'Original' if cluster == 0 else 'Plagio'
                    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, label)
                    y_lower = y_upper + 10
                
                # Add vertical line for average silhouette score
                plt.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--")
                
                plt.title('Análisis de Silhouette por Clasificación')
                plt.xlabel('Coeficientes de Silhouette')
                plt.ylabel('Cluster')
                plt.yticks([])  # Clear y-axis labels
                plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
                plt.tight_layout()
                
                ruta_silhouette = os.path.join(ruta_carpeta, 'silhouette_plot.png')
                plt.savefig(ruta_silhouette)
                plt.close()
                
        except Exception as e:
            print(f"Error creating feature importance visualization: {e}")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def ejecutar_analisis_embeddings(ruta_carpeta_salida=None):
    """Runs the complete similarity analysis between texts using multiple methods.
    
    Args:
        ruta_carpeta_salida: Folder where to save results
        
    Returns:
        dict: Analysis results
    """
    tiempo_inicio = time.time()
    
    # Get folder paths from configuration
    ruta_originales = CONFIG['rutas']['documentos_originales']
    ruta_sospechosos = CONFIG['rutas']['documentos_sospechosos']
    
    # Check that folders exist
    if not os.path.exists(ruta_originales):
        return {"error": f"The original documents folder {ruta_originales} does not exist"}
    
    if not os.path.exists(ruta_sospechosos):
        return {"error": f"The suspicious documents folder {ruta_sospechosos} does not exist"}
    
    # Create output folder if it doesn't exist and none is specified
    if ruta_carpeta_salida is None:
        ruta_carpeta_salida = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
    
    if not os.path.exists(ruta_carpeta_salida):
        os.makedirs(ruta_carpeta_salida)
    
    # Get list of original and suspicious files
    archivos_originales = [f for f in os.listdir(ruta_originales) if f.endswith('.txt')]
    archivos_sospechosos = [f for f in os.listdir(ruta_sospechosos) if f.endswith('.txt')]
    
    # Check that there are files to compare
    if not archivos_originales or not archivos_sospechosos:
        return {"error": "Not enough files found to compare"}
    
    # Create file pairs to compare (by ID)
    pares_archivos = []
    for archivo_original in archivos_originales:
        id_original = archivo_original.replace('source-document', '').replace('.txt', '')
        archivo_sospechoso = f"suspicious-document{id_original}.txt"
        
        if archivo_sospechoso in archivos_sospechosos:
            pares_archivos.append({
                'original': os.path.join(ruta_originales, archivo_original),
                'sospechoso': os.path.join(ruta_sospechosos, archivo_sospechoso)
            })
    
    # If there are no pairs to compare
    if not pares_archivos:
        return {"error": "No document pairs found to compare"}
    
    print(f"Found {len(pares_archivos)} document pairs to analyze")
    
    # Process file pairs
    resultados = []
    for par in pares_archivos:
        resultado = procesar_archivo(par['original'], par['sospechoso'])
        resultados.append(resultado)
        # Show progress
        if len(resultados) % 10 == 0:
            print(f"Processed {len(resultados)} of {len(pares_archivos)} documents")
    
    # Create DataFrame with results
    try:
        df_resultados = pd.DataFrame(resultados)
        
        # Evaluate results
        metricas = evaluar_resultados(df_resultados)
        
        # Save results to CSV
        ruta_salida = os.path.join(ruta_carpeta_salida, "resultados_similitud.csv")
        df_resultados.to_csv(ruta_salida, index=False)
        
        # Custom class to encode NumPy types to JSON
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
        
        # Save metrics using custom encoder
        ruta_metricas = os.path.join(ruta_carpeta_salida, "metricas_similitud.json")
        with open(ruta_metricas, 'w', encoding='utf-8') as f:
            json.dump(metricas, f, indent=4, cls=NumpyEncoder)
        
        tiempo_total = time.time() - tiempo_inicio
        
        # Create visual results
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
            "error": f"Error creating results: {e}",
            "resultados_parciales": resultados
        }

if __name__ == "__main__":
    # Run analysis with the new folder structure
    print("Starting analysis...")
    resultado = ejecutar_analisis_embeddings(
        ruta_carpeta_salida="./resultados"
    )
    
    # Show results
    if "error" in resultado:
        print(f"Error: {resultado['error']}")
    else:
        print(f"Analysis completed in {resultado['tiempo_ejecucion']:.2f} seconds")
        print(f"Total documents analyzed: {resultado['total_documentos']}")
        print(f"Classification distribution:")
        for clasificacion, cantidad in resultado['metricas']['distribucion'].items():
            porcentaje = resultado['metricas']['porcentajes'][clasificacion] * 100
            print(f"  - {clasificacion}: {cantidad} documents ({porcentaje:.2f}%)")
        print(f"Results saved to: {resultado['ruta_resultados']}")
        print(f"Metrics saved to: {resultado['ruta_metricas']}")
        print("\nVisualizations generated in the results folder.")