import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from detector_plagio import (
    comparar_textos, ejecutar_analisis, 
    cargar_configuracion, guardar_configuracion
)

# Configuración de página
st.set_page_config(
    page_title="Sistema de Detección de Infracciones de Derechos de Autor",
    page_icon="📝",
    layout="wide"
)

# Estilos CSS
st.markdown("""
<style>
    .main {
        padding: 1rem;
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stButton button {
        width: 100%;
        background-color: #0066CC;
        color: white;
    }
    .info-box {
        background-color: #2D2D2D;
        border: 1px solid #404040;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: #FFFFFF;
    }
    .result-box {
        background-color: #2D2D2D;
        border: 1px solid #404040;
        border-radius: 5px;
        padding: 1rem;
        margin-top: 1rem;
        color: #FFFFFF;
    }
    .warning {
        color: #FF6B6B;
        font-weight: bold;
    }
    .success {
        color: #4CAF50;
        font-weight: bold;
    }
    code {
        background-color: #363636;
        color: #E0E0E0;
        padding: 2px 6px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# Título
st.title("📝 Sistema de Detección de Infracciones de Derechos de Autor")

# Cargar configuración
CONFIG = cargar_configuracion()

# Menú lateral
with st.sidebar:
    st.header("Opciones")
    
    # Selección de modo
    modo = st.radio(
        "Modo de análisis",
        ["Análisis de textos individuales", "Análisis de corpus completo"]
    )
    
    # Configuración avanzada
    st.subheader("Configuración avanzada")
    show_config = st.checkbox("Mostrar configuración avanzada", False)
    
    if show_config:
        st.subheader("Umbrales de similitud")
        umbral_plagio = st.slider(
            "Umbral para detección de plagio", 
            min_value=0.5, 
            max_value=1.0, 
            value=CONFIG['umbrales']['plagio'],
            step=0.01
        )
        umbral_sospechoso = st.slider(
            "Umbral para contenido sospechoso", 
            min_value=0.1, 
            max_value=0.8, 
            value=CONFIG['umbrales']['sospechoso'],
            step=0.01
        )
        
        st.subheader("Pesos para métodos")
        peso_bow = st.slider("Peso BoW", 0.0, 1.0, CONFIG['pesos']['bow'], 0.05)
        peso_tfidf = st.slider("Peso TF-IDF", 0.0, 1.0, CONFIG['pesos']['tfidf'], 0.05)
        peso_semantico = st.slider("Peso Semántico", 0.0, 1.0, CONFIG['pesos']['semantico'], 0.05)
        peso_ngrama_palabra = st.slider("Peso N-grama (Palabra)", 0.0, 1.0, CONFIG['pesos']['ngrama_palabra'], 0.05)
        peso_ngrama_caracter = st.slider("Peso N-grama (Carácter)", 0.0, 1.0, CONFIG['pesos']['ngrama_caracter'], 0.05)
        peso_markov = st.slider("Peso Markov", 0.0, 1.0, CONFIG['pesos']['markov'], 0.05)
        peso_estilo = st.slider("Peso Estilo", 0.0, 1.0, CONFIG['pesos']['estilo'], 0.05)
        
        # Normalizar pesos
        pesos_total = (peso_bow + peso_tfidf + peso_semantico + 
                      peso_ngrama_palabra + peso_ngrama_caracter + 
                      peso_markov + peso_estilo)
        
        if pesos_total > 0:
            peso_bow = peso_bow / pesos_total
            peso_tfidf = peso_tfidf / pesos_total
            peso_semantico = peso_semantico / pesos_total
            peso_ngrama_palabra = peso_ngrama_palabra / pesos_total
            peso_ngrama_caracter = peso_ngrama_caracter / pesos_total
            peso_markov = peso_markov / pesos_total
            peso_estilo = peso_estilo / pesos_total
        
        # Mostrar distribución de pesos
        st.subheader("Distribución de pesos")
        fig, ax = plt.subplots(figsize=(6, 3))
        pesos = [peso_bow, peso_tfidf, peso_semantico, peso_ngrama_palabra, 
                peso_ngrama_caracter, peso_markov, peso_estilo]
        etiquetas = ['BoW', 'TF-IDF', 'Semántico', 'N-grama (Pal.)', 
                    'N-grama (Car.)', 'Markov', 'Estilo']
        ax.barh(etiquetas, pesos)
        ax.set_xlim(0, max(pesos) * 1.1)
        ax.set_xlabel('Peso normalizado')
        st.pyplot(fig)
        
        # Guardar configuración
        if st.button("Guardar configuración"):
            nueva_config = {
                'umbrales': {
                    'plagio': umbral_plagio,
                    'sospechoso': umbral_sospechoso
                },
                'pesos': {
                    'bow': peso_bow,
                    'tfidf': peso_tfidf,
                    'semantico': peso_semantico,
                    'ngrama_palabra': peso_ngrama_palabra,
                    'ngrama_caracter': peso_ngrama_caracter,
                    'markov': peso_markov,
                    'estilo': peso_estilo
                }
            }
            
            if guardar_configuracion(nueva_config):
                st.success("Configuración guardada correctamente")
                CONFIG = nueva_config
            else:
                st.error("Error al guardar la configuración")

# Análisis de textos individuales
if modo == "Análisis de textos individuales":
    st.header("Análisis de similitud entre dos textos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Texto Original")
        texto_original = st.text_area(
            "Ingrese el texto original",
            height=250,
            placeholder="Pegue aquí el texto original..."
        )
    
    with col2:
        st.subheader("Texto a Comparar")
        texto_comparar = st.text_area(
            "Ingrese el texto a comparar",
            height=250,
            placeholder="Pegue aquí el texto a comparar..."
        )
    
    # Botón para analizar
    if st.button("🔍 Analizar Similitud"):
        if not texto_original.strip() or not texto_comparar.strip():
            st.warning("⚠️ Por favor, ingrese ambos textos para comparar.")
        else:
            with st.spinner("Analizando similitud entre textos..."):
                # Realizar análisis
                resultados = comparar_textos(texto_original, texto_comparar)
                
                # Mostrar resultados
                st.subheader("Resultados del Análisis")
                
                # Nivel de similitud
                nivel = resultados['nivel_similitud'].upper()
                color = {
                    'PLAGIO': 'red',
                    'SOSPECHOSO': 'orange',
                    'ORIGINAL': 'green'
                }.get(nivel, 'blue')
                
                # Descripción del nivel
                descripcion = {
                    'PLAGIO': "⚠️ **PLAGIO DETECTADO**: Los textos presentan un nivel significativo de similitud que podría constituir una posible infracción de derechos de autor.",
                    'SOSPECHOSO': "⚠️ **CONTENIDO SOSPECHOSO**: Los textos presentan algunas similitudes. Se recomienda revisar con detalle.",
                    'ORIGINAL': "✅ **CONTENIDO ORIGINAL**: Los textos son suficientemente diferentes."
                }.get(nivel, "")
                
                # Mostrar nivel y puntuación
                st.markdown(f"""
                <div class="result-box">
                    <h3 style="color:{color};">Nivel de Similitud: {nivel}</h3>
                    <p>{descripcion}</p>
                    <h4>Puntuación Combinada: {resultados['similitud_combinada']:.4f}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Mostrar detalle de similitudes
                st.subheader("Desglose por Método")
                
                # Crear gráfico
                fig, ax = plt.subplots(figsize=(10, 6))
                
                metodos = [
                    'BOW', 'TF-IDF', 'Semántica', 'N-grama (Palabra)', 
                    'N-grama (Carácter)', 'Markov', 'Estilo', 'Combinada'
                ]
                
                valores = [
                    resultados['similitud_bow'],
                    resultados['similitud_tfidf'],
                    resultados['similitud_semantica'],
                    resultados['similitud_ngrama_palabra'],
                    resultados['similitud_ngrama_caracter'],
                    resultados['similitud_markov'],
                    resultados['similitud_estilo'],
                    resultados['similitud_combinada']
                ]
                
                # Umbrales para las líneas horizontales
                umbral_plagio = CONFIG['umbrales']['plagio']
                umbral_sospechoso = CONFIG['umbrales']['sospechoso']
                
                # Crear barras
                bars = ax.bar(metodos, valores, color='skyblue')
                
                # Añadir líneas de umbral
                ax.axhline(y=umbral_plagio, color='red', linestyle='--', alpha=0.7, label='Umbral de plagio')
                ax.axhline(y=umbral_sospechoso, color='orange', linestyle='--', alpha=0.7, label='Umbral de sospecha')
                
                # Etiquetas
                ax.set_ylabel('Puntuación de Similitud')
                ax.set_title('Comparación de Métodos de Similitud')
                ax.set_ylim(0, 1.05)
                
                # Rotar etiquetas del eje x
                plt.xticks(rotation=45, ha='right')
                
                # Añadir valores sobre las barras
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{height:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Tabla de resultados
                st.subheader("Detalles de Similitud")
                df_resultados = pd.DataFrame({
                    'Método': metodos,
                    'Puntuación': valores
                })
                st.dataframe(df_resultados)
                
                # Recomendaciones
                st.subheader("Interpretación y Recomendaciones")
                if nivel == 'PLAGIO':
                    st.markdown("""
                    📌 **Interpretación**: Los textos muestran una similitud muy alta que podría indicar:
                    - Copia directa o con modificaciones mínimas
                    - Parafraseo manteniendo la estructura y contenido principal
                    
                    📋 **Recomendaciones**:
                    - Revisar detalladamente ambos textos para identificar secciones copiadas
                    - Verificar si existe atribución de la fuente original
                    - Considerar si existe una posible infracción de derechos de autor
                    """)
                elif nivel == 'SOSPECHOSO':
                    st.markdown("""
                    📌 **Interpretación**: Los textos muestran una similitud moderada que podría indicar:
                    - Parafraseo extendido del contenido original
                    - Uso de ideas y estructura similar con diferente redacción
                    - Posible inspiración en el texto original
                    
                    📋 **Recomendaciones**:
                    - Revisar qué aspectos son similares (ideas, estructura, frases específicas)
                    - Evaluar si la similitud es coincidencia o derivación intencional
                    - Considerar si es necesario citar o atribuir la fuente original
                    """)
                else:  # ORIGINAL
                    st.markdown("""
                    📌 **Interpretación**: Los textos muestran una similitud baja que podría indicar:
                    - Textos con temas relacionados pero desarrollo independiente
                    - Diferentes enfoques sobre un tema similar
                    - Textos originales sin relación directa
                    
                    📋 **Recomendaciones**:
                    - No se requieren acciones específicas desde la perspectiva de derechos de autor
                    - Cualquier similitud probablemente se debe a tema común o coincidencia
                    """)

# Análisis de corpus completo
else:
    st.header("Análisis de corpus completo")
    
    st.markdown("""
    <div class="info-box">
        <h4>Instrucciones:</h4>
        <p>Para analizar un corpus completo, debe tener una carpeta con la siguiente estructura:</p>
        <ul>
            <li><code>Original/</code>: Carpeta con los textos originales (ej: source-document01501.txt)</li>
            <li><code>Copy/</code>: Carpeta con los textos a comparar (ej: suspicious-document01501.txt)</li>
        </ul>
        <p>El sistema comparará documentos seleccionados de ambas carpetas para detectar posibles similitudes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Selector de carpeta
    ruta_carpeta = st.text_input(
        "Ruta de la carpeta principal con los textos",
        placeholder="Ej: /Users/santiago/School/tc3002b/reto/Dokumen Teks"
    )
    
    # Configuración del análisis
    col1, col2 = st.columns(2)
    
    with col1:
        max_comparaciones = st.slider(
            "Número máximo de comparaciones",
            min_value=5,
            max_value=100,
            value=25,
            step=5,
            help="Limita el número total de comparaciones para mejorar el rendimiento"
        )
    
    # Botón para analizar corpus
    if st.button("🔍 Analizar Corpus"):
        if not ruta_carpeta or not os.path.exists(ruta_carpeta):
            st.warning("⚠️ Por favor, ingrese una ruta de carpeta válida.")
        else:
            # Verificar estructura de carpetas
            ruta_original = os.path.join(ruta_carpeta, "Original")
            ruta_copy = os.path.join(ruta_carpeta, "Copy")
            
            if not os.path.exists(ruta_original) or not os.path.exists(ruta_copy):
                st.error("⚠️ La carpeta debe contener las subcarpetas 'Original' y 'Copy'")
            else:
                with st.spinner(f"Analizando corpus (máximo {max_comparaciones} comparaciones)... Esto puede tomar varios minutos."):
                    # Realizar análisis con límite de comparaciones
                    resultados = ejecutar_analisis(ruta_carpeta, max_comparaciones=max_comparaciones)
                
                if "error" in resultados:
                    st.error(f"⚠️ Error: {resultados['error']}")
                else:
                    # Mostrar resultados
                    st.subheader("Resultados del Análisis")
                    
                    # Información general
                    st.markdown(f"""
                    <div class="result-box">
                        <h3>Resumen de Análisis</h3>
                        <p>Total de comparaciones realizadas: <b>{resultados['total_documentos']}</b></p>
                        <p>Tiempo de ejecución: <b>{resultados['tiempo_ejecucion']:.2f} segundos</b></p>
                        <p>Resultados guardados en: <code>{resultados['ruta_resultados']}</code></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mostrar visualizaciones
                    st.subheader("Visualizaciones")
                    
                    # Distribución de similitudes
                    ruta_dist = os.path.join(ruta_carpeta, 'distribucion_similitudes.png')
                    if os.path.exists(ruta_dist):
                        st.image(ruta_dist, caption="Distribución de Puntuaciones de Similitud")
                    
                    # Mapa de calor
                    ruta_heatmap = os.path.join(ruta_carpeta, 'mapa_calor_similitud.png')
                    if os.path.exists(ruta_heatmap):
                        st.image(ruta_heatmap, caption="Mapa de Calor de Similitud entre Documentos")
                    
                    # Comparación de métodos
                    ruta_comparacion = os.path.join(ruta_carpeta, 'grafico_comparacion_metodos.png')
                    if os.path.exists(ruta_comparacion):
                        st.image(ruta_comparacion, caption="Comparación de Similitudes por Método")
                    
                    # Clustering si está disponible
                    ruta_clusters = os.path.join(ruta_carpeta, 'clustering_documentos.png')
                    if os.path.exists(ruta_clusters):
                        st.subheader("Análisis de Clustering")
                        st.image(ruta_clusters, caption="Clustering de Documentos por Similitud")
                        
                        ruta_resumen = os.path.join(ruta_carpeta, 'resumen_clusters.png')
                        if os.path.exists(ruta_resumen):
                            st.image(ruta_resumen, caption="Similitud Media por Cluster")
                    
                    # Mostrar resultados detallados
                    st.subheader("Resultados Detallados")
                    
                    # Mostrar DataFrame de resultados
                    if 'df_resultados' in resultados:
                        df = resultados['df_resultados']
                        
                        # Seleccionar columnas relevantes para mostrar
                        cols_mostrar = []
                        for col in ['original', 'similar', 'similitud', 'nivel', 'cluster']:
                            if col in df.columns:
                                cols_mostrar.append(col)
                        
                        # Añadir columnas de similitudes específicas si existen
                        for col in df.columns:
                            if col.startswith('cos_') and col != 'cos_COMBINADO' and col in df.columns:
                                cols_mostrar.append(col)
                        
                        # Asegurarse que todas las columnas existan
                        cols_disponibles = [col for col in cols_mostrar if col in df.columns]
                        df_mostrar = df[cols_disponibles].copy()
                        
                        # Renombrar columnas para mejor visualización
                        nombres_cols = {
                            'original': 'Archivo Original',
                            'similar': 'Archivo Comparado', 
                            'similitud': 'Similitud Combinada',
                            'nivel': 'Nivel Detectado',
                            'cluster': 'Cluster',
                            'cos_BOW': 'Similitud BoW',
                            'cos_TFIDF': 'Similitud TF-IDF',
                            'cos_SEMANTICO': 'Similitud Semántica',
                            'cos_NGRAMA_PALABRA': 'Similitud N-grama Palabra',
                            'cos_NGRAMA_CARACTER': 'Similitud N-grama Carácter',
                            'cos_MARKOV': 'Similitud Markov',
                            'cos_ESTILO': 'Similitud Estilo'
                        }
                        
                        df_mostrar.rename(columns={col: nombres_cols.get(col, col) for col in df_mostrar.columns}, inplace=True)
                        
                        # Aplicar formato, destacando niveles sospechosos o de plagio si existe la columna
                        if 'Nivel Detectado' in df_mostrar.columns:
                            st.dataframe(df_mostrar.style.applymap(
                                lambda x: 'background-color: rgba(255, 0, 0, 0.2)' if x == 'plagio' else 
                                        ('background-color: rgba(255, 165, 0, 0.2)' if x == 'sospechoso' else ''),
                                subset=['Nivel Detectado']
                            ))
                        else:
                            st.dataframe(df_mostrar)
                        
                        # Opción para descargar resultados completos
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar Resultados Completos (CSV)",
                            data=csv,
                            file_name='resultados_completos.csv',
                            mime='text/csv',
                        )
                        
                        # Análisis de los resultados
                        st.subheader("Análisis de los Resultados")
                        
                        # Mostrar estadísticas de similitud
                        if 'similitud' in df.columns:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Similitud Media", f"{df['similitud'].mean():.3f}")
                            
                            with col2:
                                st.metric("Similitud Máxima", f"{df['similitud'].max():.3f}")
                            
                            with col3:
                                st.metric("Similitud Mínima", f"{df['similitud'].min():.3f}")
                        
                        # Mostrar distribución de niveles si existe
                        if 'nivel' in df.columns:
                            dist_niveles = df['nivel'].value_counts()
                            
                            st.subheader("Distribución de Niveles de Similitud")
                            
                            # Gráfico de barras para niveles
                            fig, ax = plt.subplots(figsize=(10, 6))
                            colores = {'plagio': 'red', 'sospechoso': 'orange', 'original': 'green'}
                            
                            for nivel in dist_niveles.index:
                                ax.bar(
                                    nivel.capitalize(), 
                                    dist_niveles[nivel],
                                    color=colores.get(nivel, 'blue')
                                )
                            
                            ax.set_ylabel('Cantidad de Documentos')
                            ax.set_title('Distribución de Niveles de Similitud')
                            plt.tight_layout()
                            
                            st.pyplot(fig)
                            
                            # Mostrar pie chart de porcentajes
                            fig, ax = plt.subplots(figsize=(8, 8))
                            ax.pie(
                                dist_niveles, 
                                labels=[n.capitalize() for n in dist_niveles.index], 
                                autopct='%1.1f%%',
                                colors=[colores.get(n, 'blue') for n in dist_niveles.index],
                                startangle=90
                            )
                            ax.axis('equal')
                            ax.set_title('Porcentaje por Nivel de Similitud')
                            
                            st.pyplot(fig)

# Pie de página
st.markdown("""
---
<div style="text-align: center; color: #666666;">
    <p>Sistema de Detección de Infracciones de Derechos de Autor © 2023</p>
    <p style="font-size: 0.8em;">Desarrollado con Streamlit y TensorFlow</p>
</div>
""", unsafe_allow_html=True)