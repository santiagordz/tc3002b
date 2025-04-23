# Detector de Cartas

Este proyecto implementa un clasificador multiclase de imágenes utilizando Redes Neuronales Convolucionales (CNN) para distinguir entre diferentes palos de cartas. El modelo utiliza la técnica de transfer learning a través de la arquitectura ResNet50 preentrenada para maximizar el rendimiento con menor tiempo de entrenamiento.

## Estructura del Proyecto

El proyecto está organizado en dos archivos principales que conforman un flujo completo de machine learning y una aplicación web para probar el modelo resultante:

1. **main.ipynb**: Jupiter Notebook con toda la implementación con TensorFlow
2. **app.py**: Aplicación de Flask para probar el modelo

## Requisitos

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib

## Preparación del Entorno

1. Usar un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate 
```

2. Instalar dependencias:

```bash
pip install tensorflow numpy matplotlib pillow
```

## Ejecución del Pipeline

Ejecutar todo el archivo main.ipynb

## Parámetros del Modelo

- **Arquitectura**: ResNet50 preentrenada en ImageNet (transfer learning)
- **Input**: Imágenes RGB de 224x224 píxeles
- **Optimizador**: Adam con learning rate adaptativo
- **Función de pérdida**: Cross-Entropy
- **Métricas**: Accuracy, AUC-ROC, Precision, Recall, F1-Score
- **Regularización**: Dropout (0.3), Early Stopping, Reducción de learning rate

## Técnicas de Data Augmentation

Para mejorar la robustez del modelo, se aplican las siguientes transformaciones:
- Rotación aleatoria (±15°)
- Desplazamiento horizontal y vertical (10%)
- Volteo horizontal
- Normalización de píxeles al rango [0,1]

## Resultados y Visualización

Los resultados del entrenamiento se guardan en el directorio `resultados`:
- Modelo con mejores pesos (`cnn_detector_*_best.keras`)
- Modelo final (`cnn_detector_*_final.h5`)
- Curvas de aprendizaje (accuracy y loss)
- Historial de entrenamiento (formato JSON)