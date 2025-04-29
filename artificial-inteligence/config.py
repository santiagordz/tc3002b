import os
import time

# Rutas a directorios principales
TRAIN_DIR = "./dataset/train"
TEST_DIR = "./dataset/test"
VALID_DIR = "./dataset/valid"
MODEL_DIR = "models"
LOGS_DIR = "logs"
RESULTS_DIR = "results"
BACKUP_DIR = os.path.join(RESULTS_DIR, "backups")

# Crear directorios si no existen
for dir_path in [MODEL_DIR, LOGS_DIR, RESULTS_DIR, BACKUP_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Hiperparámetros de preprocesamiento
IMG_SIZE = 224  # Dimensión espacial para entrada de la red
BATCH_SIZE = 32  # Tamaño de mini-batch para SGD

# Hiperparámetros de data augmentation
ROTATION_RANGE = 20  # Grados de rotación aleatoria
WIDTH_SHIFT_RANGE = 0.2  # Desplazamiento horizontal (fracción del ancho total)
HEIGHT_SHIFT_RANGE = 0.2  # Desplazamiento vertical (fracción del alto total)
HORIZONTAL_FLIP = True  # Volteo horizontal aleatorio
VALIDATION_SPLIT = 0.2  # Fracción para validación interna

# Hiperparámetros del modelo
LEARNING_RATE = 1e-4  # Tasa de aprendizaje inicial
FINI_TUNING_LR = 1e-5  # Tasa de aprendizaje inicial
DROPOUT_RATE = 0.3  # Tasa de dropout para regularización

# Hiperparámetros de entrenamiento
EPOCHS = 10  # Número de épocas máximas
PATIENCE = 5  # Épocas para early stopping

# Nombre del modelo basado en timestamp
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
MODEL_NAME = f"cnn_detector_{TIMESTAMP}"

# Rutas a archivos del modelo
MODEL_PATH = os.path.join(MODEL_DIR, "modelo_detector.keras")
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, f'{MODEL_NAME}_best.keras')

# Clases para clasificación
CLASS_NAMES = ["clubs", "diamonds", "hearts", "spades"]