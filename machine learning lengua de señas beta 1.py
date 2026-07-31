"""
Machine Learning Leangua de Senhas
Sistema de Machine Learning entrenable
"""

import cv2
import numpy as np
from collections import deque
import os
import pickle

# Importar MediaPipe Tasks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

# Importar TensorFlow/Keras para la IA
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks

# Modelo
MODEL_FILENAME = 'hand_landmarker.task'
DATA_FILE = 'sign_data.pkl'
MODEL_FILE = 'sign_model.keras'
WATERMARK = '@alejondre._ en instagram'


# CONFIGURACIÓN DE LA MACHINE LEARNING

# Número de landmarks de la mano (21 puntos)
NUM_LANDMARKS = 21
# Características por landmark (x, y, z)
NUM_FEATURES = 3
# Input total: 21 landmarks * 3 coordenadas = 63 características
INPUT_SIZE = NUM_LANDMARKS * NUM_FEATURES

# Alfabeto del Lenguaje de Señas Boliviano (LSB)
LABELS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
NUM_CLASSES = len(LABELS)

# ============================================
# FUNCIONES DE DATOS
# ============================================

def download_model():
    """Descarga el modelo de detección de manos"""
    model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    
    if not os.path.exists(MODEL_FILENAME):
        print("Descargando modelo de detección de manos...")
        try:
            import urllib.request
            urllib.request.urlretrieve(model_url, MODEL_FILENAME)
            print("Modelo descargado correctamente")
            return True
        except Exception as e:
            print(f"Error al descargar modelo: {e}")
            return False
    return True

def extract_features(hand_landmarks):
    """Extrae características de los landmarks de la mano, normalizándolas"""
    features = []
    
    # Tomar la muñeca como punto de referencia (landmark 0)
    base_x = hand_landmarks[0].x
    base_y = hand_landmarks[0].y
    base_z = hand_landmarks[0].z
    
    for landmark in hand_landmarks:
        # Calcular coordenadas relativas a la muñeca
        rel_x = landmark.x - base_x
        rel_y = landmark.y - base_y
        rel_z = landmark.z - base_z
        features.extend([rel_x, rel_y, rel_z])
        
    # Convertir a array de numpy
    features = np.array(features)
    
    # Normalizar por el valor máximo absoluto para ser invariante al tamaño (distancia a la cámara)
    max_val = np.max(np.abs(features))
    if max_val > 0:
        features = features / max_val
        
    return features

def load_data():
    """Carga los datos de entrenamiento"""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'rb') as f:
            data = pickle.load(f)
        return data['X'], data['y'], data['labels']
    return None, None, None

def save_data(X, y, labels):
    """Guarda los datos de entrenamiento"""
    with open(DATA_FILE, 'wb') as f:
        pickle.dump({'X': X, 'y': y, 'labels': labels}, f)
    print(f"Datos guardados: {len(X)} muestras")


def watermark_check():
    """Verifica que la marca de agua permanezca en el código."""
    if '@alejondre' not in globals().get('WATERMARK', ''):
        raise RuntimeError('Marca de agua removida. El programa no puede ejecutarse.')
    if WATERMARK.strip() == '':
        raise RuntimeError('Marca de agua inválida. El programa no puede ejecutarse.')


def draw_watermark(frame):
    """Dibuja la marca de agua en la imagen de la cámara."""
    h, w, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    text_size, _ = cv2.getTextSize(WATERMARK, font, scale, thickness)
    text_w, text_h = text_size
    x = w - text_w - 10
    y = text_h + 10
    cv2.putText(frame, WATERMARK, (x, y),
                font, scale, (0, 180, 255), thickness, cv2.LINE_AA)


# RED NEURONAl


def create_model():
    """Crea una red neuronal para reconocimiento de señas"""
    model = models.Sequential([
        layers.Input(shape=(INPUT_SIZE,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(X, y, epochs=50, batch_size=32):
    """Entrena el modelo de IA"""
    print("\n" + "="*60)
    print("  ENTRENANDO MODELO DE IA")
    print("="*60)
    
    model = create_model()
    model.summary()
    
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"\nDatos de entrenamiento: {len(X_train)}")
    print(f"Datos de validación: {len(X_val)}")
    
    cb = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb,
        verbose=1
    )
    
    model.save(MODEL_FILE)
    print(f"\nModelo guardado en: {MODEL_FILE}")
    
    return model, history

def load_trained_model():
    """Carga un modelo entrenado"""
    if os.path.exists(MODEL_FILE):
        return keras.models.load_model(MODEL_FILE)
    return None

# ============================================
# RECOLECCIÓN DE DATOS
# ============================================

def collect_data(detector, cap, num_samples=50, label='A'):
    """Recolecta datos de entrenamiento para una letra"""
    watermark_check()
    print(f"*** {WATERMARK} ***")
    print(f"\nRecolectando datos para la letra: {label}")
    print("Presiona 'ESPACIO' para capturar, 'ESC' para salir")
    
    X_data = []
    y_data = []
    sample_count = 0
    
    while sample_count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        cv2.putText(frame, f"Letra: {label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, f"Muestras: {sample_count}/{num_samples}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Presiona ESPACIO para capturar", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        draw_watermark(frame)
        cv2.imshow('Recolección de Datos', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:
            break
        elif key == 32:
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
            result = detector.detect(mp_image)
            
            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                features = extract_features(result.hand_landmarks[0])
                X_data.append(features)
                y_data.append(LABELS.index(label))
                sample_count += 1
                print(f"  Muestra {sample_count}: capturada")
        
        draw_watermark(frame)
        cv2.imshow('Recolección de Datos', frame)
    
    cv2.destroyAllWindows()
    return np.array(X_data), np.array(y_data)

def collect_all_data(detector, cap, samples_per_letter=50):
    """Recolecta datos para todas las letras"""
    print("\n" + "="*60)
    print("  RECOLECCIÓN DE DATOS PARA ENTRENAMIENTO")
    print("="*60)
    print("Instrucciones:")
    print("- Para cada letra, muestra la seña correspondiente")
    print("- Presiona ESPACIO para capturar cada muestra")
    print("- Presiona ESC para saltar a la siguiente letra")
    print("="*60)
    
    all_X = []
    all_y = []
    
    for label in LABELS:
        X, y = collect_data(detector, cap, samples_per_letter, label)
        if len(X) > 0:
            all_X.extend(X)
            all_y.extend(y)
    
    return np.array(all_X), np.array(all_y)

def extract_from_video(detector, video_path, label, max_samples=50):
    """Extrae datos de entrenamiento desde un archivo de video"""
    if not os.path.exists(video_path):
        print(f"Error: No se encontró el archivo {video_path}")
        return np.array([]), np.array([])
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return np.array([]), np.array([])
        
    print(f"\nExtrayendo datos de {video_path} para la letra {label}...")
    
    X_data = []
    y_data = []
    sample_count = 0
    frame_skip = 3 # Analizar 1 de cada 3 frames para evitar redundancia
    frame_count = 0
    
    while sample_count < max_samples:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
            
        display_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect(mp_image)
        
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            features = extract_features(result.hand_landmarks[0])
            X_data.append(features)
            y_data.append(LABELS.index(label))
            sample_count += 1
            print(f"  Muestra {sample_count}/{max_samples}: capturada")
            draw_hand_landmarks(display_frame, result.hand_landmarks[0])
            
        cv2.putText(display_frame, f"Extrayendo: {label} ({sample_count}/{max_samples})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Extrayendo de Video', display_frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            print("Extracción cancelada.")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print(f"Se extrajeron {sample_count} muestras exitosamente.")
    return np.array(X_data), np.array(y_data)

def extract_from_images_folder(detector, folder_path):
    """Extrae datos desde una carpeta organizada por letras (ej: dataset/A/img1.jpg)"""
    if not os.path.exists(folder_path):
        print(f"Error: No se encontró la carpeta {folder_path}")
        return np.array([]), np.array([])
        
    print(f"\nExtrayendo datos de {folder_path}...")
    X_data = []
    y_data = []
    
    for letter in os.listdir(folder_path):
        letter_upper = letter.upper()
        if letter_upper not in LABELS:
            continue
            
        letter_path = os.path.join(folder_path, letter)
        if not os.path.isdir(letter_path):
            continue
            
        print(f"Procesando carpeta de la letra: {letter_upper}")
        sample_count = 0
        
        for img_name in os.listdir(letter_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(letter_path, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                continue
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
            result = detector.detect(mp_image)
            
            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                features = extract_features(result.hand_landmarks[0])
                X_data.append(features)
                y_data.append(LABELS.index(letter_upper))
                sample_count += 1
                
        print(f"  -> {sample_count} muestras extraídas para {letter_upper}")
        
    return np.array(X_data), np.array(y_data)

# ============================================
# MODO RECONOCIMIENTO
# ============================================

def recognize_sign(model, detector, cap):
    """Modo reconocimiento en tiempo real"""
    watermark_check()
    print(f"*** {WATERMARK} ***")
    print("\n" + "="*60)
    print("  MODO RECONOCIMIENTO EN TIEMPO REAL")
    print("="*60)
    print("Controles:")
    print("- ESC: Salir")
    print("- ESPACIO: Añadir espacio")
    print("- C: Borrar todo")
    print("- RETROCESO: Borrar última letra")
    print("="*60)
    
    predictions = deque(maxlen=10)
    current_letter = "?"
    spelled_text = ""
    last_added_letter = None
    frames_stable = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect(mp_image)
        
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            hand_landmarks = result.hand_landmarks[0]
            features = extract_features(hand_landmarks)
            
            pred = model.predict(features.reshape(1, -1), verbose=0)
            letter_idx = np.argmax(pred[0])
            confidence = pred[0][letter_idx]
            letter = LABELS[letter_idx]
            
            predictions.append(letter)
            
            if len(predictions) >= 5:
                pred_list = list(predictions)
                current_letter = max(set(pred_list), key=pred_list.count)
                confidence = pred_list.count(current_letter) / len(pred_list)
                
                # Lógica de deletreo
                if current_letter != "?" and current_letter != last_added_letter and confidence >= 0.7:
                    frames_stable += 1
                    if frames_stable >= 15:  # ~15 frames para confirmar letra
                        spelled_text += current_letter
                        last_added_letter = current_letter
                        frames_stable = 0
                elif current_letter != last_added_letter:
                    frames_stable = 0
            
            draw_hand_landmarks(frame, hand_landmarks)
            cv2.putText(frame, f"Letra: {current_letter}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, f"Confianza: {confidence:.1%}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        
            if current_letter != "?" and current_letter != last_added_letter and frames_stable > 0:
                progress = min(100, int((frames_stable / 15) * 100))
                cv2.putText(frame, f"Aceptando: {progress}%", (10, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
        else:
            cv2.putText(frame, "No se detecta mano", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            current_letter = "?"
            predictions.clear()
            last_added_letter = None
            frames_stable = 0
            
        # Dibujar área de texto en la parte inferior
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, f"Texto: {spelled_text}", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "ESPACIO: espacio | C: borrar todo | BACKSPACE: borrar ult | ESC: salir", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        draw_watermark(frame)
        cv2.imshow('Reconocimiento de Lenguaje de Señas', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # ESPACIO
            spelled_text += " "
            last_added_letter = None
        elif key == ord('c') or key == ord('C'):
            spelled_text = ""
            last_added_letter = None
        elif key == 8:  # Backspace
            spelled_text = spelled_text[:-1]
            last_added_letter = None
    
    cv2.destroyAllWindows()

def draw_hand_landmarks(image, hand_landmarks):
    """Dibuja los landmarks de la mano"""
    h, w, _ = image.shape
    for idx in range(min(21, len(hand_landmarks))):
        landmark = hand_landmarks[idx]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.circle(image, (x, y), 5, (0, 0, 255), 2)

# ============================================
# MENÚ PRINCIPAL
# ============================================

def main_menu():
    """Menú principal del programa"""
    print("\n" + "="*60)
    print("  MACHINE LEARNING LEANGUA DE SENHAS")
    print("="*60)
    print("1. Recolectar datos de entrenamiento (Cámara en vivo)")
    print("2. Entrenar modelo de IA")
    print("3. Ejecutar reconocimiento en tiempo real")
    print("4. Extraer datos desde imágenes/videos locales")
    print("5. Salir")
    print("="*60)
    
    option = input("Selecciona una opción: ")
    return option

def test_camera():
    """Prueba la cámara"""
    watermark_check()
    print(f"*** {WATERMARK} ***")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return None
    ret, frame = cap.read()
    if ret:
        print(f"Cámara funcionando: {frame.shape}")
    cap.release()
    return cv2.VideoCapture(0)

def main():
    """Función principal"""
    if not download_model():
        print("Error: No se pudo descargar el modelo")
        return
    
    base_options = python.BaseOptions(model_asset_path=MODEL_FILENAME)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
    
    while True:
        option = main_menu()
        
        if option == '1':
            cap = test_camera()
            if cap is None:
                continue
            
            samples = input("Muestras por letra (default 50): ")
            try:
                samples = int(samples) if samples else 50
            except ValueError:
                samples = 50
            
            letter = input("¿Qué letra quieres grabar? (A-Z, o presiona Enter para grabar todas): ").upper()
            
            if letter in LABELS:
                X_new, y_new = collect_data(detector, cap, samples, letter)
            else:
                print("Se grabarán todas las letras.")
                X_new, y_new = collect_all_data(detector, cap, samples)
                
            if len(X_new) > 0:
                X_exist, y_exist, _ = load_data()
                if X_exist is not None:
                    # Combinar datos existentes con los nuevos
                    X = np.vstack((X_exist, X_new))
                    y = np.concatenate((y_exist, y_new))
                else:
                    X = X_new
                    y = y_new
                save_data(X, y, LABELS)
            
            cap.release()
        
        elif option == '2':
            X, y, labels = load_data()
            if X is None:
                print("No hay datos. Primero recolecta datos (opción 1)")
                continue
            
            epochs = input("Épocas (default 50): ")
            epochs = int(epochs) if epochs else 50
            
            train_model(X, y, epochs=epochs)
        
        elif option == '3':
            model = load_trained_model()
            if model is None:
                print("No hay modelo entrenado. Entrena primero (opción 2)")
                continue
            
            cap = test_camera()
            if cap is None:
                continue
            
            recognize_sign(model, detector, cap)
            cap.release()
        
        elif option == '4':
            print("\n" + "-"*40)
            print("1. Extraer de un archivo de video (para una sola letra)")
            print("2. Extraer de una carpeta de imágenes (dataset estructurado)")
            print("-"*40)
            sub_opt = input("Elige una opción: ")
            
            X_new, y_new = [], []
            
            if sub_opt == '1':
                video_path = input("Ruta del archivo de video (ej: video.mp4): ")
                # Quitar comillas si el usuario arrastró el archivo a la consola
                video_path = video_path.strip('"\'')
                letter = input("¿Qué letra representa este video? (A-Z): ").upper()
                
                if letter in LABELS:
                    samples = input("Máximo de muestras a extraer (default 50): ")
                    try:
                        samples = int(samples) if samples else 50
                    except ValueError:
                        samples = 50
                    X_new, y_new = extract_from_video(detector, video_path, letter, samples)
                else:
                    print("Letra inválida. Debe ser de la A a la Z.")
            
            elif sub_opt == '2':
                print("NOTA: La carpeta debe contener subcarpetas llamadas A, B, C... con las fotos dentro.")
                folder_path = input("Ruta de la carpeta principal: ")
                folder_path = folder_path.strip('"\'')
                X_new, y_new = extract_from_images_folder(detector, folder_path)
            
            if len(X_new) > 0:
                X_exist, y_exist, _ = load_data()
                if X_exist is not None:
                    # Combinar datos existentes con los nuevos
                    X = np.vstack((X_exist, X_new))
                    y = np.concatenate((y_exist, y_new))
                else:
                    X = X_new
                    y = y_new
                save_data(X, y, LABELS)
                
        elif option == '5':
            break
        else:
            print("Opción inválida")
    
    print("\n¡Gracias por usar el programa!")

if __name__ == "__main__":
    main()