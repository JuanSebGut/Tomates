from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import base64
import os
from ultralytics import YOLO
import cv2

app = Flask(__name__)
CORS(app)

# MongoDB Atlas Connection Setup
# Establishes connection to cloud database using connection string with credentials

from pymongo import MongoClient

client = MongoClient(
    "mongodb+srv://juansegutt:300716@cluster0.erc6l2t.mongodb.net/tomates_db")
db = client["IdentificadorTomates"]

# Colecciones
predictions_collection = db["predictions"]
segmentations_collection = db["segmentations"]

como pongo estas lineas, es para base de datos local 

# ---------------------------------------------------------------------------------------------------------
# 2. Model Configuration / Configuración de Modelos
# ---------------------------------------------------------------------------------------------------------

# Path to trained models directory / Ruta al directorio de modelos entrenados
MODEL_PATH = "models/"

# Model configuration (will be updated when loading models)
# Configuración de modelos (se actualizará al cargar los modelos)
MODEL_CONFIG = {
    'efficientnetb3': {
        'img_size': (224, 224),
        'preprocess': tf.keras.applications.efficientnet.preprocess_input
    },
    'resnet50': {
        'img_size': (224, 224),# Input image size in pixels (width, height) / Tamaño de imagen de entrada en píxeles (ancho, alto)
        'preprocess': tf.keras.applications.resnet50.preprocess_input# Preprocessing function / Función de preprocesamiento
    },
    'inceptionv3': {
        'img_size': (224, 224),
        'preprocess': tf.keras.applications.inception_v3.preprocess_input
    }
}

# Class dictionary for tomato states / Diccionario de clases para estados del tomate
CLASSES = ['damaged', 'old', 'ripe', 'unripe']

# --------------------------------------------------------------------------------
# 3. Load Classification Models / Cargar Modelos de Clasificación
# --------------------------------------------------------------------------------
models = {} # Dictionary to store loaded models / Diccionario para almacenar modelos cargados

def get_model_input_shape(model):
      #Gets the model input size automatically / Obtiene el tamaño de entrada del modelo automáticamente
    return model.input_shape[1:3] # Returns (height, width) / Retorna (alto, ancho)

def load_classification_models():
    # Loads classification models when the app starts / Carga los modelos de clasificación al iniciar la app
    try:
        models['efficientnetb3'] = keras.models.load_model(
            os.path.join(MODEL_PATH, 'efficientnetb3_model.h5')
        )
        # Dynamically get actual input shape from loaded model / Obtiene dinámicamente el shape de entrada del modelo cargado
        input_shape = get_model_input_shape(models['efficientnetb3'])
        # Update configuration with real model input size / Actualiza configuración con tamaño real del modelo
        MODEL_CONFIG['efficientnetb3']['img_size'] = input_shape
        print(f"✓ EfficientNetB3 cargado - Input shape: {input_shape}")
    except Exception as e:
        print(f"✗ Error cargando EfficientNetB3: {e}")
    
    try:
        # Load ResNet50 model / Carga modelo ResNet50
        models['resnet50'] = keras.models.load_model(
            os.path.join(MODEL_PATH, 'resnet50_model.h5')
        )
        input_shape = get_model_input_shape(models['resnet50'])
        MODEL_CONFIG['resnet50']['img_size'] = input_shape
        print(f"✓ ResNet50 cargado - Input shape: {input_shape}")
    except Exception as e:
        print(f"✗ Error cargando ResNet50: {e}")
    
    try:
        # Load InceptionV3 model / Carga modelo InceptionV3
        models['inceptionv3'] = keras.models.load_model(
            os.path.join(MODEL_PATH, 'inceptionv3_model.h5')
        )
        input_shape = get_model_input_shape(models['inceptionv3'])
        MODEL_CONFIG['inceptionv3']['img_size'] = input_shape
        print(f"✓ InceptionV3 cargado - Input shape: {input_shape}")
    except Exception as e:
        print(f"✗ Error cargando InceptionV3: {e}")

# --------------------------------------------------------------------------------
# 4. Load YOLO11 Model for Segmentation / Cargar Modelo YOLO11 para Segmentación
# ---------------------------------------------------------------------------------
yolo_model = None # Global variable for YOLO model / Variable global para modelo YOLO

def load_yolo_model():
    #Loads YOLO11 model in .pt format / Carga el modelo YOLO11 en formato .pt
    global yolo_model
    try:
        yolo_model = YOLO(os.path.join(MODEL_PATH, 'yolo11_segment.pt'))
        print("✓ YOLO11 cargado")
    except Exception as e:
        print(f"✗ Error cargando YOLO11: {e}")

# ---------------------------------------------------------
# 5.  Improved Auxiliary Functions / Funciones auxiliares 
# ---------------------------------------------------------
#These functions handle preparing and enhancing tomato images before the AI
#models analyze them. They process images to highlight important features like
#colors and textures, convert images to formats the models can understand, and
#execute predictions returning results with confidence levels.
#
def analyze_color_profile(img_array):
    """
    Analiza el perfil de color para detectar características de tomates old vs ripe
    Retorna métricas útiles para ajustar el preprocesamiento
    """
  # Convert to HSV for hue, saturation, value analysis 
  # Convertir a HSV para analizar tono, saturación y valor
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    # Calculate statistics/Calcular estadísticas
    avg_hue = np.mean(h)
    avg_saturation = np.mean(s)
    avg_value = np.mean(v)
    
    # Detect dark tones (old) vs bright tones (ripe) / Detectar si tiene tonos oscuros (old) o brillantes (ripe)
    dark_pixels_ratio = np.sum(v < 100) / v.size
    
    return {
        'avg_hue': avg_hue,
        'avg_saturation': avg_saturation,
        'avg_value': avg_value,
        'dark_ratio': dark_pixels_ratio
    }

def enhance_image_for_classification(img, model_name):
    #Enhances image/ mejora la imagen

    img_array = np.array(img)
    
    # Analizar perfil de color primero
    color_profile = analyze_color_profile(img_array)
    
    if model_name == 'efficientnetb3':
        # adjustments for EfficientNet
        from PIL import ImageEnhance
        
        # Contraste / contrast 
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.15)
        # saturation
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.08)
        
        # Sharpness / Nitidez
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.2)
    #And if it's resnet50
    elif model_name == 'resnet50':
        # PRESERVE DARK TONES for old tomatoes / PRESERVAR TONOS OSCUROS para tomates old
        img_array = np.array(img)
        
        # Convertir a LAB
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # SMOOTHER CLAHE to avoid artificially lighting old tomatoes
        # CLAHE MÁS SUAVE para no iluminar artificialmente tomates old
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        img = Image.fromarray(img_array)
        
        # Minimal contrast and saturation adjustments / Ajustes mínimos de contraste y saturación
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.08)

        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.02)
    
    #And if it's resnet50
    elif model_name == 'inceptionv3':
     
        img_array = np.array(img)
        
        # reduction to avoid losing texture / Reducción de ruido MUY suave para no perder textura
        img_array = cv2.bilateralFilter(img_array, 7, 50, 50)
        
        # CONSERVATIVE white balance / Balance de blancos CONSERVADOR
        result = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 0.8)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 0.8)
        img_array = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
        
        img = Image.fromarray(img_array.astype(np.uint8))
        
        # soft contrast / Contraste suave
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)
    
    return img

def preprocess_image(image_bytes, model_name):
    #Preprocesses image according to specific model/Preprocesa la imagen según el modelo específico
    config = MODEL_CONFIG[model_name]
    target_size = config['img_size']
    preprocess_fn = config['preprocess']
    
    # Load original image / Cargar imagen original
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Apply model-specific enhancements / Aplicar mejoras específicas del modelo
    img_enhanced = enhance_image_for_classification(img, model_name)
    
    # Resize with high-quality interpolation / Redimensionar con interpolación de alta calidad
    img_resized = img_enhanced.resize(target_size, Image.LANCZOS)
    
    # Convert to array and preprocess / Convertir a array y preprocesar
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_fn(img_array)
    
    return img_array, img

def image_to_base64(image):
    # Converts PIL Image to base64 / Convierte PIL Image a base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def predict_with_model(model, img_array, model_name):
    # Makes prediction with a model / Realiza predicción con un modelo
    import time
    start_time = time.time()
    predictions = model.predict(img_array, verbose=0)
    inference_time = time.time() - start_time
    
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    predicted_class = CLASSES[predicted_class_idx]
    
    # Información adicional para debugging
    all_probs = {CLASSES[i]: float(predictions[0][i]) for i in range(len(CLASSES))}
    
    # Additional information for debugging / Detectar casos ambiguos entre ripe y old
    ripe_prob = all_probs.get('ripe', 0)
    old_prob = all_probs.get('old', 0)
    diff_ripe_old = abs(ripe_prob - old_prob)
    
    return {
        'model': model_name,
        'prediction': predicted_class,
        'confidence': confidence,
        'inference_time': round(inference_time, 3),
        'all_probabilities': all_probs,
        'ambiguity_warning': diff_ripe_old < 0.15  
    }

# ---------------------------------------------------------
# 6. ENDPOINTS
# ---------------------------------------------------------

@app.route('/api/health', methods=['GET'])
def health_check():
    """Verifica que el servidor está funcionando"""
    return jsonify({
        'status': 'ok',
        'models_loaded': list(models.keys()),
        'yolo_loaded': yolo_model is not None,
        'model_shapes': {name: MODEL_CONFIG[name]['img_size'] for name in models.keys()}
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    """Lista los modelos disponibles"""
    available_models = [
        {'name': 'efficientnetb3', 'display_name': 'EfficientNet-B3', 'available': 'efficientnetb3' in models},
        {'name': 'resnet50', 'display_name': 'ResNet-50', 'available': 'resnet50' in models},
        {'name': 'inceptionv3', 'display_name': 'Inception-V3', 'available': 'inceptionv3' in models}
    ]
    return jsonify(available_models)

@app.route('/api/classify', methods=['POST'])
def classify_image():
    """Clasifica una imagen con el modelo seleccionado o todos"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No se envió ninguna imagen'}), 400
        
        image_file = request.files['image']
        selected_model = request.form.get('model', 'all')
        
        # Leer imagen una sola vez
        image_bytes = image_file.read()
        
        results = {}
        original_img = None
        
        if selected_model == 'all':
            for model_name, model in models.items():
                img_array, img = preprocess_image(image_bytes, model_name)
                if original_img is None:
                    original_img = img
                results[model_name] = predict_with_model(model, img_array, model_name)
        else:
            if selected_model not in models:
                return jsonify({'error': f'Modelo {selected_model} no disponible'}), 400
            
            img_array, original_img = preprocess_image(image_bytes, selected_model)
            results[selected_model] = predict_with_model(models[selected_model], img_array, selected_model)
        
        response_data = {
            'timestamp': datetime.now().isoformat(),
            'image_name': image_file.filename,
            'selected_model': selected_model,
            'results': results
        }
        
        prediction_doc = {
            'timestamp': datetime.now(),
            'image_name': image_file.filename,
            'image_base64': image_to_base64(original_img),
            'selected_model': selected_model,
            'results': results
        }
        inserted = predictions_collection.insert_one(prediction_doc)
        response_data['prediction_id'] = str(inserted.inserted_id)
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"❌ ERROR EN CLASIFICACIÓN: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/segment', methods=['POST'])
def segment_image():
    """Segmenta una imagen usando YOLO11"""
    try:
        if yolo_model is None:
            return jsonify({'error': 'Modelo YOLO no cargado'}), 500
        
        if 'image' not in request.files:
            return jsonify({'error': 'No se envió ninguna imagen'}), 400
        
        image_file = request.files['image']
        
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        import time
        start_time = time.time()
        results = yolo_model(img)
        inference_time = time.time() - start_time
        
        result = results[0]
        annotated_img = result.plot()
        
        _, buffer = cv2.imencode('.png', annotated_img)
        segmented_base64 = base64.b64encode(buffer).decode()
        
        _, buffer_original = cv2.imencode('.png', img)
        original_base64 = base64.b64encode(buffer_original).decode()
        
        num_objects = len(result.boxes) if result.boxes is not None else 0
        
        response_data = {
            'timestamp': datetime.now().isoformat(),
            'image_name': image_file.filename,
            'original_image': f'data:image/png;base64,{original_base64}',
            'segmented_image': f'data:image/png;base64,{segmented_base64}',
            'num_objects': num_objects,
            'inference_time': round(inference_time, 3)
        }
        
        segmentation_doc = {
            'timestamp': datetime.now(),
            'image_name': image_file.filename,
            'original_image': original_base64,
            'segmented_image': segmented_base64,
            'num_objects': num_objects,
            'inference_time': round(inference_time, 3)
        }
        inserted = segmentations_collection.insert_one(segmentation_doc)
        response_data['segmentation_id'] = str(inserted.inserted_id)
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"ERROR EN SEGMENTACIÓN: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """Obtiene el historial de predicciones"""
    try:
        limit = int(request.args.get('limit', 50))
        predictions = list(predictions_collection.find(
            {}, 
            {'image_base64': 0}
        ).sort('timestamp', -1).limit(limit))
        
        for pred in predictions:
            pred['_id'] = str(pred['_id'])
            pred['timestamp'] = pred['timestamp'].isoformat()
        
        return jsonify(predictions), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/segmentations', methods=['GET'])
def get_segmentations():
    """Obtiene el historial de segmentaciones"""
    try:
        limit = int(request.args.get('limit', 50))
        segmentations = list(segmentations_collection.find(
            {},
            {'original_image': 0, 'segmented_image': 0}
        ).sort('timestamp', -1).limit(limit))
        
        for seg in segmentations:
            seg['_id'] = str(seg['_id'])
            seg['timestamp'] = seg['timestamp'].isoformat()
        
        return jsonify(segmentations), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Obtiene estadísticas generales"""
    try:
        total_predictions = predictions_collection.count_documents({})
        total_segmentations = segmentations_collection.count_documents({})
        
        pipeline = [
            {'$unwind': '$results'},
            {'$group': {
                '_id': '$results.prediction',
                'count': {'$sum': 1}
            }},
            {'$sort': {'count': -1}},
            {'$limit': 1}
        ]
        
        most_common = list(predictions_collection.aggregate(pipeline))
        
        stats = {
            'total_predictions': total_predictions,
            'total_segmentations': total_segmentations,
            'most_common_class': most_common[0]['_id'] if most_common else None
        }
        
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---------------------------------------------------------
# 7. Inicialización
# ---------------------------------------------------------
if __name__ == '__main__':
    print("Cargando modelos...")
    load_classification_models()
    load_yolo_model()
    print("\n🍅 Servidor de Identificación de Tomates iniciado")
    print("   - Preprocesamiento optimizado para 4 clases")
    print("   - Mejoras conservadoras de imagen")
    print("   - Detección de ambigüedad ripe/old")
    print("\nServidor en http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)