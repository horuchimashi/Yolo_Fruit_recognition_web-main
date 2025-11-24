import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory
from ultralytics import YOLO
import uuid # Для генерації унікальних імен файлів

app = Flask(__name__)

# --- ВАЖЛИВО: Шлях до вашої натренованої моделі ---
# Цей шлях має відповідати тому, що ви бачили в консолі після train.py
# Зазвичай це 'runs/detect/yolo11n_fruit_detector/weights/best.pt'
MODEL_PATH = os.path.join('runs', 'detect', 'yolo11n_fruit_detector2', 'weights', 'best.pt')

# Директорія для завантажених та оброблених файлів
STATIC_DIR = 'static'
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Завантажуємо модель
try:
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Make sure the model file exists at: {MODEL_PATH}")
    model = None


@app.route('/', methods=['GET', 'POST'])
def predict():
    if model is None:
        return "Model not loaded. Please check server console.", 500

    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            try:
                img_bytes = file.read()
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    return jsonify({'error': 'Could not decode image'}), 400

                results = model(img)

                img_with_boxes = results[0].plot()

                filename = f"{uuid.uuid4()}.jpg"
                save_path = os.path.join(STATIC_DIR, filename)
                
                cv2.imwrite(save_path, img_with_boxes)

                return jsonify({'result_image_url': f'/{save_path}'})

            except Exception as e:
                print(f"Error during prediction: {e}")
                return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def send_file(filename):
    return send_from_directory(STATIC_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True)