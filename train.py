import os
from ultralytics import YOLO
from roboflow import Roboflow

# --- 1. Налаштування Roboflow ---
# !!! ВСТАВТЕ СЮДИ СВОЇ ДАНІ З ROBOFLOW !!!
# (Отримайте їх з Roboflow після експорту набору даних)
ROBOFLOW_API_KEY = "UyWmPM10Ijo2MRZOcE6Z"
ROBOFLOW_WORKSPACE = "yolo-0hsar"
ROBOFLOW_PROJECT = "fruit-detector-esfls"
ROBOFLOW_VERSION = 1 # Зазвичай 1, якщо це перша версія

print("Connecting to Roboflow and downloading dataset...")
try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    dataset = project.version(ROBOFLOW_VERSION).download("yolov8")
    DATA_YAML_PATH = os.path.join(dataset.location, "data.yaml")
    print(f"Dataset downloaded successfully to: {dataset.location}")
except Exception as e:
    print(f"Error downloading dataset: {e}")
    print("Please check your API key and project settings in Roboflow.")
    exit()

print("Loading pre-trained YOLOv11n model...")
model = YOLO('yolo11n.pt') 

print("Starting model training (fine-tuning)...")
try:
    results = model.train(
        data=DATA_YAML_PATH,  # Шлях до data.yaml з Roboflow
        epochs=50,            # Кількість епох (50-100 - гарний старт)
        imgsz=640,            # Розмір зображення
        name='yolo11n_fruit_detector'
    )
    
    print("Training finished successfully.")
    print(f"Your trained model is saved in: runs/detect/yolo11n_fruit_detector/weights/best.pt")

except Exception as e:
    print(f"An error occurred during training: {e}")