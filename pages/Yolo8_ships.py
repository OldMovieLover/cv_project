import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
import io
import time
import matplotlib.pyplot as plt

# Загрузка предобученной модели YOLO
model = YOLO("models/Yolo8_ships.pt")  # Укажите путь к предобученной модели

# Путь к директории со статистикой
stats_dir = "images/Yolov8_ships"  # Укажите путь к вашей папке

# Загрузка статистики на изображениях
def load_statistics_images(directory):
    if os.path.exists(directory):
        image_files = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            st.subheader("Статистика обучения")
            for image_file in image_files:
                image_path = os.path.join(directory, image_file)
                img = Image.open(image_path)
                st.image(img, caption=image_file, use_column_width=True)
        else:
            st.write("В указанной директории нет изображений.")
    else:
        st.write("Директория со статистикой не найдена. Проверьте путь.")

# Интерфейс Streamlit
st.title("Обнаружение объектов с помощью YOLOv8")
st.write("Загрузите изображение или введите URL для анализа.")

# Загрузка и отображение статистики
load_statistics_images(stats_dir)

# Функция для загрузки изображения из URL
def load_image_from_url(url: str):
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content))
    return img

# Выбор режима: загрузить одно или несколько изображений
upload_mode = st.radio("Выберите режим загрузки:", ("Одно изображение", "Множественные изображения"))

# Массив для хранения изображений и результатов
images = []
results_list = []

if upload_mode == "Одно изображение":
    # Загрузка одного изображения
    uploaded_file = st.file_uploader("Выберите файл", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        images.append(image)
        st.image(image, caption="Загруженное изображение", use_column_width=True)
else:
    # Загрузка нескольких изображений
    uploaded_files = st.file_uploader("Выберите файлы", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            images.append(image)
            st.image(image, caption=f"Загруженное изображение {uploaded_file.name}", use_column_width=True)

# Ввод ссылки на изображение
url_input = st.text_input("Или введите ссылку на изображение")

if url_input:
    try:
        image = load_image_from_url(url_input)
        images.append(image)
        st.image(image, caption="Изображение с URL", use_column_width=True)
    except Exception as e:
        st.error(f"Не удалось загрузить изображение из URL: {e}")

# Если изображения загружены, применяем YOLO
if images:
    for i, image in enumerate(images):
        # Визуализация времени обработки
        start_time = time.time()

        # Предсказание YOLO
        results = model(image)

        end_time = time.time()
        response_time = end_time - start_time

        # Сохранение изображения с аннотациями
        annotated_image = results[0].plot()  # Получение изображения с предсказанными аннотациями

        # Вывод аннотированного изображения
        st.image(annotated_image, caption=f"Результат предсказания для изображения {i+1}", use_column_width=True)

        # Визуализация времени ответа
        st.write(f"Время классификации для изображения {i+1}: {response_time:.4f} секунд")
else:
    st.write("Пожалуйста, загрузите изображение для анализа.")
