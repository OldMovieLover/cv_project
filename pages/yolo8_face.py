import torch
import os
import matplotlib.pyplot as plt
from torchvision.io import read_image
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import requests
from io import BytesIO

# Заголовок
st.title("YOLOv8 Object Detection App with Face Blurring")
st.header("Детекция и размытие лиц на изображениях")

# Статистика
st.title("Статистика модели")

# Раскрывающийся контейнер для статистики
with st.expander("Показать статистику модели", expanded=False):
    st.write("В train выборке было: 13386 фотографий")
    st.write("В valid выборке было: 3347 фотографий")
    st.write("F1 Score: я забыл посмотреть!!!!")
    st.image("images/Yolov8_face/F1_curve.png", caption="График F1", use_container_width=True)
    st.write("PR curve:")
    st.image("images/Yolov8_face/PR_curve.png", caption="График PR curve", use_container_width=True)
    st.write("Precision")
    st.image("images/Yolov8_face/P_curve.png", caption="График Precision", use_container_width=True)
    st.write("Recall:")
    st.image("images/Yolov8_face/R_curve.png", caption="График Recall", use_container_width=True)
    st.write("Confusion Matrix:")
    st.image("images/Yolov8_face/confusion_matrix.png", caption="Матрица ошибок Модели 1", use_container_width=True)
    st.image("images/Yolov8_face/confusion_matrix_normalized.png", caption="Матрица ошибок Модели 1", use_container_width=True)
    st.write("Результаты:")
    st.image("images/Yolov8_face/results.png", caption="Результаты модели", use_container_width=True)

if st.button("Ссылки для проверки"):
    st.write("https://fotoblik.ru/wp-content/uploads/2023/09/buratino-malvtna-pero-5.webp")
    st.write("https://cdnn21.img.ria.ru/images/07e6/03/18/1779925111_0:197:3258:2030_1920x1080_80_0_0_dff6319c6d3c660563ba773e8a1353d4.jpg")

# Модель и веса
model_path = "models/Yolov8_face.pt"
model = YOLO(model_path)

# Выбор типа загрузки
upload_mode = st.radio("Выберите способ загрузки", ("Загрузить изображения", "Загрузить по ссылке"))

# Ползунок порога
confidence_threshold = st.slider(
    "Порог уверенности",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05,
)

images = []

if upload_mode == "Загрузить изображения":
    # Загрузка нескольких
    uploaded_files = st.file_uploader("Загрузите изображения", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            images.append(np.array(image))

elif upload_mode == "Загрузить по ссылке":
    # Загрузка по ссылке
    image_url = st.text_input("Введите ссылку на изображение:")
    if image_url:
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            images.append(np.array(image))
        except Exception as e:
            st.error(f"Ошибка загрузки изображения: {e}")

if images:
    blur_clicked = False

    # Кнопка для размытия лиц
    blur_button = st.button("Размыть все лица на изображениях")

    for idx, image_np in enumerate(images):
        st.subheader(f"Изображение {idx + 1}")
        st.image(image_np, caption=f"Оригинальное изображение {idx + 1}", use_container_width=True)

        # Детекция с учетом порога
        st.text(f"Обрабатываем изображение {idx + 1}...")
        results = model(image_np, conf=confidence_threshold)

        annotated_image = results[0].plot()

        # Показ детекции
        st.image(annotated_image, caption=f"Результат детекции {idx + 1} с порогом {confidence_threshold:.2f}", use_container_width=True)

        # Если кнопка для размытия была нажата, размываем все изображения
        if blur_button:
            blur_clicked = True
            # Создаем копию изображения
            blurred_image = image_np.copy()

            # Получаем координаты боксов и размываем
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0]) 
                face = blurred_image[y1:y2, x1:x2] 
                blurred_face = cv2.GaussianBlur(face, (51, 51), 30) 
                blurred_image[y1:y2, x1:x2] = blurred_face  

            # Показ результата с размытыми лицами
            st.image(blurred_image, caption=f"Изображение {idx + 1} с размытыми лицами", use_container_width=True)

    if not blur_clicked:
        st.write("Нажмите кнопку для размытия всех лиц на изображениях.")
