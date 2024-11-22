import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import time
from tensorflow.keras.saving import register_keras_serializable
import os

# Выводим статус доступности GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
st.write("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# Получаем абсолютный путь к модели, начиная от текущего скрипта
model_path = os.path.join(os.path.dirname(__file__), '../models/deeplabv3.keras')
#model_path = 'models/deeplabv3.keras'

@register_keras_serializable()
def iou_coef(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=[1, 2, 3])  # intersection
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection  # union
    iou = tf.reduce_mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

# Функция для загрузки модели
def load_model():
    try:
        model = tf.keras.models.load_model(model_path)  # Укажите путь к модели
        st.success("Модель успешно загружена.")
        return model
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        return None

# Загружаем модель
model = load_model()

if model:
    st.write("Модель готова к использованию.")
else:
    st.write("Не удалось загрузить модель. Пожалуйста, проверьте путь и совместимость.")

# Функция для предобработки изображения
def preprocess_image(img, target_size=(256, 256)):
    if isinstance(img, Image.Image):  # Проверка, что img - это объект PIL
        img = img.resize(target_size)  # Изменение размера
    img_array = image.img_to_array(img)  # Преобразуем в массив
    img_array = np.expand_dims(img_array, axis=0)  # Добавляем размерность для батча
    img_array = img_array / 255.0  # Нормализация изображения
    return img_array

# Функция для загрузки изображения по URL
def load_image_from_url(url, target_size=(256, 256)):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize(target_size)  # Изменяем размер для соответствия модели
    return img

# Функция для предсказания сегментации
def predict_segmentation(img_array):
    prediction = model.predict(img_array)
    predicted_mask = prediction[0, :, :, 0]  # Предполагаем, что модель возвращает маску для одного канала
    return predicted_mask

# Интерфейс Streamlit
st.title("Сегментация изображений с использованием обученной модели DeeplabV3")

# Статистика модели
st.header("Статистика модели")
with st.expander("Показать статистику модели", expanded=False):
    st.write("F1 Score:")
    st.image("images/deeplabv3/F1_comparison.png", caption="График F1", use_container_width=True)
    st.write("Accuracy:")
    st.image("images/deeplabv3/accuracy_comparison.png", caption="График Accuracy", use_container_width=True)
    st.write("Loss:")
    st.image("images/deeplabv3/loss_comparison.png", caption="График Loss", use_container_width=True)
    st.write("IoU:")
    st.image("images/deeplabv3/iou_coef_comparison.png", caption="График IoU", use_container_width=True)
    st.write("Precision:")
    st.image("images/deeplabv3/precision_comparison.png", caption="График Precision", use_container_width=True)
    st.write("Recall:")
    st.image("images/deeplabv3/recall_comparison.png", caption="График Recall", use_container_width=True)

st.header("Инференс модели")
image_option = st.radio(
    "Выберите способ загрузки изображения:",
    ("Загрузить изображения из файлов", "Загрузить изображение по URL")
)

images = []  # Список для хранения изображений

if image_option == "Загрузить изображения из файлов":
    uploaded_files = st.file_uploader("Выберите изображения...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        images = [Image.open(uploaded_file) for uploaded_file in uploaded_files]
        for i, img in enumerate(images):
            st.image(img, caption=f"Загруженное изображение {i+1}", use_container_width=True)

elif image_option == "Загрузить изображение по URL":
    url = st.text_input("Введите URL изображения:")
    if url:
        try:
            img = load_image_from_url(url)  # Загружаем изображение по URL
            images = [img]
            st.image(img, caption="Изображение из URL", use_container_width=True)
        except Exception as e:
            st.error(f"Не удалось загрузить изображение. Ошибка: {e}")

# Кнопка предсказания
if st.button("Предсказать"):
    if images:
        start_time = time.time()  # Засекаем время начала предсказания
        
        for i, img in enumerate(images):
            img_array = preprocess_image(img)
            predicted_mask = predict_segmentation(img_array)
            
            # Преобразуем предсказанную маску в размер оригинального изображения
            predicted_mask_resized = Image.fromarray((predicted_mask * 255).astype(np.uint8))
            predicted_mask_resized = predicted_mask_resized.resize(img.size)
            
            # Выводим предсказанную маску
            st.image(predicted_mask_resized, caption=f"Предсказанная маска {i+1}", use_container_width=True)
        
        elapsed_time = time.time() - start_time
        st.write(f"Время ответа модели: {elapsed_time:.2f} секунд")
    else:
        st.warning("Пожалуйста, загрузите изображения.")
