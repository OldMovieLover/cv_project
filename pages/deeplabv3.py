import streamlit as st
import tensorflow as tf
import os
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import time

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#tf.config.list_physical_devices('GPU')

# Выводим статус доступности GPU
st.write("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# Включение/отключение GPU через боковую панель
use_gpu = st.checkbox("Использовать GPU", value=True)

# Если GPU выключен, скрываем его от TensorFlow
if not use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    st.warning("Использование GPU отключено. Используется только CPU.")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Предполагаем, что у вас только один GPU

@register_keras_serializable()
def iou_coef(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=[1, 2, 3])  # intersection
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection  # union
    iou = tf.reduce_mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

# Функция для загрузки модели с учетом использования GPU/CPU
def load_model(use_gpu):
    try:
        if not use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Отключаем GPU
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Включаем GPU, если оно доступно

        # Загрузка модели
        model = tf.keras.models.load_model('models/deeplabv3.keras')  # Укажите путь к модели
        st.success("Модель успешно загружена.")
        return model
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        return None

# Кнопка для загрузки модели
load_button = st.button("Загрузить модель")

if load_button:
    model = load_model(use_gpu)  # Модель будет загружена только после нажатия кнопки
else:
    model = None
    st.write("Модель не загружена. Нажмите кнопку, чтобы загрузить модель.")


if model:
    st.write("Модель готова к использованию.")
else:
    st.write("Не удалось загрузить модель. Пожалуйста, проверьте путь и совместимость.")

# Функция для предобработки изображения
def preprocess_image(img_path, target_size=(256, 256)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Добавляем размерность для батча
    img_array = img_array / 255.0  # Нормализация изображения
    return img_array

# Функция для загрузки изображения по URL
def load_image_from_url(url, target_size=(256, 256)):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize(target_size)
    img_array = np.array(img)
    return img_array

# Функция для предсказания сегментации
def predict_segmentation(img_array):
    prediction = model.predict(img_array)
    predicted_mask = prediction[0, :, :, 0]  # Предполагаем, что модель возвращает маску для одного канала
    return predicted_mask

# Функция для наложения полупрозрачной маски на изображение
def overlay_mask_on_image(img, mask):
    mask = np.expand_dims(mask, axis=-1)  # Добавляем размерность
    mask = np.repeat(mask, 3, axis=-1)  # Делаем маску трехканальной
    overlay = np.where(mask > 0.5, mask * 0.5 + img * 0.5, img)  # Наложение маски
    return overlay

# Интерфейс Streamlit
st.title("Сегментация изображений с использованием обученной модели DeeplabV3")

# Статистика
st.header("Статистика модели")
with st.expander("Показать статистику модели", expanded=False):
    st.write("F1 Score:")
    st.image("images/deeplabv3/F1_comparison.png", caption="График F1", use_container_width=True)
    st.write("PR curve:")
    st.image("images/deeplabv3/precision_comparison.png", caption="График PR curve", use_container_width=True)
    st.write("Precision")
    st.image("images/deeplabv3/precision_comparison.png", caption="График Precision", use_container_width=True)
    st.write("Recall:")
    st.image("images/deeplabv3/recall_comparison.png", caption="График Recall", use_container_width=True)

st.header("Инференс модели")
image_option = st.radio(
    "Выберите способ загрузки изображения:",
    ("Загрузить изображения из файлов", "Загрузить изображение по URL")
)

if image_option == "Загрузить изображения из файлов":
    uploaded_files = st.file_uploader("Выберите изображения...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        images = [Image.open(uploaded_file) for uploaded_file in uploaded_files]
        for i, img in enumerate(images):
            st.image(img, caption=f"Загруженное изображение {i+1}", use_container_width=True)

elif image_option == "Загрузить изображение по URL":
    url = st.text_input("Введите URL изображения:")
    if url:
        uploaded_files = None
        try:
            img_array = load_image_from_url(url)
            st.image(img_array, caption="Изображение из URL", use_container_width=True)
        except Exception as e:
            st.error(f"Не удалось загрузить изображение. Ошибка: {e}")

# Кнопка предсказания
if st.button("Предсказать"):
    if uploaded_files:
        start_time = time.time()
        for i, img in enumerate(images):
            img_array = preprocess_image(img)
            predicted_mask = predict_segmentation(img_array)
            overlayed_img = overlay_mask_on_image(np.array(img), predicted_mask)
            st.image(overlayed_img, caption=f"Изображение с маской {i+1}", use_container_width=True)
            st.image(predicted_mask, caption=f"Предсказанная маска {i+1}", use_container_widthh=True, clamp=True)
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.write(f"Время ответа модели: {elapsed_time:.2f} секунд")
    else:
        st.warning("Пожалуйста, загрузите изображения.")