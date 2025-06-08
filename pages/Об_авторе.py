import streamlit as st
import os

st.set_page_config(
    page_title="Информация о разработчике",
    page_icon="👨‍💻",
    layout="wide"
)

st.title("Информация о разработчике")

photo_path = "C:/Users/kiril/OneDrive/Рабочий стол/Учёба/2 курс/2 семестр/ML/РГР/data/rg.jpeg"
st.image(photo_path, caption="Фотография разработчика", width=500)

st.write("**ФИО:** Есаков Кирилл Николаевич")
st.write("**Номер учебной группы:** ФИТ-232")
st.write("**Тема РГР:** Разработка Web-приложения (дашборда) для инференса моделей ML и анализа данных")
