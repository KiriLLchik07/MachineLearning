import streamlit as st

st.set_page_config(
    page_title="Добро пожаловать!",
    page_icon="👋",
    layout="wide"
)

st.markdown("""
<style>
    .justified-text {
        text-align: justify;
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: green;'>Привет!</h1>", unsafe_allow_html=True)

st.markdown("""
<div class="justified-text">
    Это мой первый дашборд с использованием фреймворка Streamlit, который будет посвящен изучению инференса моделей машинного обучения. 
    Здесь я покажу, как работают различные алгоритмы и как можно визуализировать их результаты.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
<div class="justified-text">
Дашборд состоит из четырех основных страниц:
            
1) Информация о разработчике  
2) Описание датасета  
3) Визуализация данных датасета  
4) Инференс моделей машинного обучения
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
<div class="justified-text">
⏎| Для выбора нужной страницы можете воспользоваться навигацией слева на экране
</div>
""", unsafe_allow_html=True)

st.markdown("---")

photo_path = "C:/Users/kiril/OneDrive/Рабочий стол/Учёба/2 курс/2 семестр/ML/РГР/data/letsGo.jpeg"
st.image(photo_path, use_container_width=True)