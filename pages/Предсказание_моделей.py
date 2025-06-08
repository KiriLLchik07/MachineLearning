import streamlit as st
import pandas as pd
import pickle
from catboost import CatBoostRegressor

st.set_page_config(
    page_title="Предсказание модели",
    page_icon="🔍"
)

st.title("Инференс моделей ML")
st.markdown("---")

@st.cache_resource  
def load_models():
    models = {
        "Дерево решений": pickle.load(open("models/decision_tree.pkl", "rb")),
        "Градиентный бустинг": pickle.load(open("models/gradient_boost.pkl", "rb")),
        "CatBoost": CatBoostRegressor().load_model("models/catboost.cbm"),
        "Случайный лес": pickle.load(open("models/random_forest.pkl", "rb")),
        "Стекинг": pickle.load(open("models/stacking.pkl", "rb")),
    }
    return models

models = load_models()

st.header("1. Загрузка CSV-файла")
uploaded_file = st.file_uploader("kc_house_data.csv", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("Файл успешно загружен!")
    st.write("Первые 5 строк данных:")
    st.dataframe(data.head())

    if "price" in data.columns:
        st.warning("Удалена колонка 'price' (целевая переменная)")
        new_data = data.drop("price", axis=1)

    model_name = st.selectbox("Выберите модель", list(models.keys()))
    
    if st.button("Получить предсказания"):
        model = models[model_name]
        predictions = model.predict(new_data)
        
        predictions_df = pd.DataFrame({'Действительное значение (в долларах США $)': data['price'], 'Предсказанное значение (в долларах США $)': predictions})
        
        st.subheader("Результаты")
        st.write("""
        Результаты представлены в виде Таблицы, где в левом столбце находятся реальные значения стоимости дома, 
        а в правом - значения, предсказанные алгоритмом машинного обучения.
        """)
        st.dataframe(predictions_df)
        
        st.download_button(
            label="Скачать предсказания (CSV)",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

def manual_input():
    st.header("2. Ручной ввод параметров")
    
    with st.expander("Основные характеристики", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            bedrooms = st.number_input("Количество спален", min_value=1, max_value=10, value=3)
            bathrooms = st.number_input("Количество ванных", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
        with col2:
            sqft_living = st.number_input("Жилая площадь (кв. футы)", min_value=500, max_value=10000, value=1500)
            sqft_lot = st.number_input("Общая площадь участка (кв. футы)", min_value=500, max_value=50000, value=5000)
        with col3:
            floors = st.number_input("Этажность", min_value=1, max_value=3, value=1)
            waterfront = st.selectbox("Вид на воду", [0, 1], format_func=lambda x: "Да" if x == 1 else "Нет")

    with st.expander("Дополнительные параметры"):
        col1, col2 = st.columns(2)
        with col1:
            view = st.slider("Оценка вида (0-4)", 0, 4, 0)
            condition = st.slider("Состояние дома (1-5)", 1, 5, 3)
            grade = st.slider("Строительный класс (1-13)", 1, 13, 7)
        with col2:
            yr_built = st.number_input("Год постройки", min_value=1900, max_value=2023, value=1990)
            yr_renovated = st.number_input("Год ремонта (0 если не было)", min_value=0, max_value=2023, value=0)
    
    with st.expander("Технические параметры", expanded=False):
        sqft_above = st.number_input("Площадь над землей (кв. футы)", min_value=500, value=1500)
        sqft_basement = st.number_input("Площадь подвала (кв. футы)", min_value=0, value=0)
        zipcode = st.number_input("Почтовый индекс", min_value=98001, max_value=98199, value=98178)
        lat = st.number_input("Широта", min_value=47.0, max_value=48.0, value=47.5)
        long = st.number_input("Долгота", min_value=-123.0, max_value=-121.0, value=-122.0)
        sqft_living15 = st.number_input("Средняя жилая площадь соседей (кв. футы)", min_value=500, value=1500)
        sqft_lot15 = st.number_input("Средняя площадь участка соседей (кв. футы)", min_value=500, value=5000)

    input_data = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'sqft_lot': sqft_lot,
        'floors': floors,
        'waterfront': waterfront,
        'view': view,
        'condition': condition,
        'grade': grade,
        'sqft_above': sqft_above,
        'sqft_basement': sqft_basement,
        'yr_built': yr_built,
        'yr_renovated': yr_renovated,
        'zipcode': zipcode,
        'lat': lat,
        'long': long,
        'sqft_living15': sqft_living15,
        'sqft_lot15': sqft_lot15
    }

    return pd.DataFrame([input_data])

manual_data = manual_input()

if st.button("Получить предсказание (ручной ввод)"):
    model_name = st.selectbox("Выберите модель", list(models.keys()), key="manual_model")
    model = models[model_name]
    
    try:
        prediction = model.predict(manual_data)[0]
        st.success(f"### Предсказанная стоимость дома: **${prediction:,.2f}**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Предсказанная стоимость", f"${prediction:,.0f}")
        with col2:
            st.write("""
            **Пояснение:**  
            Модель учитывает все введенные параметры.
            """)
    except Exception as e:
        st.error(f"Ошибка: {str(e)}")