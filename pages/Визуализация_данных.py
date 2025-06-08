import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Визуализация данных",
    page_icon="📊",
)

st.title('Визуализация данных')

@st.cache_data
def load_data():
    data = pd.read_csv("C:/Users/kiril/OneDrive/Рабочий стол/Учёба/2 курс/2 семестр/ML/РГР/data/predData_regression.csv")
    return data

data = load_data()

st.subheader("Распределение целевой переменной (price)")
fig1, ax1 = plt.subplots()
median_ = data['price'].median()
mean_ = data['price'].mean()
sns.histplot(data["price"], kde=True, ax=ax1)
plt.axvline(x=median_, color='red', linestyle='dashed', linewidth=2, label=f'Median: {median_:.2f}')
plt.axvline(x=mean_, color='green',linestyle='dashed', linewidth=2, label=f'Mean: {mean_:.2f}')
plt.legend()
st.pyplot(fig1)

continuous_features = ['sqft_living', 'sqft_lot', "sqft_lot15", "sqft_living15", "sqft_basement", "sqft_above"]

st.subheader("Распределение непрерывных признаков")
for feature in continuous_features:
    fig5, ax5 = plt.subplots()
    median_ = data[feature].median()
    mean_ = data[feature].mean()
    sns.histplot(data=data[feature], log=True, kde=True)
    plt.axvline(x=median_, color='red', linestyle='dashed', linewidth=2, label=f'Median: {median_:.2f}')
    plt.axvline(x=mean_, color='green',linestyle='dashed', linewidth=2, label=f'Mean: {mean_:.2f}')
    plt.legend()
    ax5.set_title(f'Распределение признака "{feature}"')
    st.pyplot(fig5)

num_features = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement',
                'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'house_age', "sqft_lot_per_living"]

categor_features = ['floors', 'waterfront', 'view', 'condition', 'grade', "bedrooms"]

st.subheader("Корреляционная матрица непрерывных признаков")
corr = data[num_features].corr(numeric_only=True)
fig2, ax2 = plt.subplots(figsize=(20,10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

st.subheader("Корреляционная матрица категориальных признаков")
corr_matrix = data[categor_features].corr(method='spearman', numeric_only=True)
fig2_2, ax2_2 = plt.subplots(figsize=(20,10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2_2)
st.pyplot(fig2_2)

st.subheader("Корреляция таргета и непрерывных признаков")
fig7, ax7 = plt.subplots()
corr_with_target = data[num_features + ['price']].corr().iloc[:-1, -1].sort_values(ascending=False)
sns.barplot(x=corr_with_target.values, y=corr_with_target.index, ax=ax7)
st.pyplot(fig7)

st.subheader("Корреляция таргета и категориальных признаков")
fig8, ax8 = plt.subplots()
corr_with_target = data[categor_features + ['price']].corr(method='spearman').iloc[:-1, -1].sort_values(ascending=False)
sns.barplot(x=corr_with_target.values, y=corr_with_target.index, ax=ax8)
st.pyplot(fig8)


st.subheader("Boxplot: Цена в разрезе категориальных признаков")
for feature in categor_features:
    fig6, ax6 = plt.subplots()
    sns.boxplot(x=feature, y="price", data=data, ax=ax6)
    st.pyplot(fig6)

important_continuous_features = ['sqft_living', 'sqft_lot', 'lat', 'long', 'sqft_living15', "sqft_lot_per_living"]

st.subheader("Зависимость цены от наиболее важных непрерывных признаков")

for feature in important_continuous_features:
    fig4, ax4 = plt.subplots()
    sns.scatterplot(x=feature, y="price", data=data, ax=ax4)
    st.pyplot(fig4)

st.header('Выводы по EDA')

st.write("""
Факторы, влияющие на цену: 
- **Жилая площадь:**
Знак корреляции положительный, значит если растет жилая площадь, то растет и цена.

- **Индекс качества строительства:**
Данный индекс влияет на цену дома больше всего. Знак корреляции положительный, значит если растет индекс качества строительства, то растет и цена.

- **Жилая площадь соседей:**
Знак корреляции положительный, значит если растет жилая площадь соседей, то растет и цена.

- **Расположение по широте:**
Знак корреляции положительный, значит с ростом широты растет и цена.

- **Количество спален:**
Количество спален незначительно влияет на цену дома. Знак корреляции положительный, значит с ростом количества спален растет и цена.

- **Количество ванных комнат:**
Количество вынных комнат также незначительно влияет на цену дома. Знак корреляции положительный, значит с ростом количества ванных комнат растет и цена.

- **Доля общей площади к жилой:**
Знак корреляции отрицательный, значит если растет доля общей площади к жилой, то цена падает
""")