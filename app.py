import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Car price prediction", layout="wide")

TRAIN_URL = 'https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv'

cat_cols = ['fuel', 'transmission', 'owner', 'seller_type']
cols_to_numeric = ['mileage', 'engine', 'max_power']


@st.cache_resource
def load_model():
    with open('model.pickle', 'rb') as f:
        return pickle.load(f)


@st.cache_data
def load_train():
    # та же предобработка, что и в ноутбуке
    df = pd.read_csv(TRAIN_URL)
    no_target = df.drop(['selling_price'], axis=1)
    df = df.drop(no_target[no_target.duplicated()].index, axis=0).reset_index(drop=True)
    df = df.drop(['torque'], axis=1)
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col].str.split().str[0], errors='coerce')
    num = df.select_dtypes(include='number').columns
    df[num] = df[num].fillna(df[num].median())
    df[['engine', 'seats']] = df[['engine', 'seats']].astype(int)
    return df


def preprocess(df, file):
    # приводим датафрейм к виду, на котором училась модель
    df = df.copy()
    for col in ['selling_price', 'name', 'torque']:
        if col in df.columns:
            df = df.drop([col], axis=1)
    for col in cols_to_numeric:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].str.split().str[0], errors='coerce')
    for col, med in file['medians'].items():
        if col in df.columns:
            df[col] = df[col].fillna(med)
    df['seats'] = df['seats'].astype(int)
    df = pd.get_dummies(df, columns=cat_cols + ['seats'], dtype=int)
    df = df.reindex(columns=file['columns'], fill_value=0)
    return df


def business_metric(y_true, y_pred):
    n = len(y_true)
    return float(np.sum(np.abs(y_pred - y_true) / y_true <= 0.1) / n)


file = load_model()
model = file['model']
df_train = load_train()

st.title("Предсказание стоимости автомобиля")
st.write("ДЗ №1, Егупов А.С. Модель: Ridge + OHE категориальных признаков (test R2 = 0.645)")

page = st.sidebar.radio("Раздел", ["EDA", "Предсказание", "Веса модели"])

# ---------------- EDA ----------------
if page == "EDA":
    st.header("EDA по тренировочным данным")
    st.write(f"Размер train после удаления дубликатов: {df_train.shape}")

    if st.checkbox("Показать данные"):
        st.dataframe(df_train.head(50))

    st.subheader("Распределение цены")
    log_scale = st.checkbox("Логарифм цены", value=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    if log_scale:
        sns.histplot(np.log1p(df_train['selling_price']), bins=50, ax=ax)
        ax.set_xlabel('log(selling_price)')
    else:
        sns.histplot(df_train['selling_price'], bins=50, ax=ax)
    st.pyplot(fig)

    st.subheader("Гистограмма числового признака")
    num_cols = df_train.select_dtypes(include='number').columns.to_list()
    col = st.selectbox("Признак", [c for c in num_cols if c != 'selling_price'])
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df_train[col], bins=40, ax=ax)
    st.pyplot(fig)

    st.subheader("Зависимость цены от признака")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(df_train, x=col, y='selling_price', alpha=0.3, ax=ax)
    st.pyplot(fig)

    st.subheader("Корреляции Пирсона")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_train.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("Цена по категориальным признакам")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, c in zip(axes.flatten(), cat_cols):
        sns.boxplot(df_train, x=c, y='selling_price', ax=ax)
        ax.set_yscale('log')
        ax.set_title(c)
        if c == 'owner':
            ax.tick_params(axis='x', rotation=-30)
    plt.tight_layout()
    st.pyplot(fig)

# ---------------- Предсказание ----------------
elif page == "Предсказание":
    st.header("Предсказание цены")
    mode = st.radio("Способ ввода", ["Ввести вручную", "Загрузить CSV"], horizontal=True)

    if mode == "Ввести вручную":
        cats = file['categories']
        c1, c2 = st.columns(2)
        with c1:
            year = st.number_input("year", 1980, 2025, 2015)
            km_driven = st.number_input("km_driven", 0, 3000000, 60000, step=1000)
            mileage = st.number_input("mileage (kmpl)", 0.0, 50.0, 20.0)
            engine = st.number_input("engine (CC)", 600, 7000, 1248)
            max_power = st.number_input("max_power (bhp)", 0.0, 600.0, 80.0)
        with c2:
            seats = st.selectbox("seats", [2, 4, 5, 6, 7, 8, 9, 10, 14], index=2)
            fuel = st.selectbox("fuel", cats['fuel'])
            seller_type = st.selectbox("seller_type", cats['seller_type'])
            transmission = st.selectbox("transmission", cats['transmission'])
            owner = st.selectbox("owner", cats['owner'])

        row = {
            'year': year, 'km_driven': km_driven, 'fuel': fuel,
            'seller_type': seller_type, 'transmission': transmission,
            'owner': owner, 'mileage': mileage, 'engine': engine,
            'max_power': max_power, 'seats': seats,
        }
        if st.button("Предсказать"):
            X = preprocess(pd.DataFrame([row]), file)
            pred = model.predict(X)[0]
            st.success(f"Предсказанная цена: {int(pred):,}".replace(',', ' '))
            if pred < 0:
                st.warning("Модель линейная, поэтому для старых/слабых машин может уйти в минус")

    else:
        st.write("CSV должен быть в формате cars_test.csv (колонка selling_price не обязательна)")
        uploaded = st.file_uploader("CSV файл", type='csv')
        if uploaded is not None:
            df_in = pd.read_csv(uploaded)
            st.write("Загружено строк:", len(df_in))
            st.dataframe(df_in.head())
            try:
                X = preprocess(df_in, file)
                preds = model.predict(X)
            except Exception as e:
                st.error(f"Не получилось обработать файл: {e}")
                st.stop()

            res = df_in.copy()
            res['predicted_price'] = preds.astype(int)
            st.dataframe(res)

            if 'selling_price' in df_in.columns:
                from sklearn.metrics import r2_score, mean_squared_error
                y = df_in['selling_price']
                st.write(f"R2: {r2_score(y, preds):.3f}")
                st.write(f"RMSE: {int(np.sqrt(mean_squared_error(y, preds)))}")
                st.write(f"Business metric (доля в пределах 10%): {business_metric(y, preds):.3f}")

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(y, preds, alpha=0.3)
                lim = max(y.max(), preds.max())
                ax.plot([0, lim], [0, lim], 'r--')
                ax.set_xlabel('real price')
                ax.set_ylabel('predicted price')
                st.pyplot(fig)

            st.download_button("Скачать предсказания", res.to_csv(index=False), "predictions.csv")

# ---------------- Веса ----------------
else:
    st.header("Веса модели")
    st.write(f"alpha = {model.alpha:.2f}, intercept = {int(model.intercept_)}")
    coefs = pd.Series(model.coef_, index=file['columns'])

    only_cat = st.checkbox("Показать только OHE-признаки")
    st.caption("Модель обучалась без стандартизации, поэтому веса числовых признаков "
               "нельзя напрямую сравнивать с весами OHE-признаков (они в разных единицах)")
    if only_cat:
        coefs = coefs[[c for c in coefs.index if '_' in c and c != 'km_driven' and c != 'max_power']]

    coefs = coefs.sort_values()
    fig, ax = plt.subplots(figsize=(8, len(coefs) * 0.35 + 1))
    colors = ['tab:red' if v < 0 else 'tab:green' for v in coefs.values]
    ax.barh(coefs.index, coefs.values, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('coef')
    st.pyplot(fig)

    st.dataframe(coefs.rename('coef').to_frame())
