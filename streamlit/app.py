import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Amazon Delivery Time Prediction",
    page_icon="🚚",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🚚 Amazon Delivery Time Prediction System")
st.markdown("Predict delivery time using Machine Learning with interactive UI")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("amazon_delivery.csv")

df = load_data()

# ---------------- TRAIN MODEL (GLOBAL) ----------------
@st.cache_resource
def train_model(df):

    df_model = df.copy()

    le_dict = {}
    for col in ["Weather", "Traffic", "Vehicle", "Area", "Category"]:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        le_dict[col] = le

    X = df_model.drop(
        ["Order_ID", "Delivery_Time", "Order_Date", "Order_Time", "Pickup_Time"],
        axis=1
    )
    y = df_model["Delivery_Time"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    score = r2_score(y, y_pred)

    return model, scaler, le_dict, X.columns, score

model, scaler, le_dict, feature_cols, model_score = train_model(df)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio(
    "Go to",
    ["Dataset Overview", "EDA", "Model Training", "Prediction"]
)

# ---------------- DATASET OVERVIEW ----------------
if menu == "Dataset Overview":
    st.header("📂 Dataset Overview")
    st.dataframe(df.head())
    st.info(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

# ---------------- EDA ----------------
elif menu == "EDA":
    st.header("📊 Exploratory Data Analysis")

    fig, ax = plt.subplots()
    sns.histplot(df["Delivery_Time"], kde=True, ax=ax)
    st.pyplot(fig)

# ---------------- MODEL TRAINING ----------------
elif menu == "Model Training":
    st.header("🤖 Model Training & Evaluation")
    st.success("✅ Model trained successfully")
    st.metric("R² Score", round(model_score, 3))

# ---------------- PREDICTION ----------------
elif menu == "Prediction":
    st.header("⏱️ Predict Delivery Time")

    col1, col2, col3 = st.columns(3)

    with col1:
        agent_age = st.number_input("Agent Age", 18, 60, 30)
        agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.5)

    with col2:
        weather = st.selectbox("Weather", df["Weather"].unique())
        traffic = st.selectbox("Traffic", df["Traffic"].unique())

    with col3:
        vehicle = st.selectbox("Vehicle", df["Vehicle"].unique())
        area = st.selectbox("Area", df["Area"].unique())
        category = st.selectbox("Category", df["Category"].unique())

    if st.button("🚀 Predict Delivery Time"):

        input_df = pd.DataFrame([{
            "Agent_Age": agent_age,
            "Agent_Rating": agent_rating,
            "Store_Latitude": df["Store_Latitude"].mean(),
            "Store_Longitude": df["Store_Longitude"].mean(),
            "Drop_Latitude": df["Drop_Latitude"].mean(),
            "Drop_Longitude": df["Drop_Longitude"].mean(),
            "Weather": weather,
            "Traffic": traffic,
            "Vehicle": vehicle,
            "Area": area,
            "Category": category
        }])

        for col in ["Weather", "Traffic", "Vehicle", "Area", "Category"]:
            input_df[col] = le_dict[col].transform(input_df[col])

        input_scaled = scaler.transform(input_df[feature_cols])
        prediction = model.predict(input_scaled)

        st.success(f"🕒 Estimated Delivery Time: **{int(prediction[0])} minutes**")
