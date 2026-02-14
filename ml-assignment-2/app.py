import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import os

st.title("ML Assignment 2 – Classification Models")

uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

model_name = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    for col in df.select_dtypes(include='object'):
        df[col] = df[col].astype('category').cat.codes

    X = df.drop("income", axis=1)
    y = df["income"]

    current_working_dir = os.getcwd()
    model_path = os.path.join(current_working_dir, 'model', f"{model_name.replace(' ', '_')}.pkl")

    model = joblib.load(model_path)

    y_pred = model.predict(X)

    st.subheader("Evaluation Metrics")
    st.text(classification_report(y, y_pred))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y, y_pred))
