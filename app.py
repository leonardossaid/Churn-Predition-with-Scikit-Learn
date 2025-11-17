import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score)


st.set_page_config(page_title="Churn Predictor - Demo", layout="wide")


# ---- CONFIG ----
MODEL_PATH = "churn_model.pkl"  # coloque aqui o caminho do seu joblib

# CATEGORIAS extraídas do dataset Telco (já incorporadas)
cat_options = {
    "gender": ["Female", "Male"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No phone service", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
}

# Defaults for numeric fields (medians from train)
numeric_defaults = {
    "SeniorCitizen": 0,
    "tenure": 29,
    "MonthlyCharges": 70.35,
    # TotalCharges may be string in original CSV; treat as numeric with default = tenure*MonthlyCharges approx or null
    "TotalCharges": None
}

# ---- Load model ----
@st.cache_data(ttl=600)
def load_model(path=MODEL_PATH):
    model = joblib.load(path)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Erro ao carregar o modelo ({MODEL_PATH}). Verifique se o arquivo existe e é compatível. Erro: {e}")
    st.stop()

st.title("📈 Churn Predictor — Demo")
st.write("App demo que usa a pipeline treinada (preprocessing + modelo). Insira os dados do cliente e pressione 'Predizer'.")

# Sidebar: threshold and batch upload
st.sidebar.header("Configurações")
threshold = st.sidebar.slider("Threshold (probabilidade >= t → churn)", 0.0, 1.0, 0.50, 0.01)
st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Upload CSV para predições em lote (mesmas colunas do treino)", type=["csv"])

# ---- Input form for single customer ----
st.subheader("Preencha os dados do cliente (entrada única)")

with st.form("input_form"):
    # categorical inputs
    gender = st.selectbox("gender", cat_options["gender"])
    senior = st.selectbox("SeniorCitizen", [0, 1], index=0)
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("tenure (months)", min_value=0, max_value=1000, value=int(numeric_defaults["tenure"]))
    phone = st.selectbox("PhoneService", cat_options["PhoneService"])
    multiple_lines = st.selectbox("MultipleLines", cat_options["MultipleLines"])
    internet = st.selectbox("InternetService", cat_options["InternetService"])
    online_sec = st.selectbox("OnlineSecurity", cat_options["OnlineSecurity"])
    online_backup = st.selectbox("OnlineBackup", cat_options["OnlineBackup"])
    device_prot = st.selectbox("DeviceProtection", cat_options["DeviceProtection"])
    tech_support = st.selectbox("TechSupport", cat_options["TechSupport"])
    streaming_tv = st.selectbox("StreamingTV", cat_options["StreamingTV"])
    streaming_movies = st.selectbox("StreamingMovies", cat_options["StreamingMovies"])
    contract = st.selectbox("Contract", cat_options["Contract"])
    paperless = st.selectbox("PaperlessBilling", cat_options["PaperlessBilling"])
    payment = st.selectbox("PaymentMethod", cat_options["PaymentMethod"])
    monthly = st.number_input("MonthlyCharges", min_value=0.0, max_value=10000.0, value=float(numeric_defaults["MonthlyCharges"]))
    # TotalCharges may be blank or string in original; allow float with default = monthly * tenure
    total_default = monthly * tenure if numeric_defaults["TotalCharges"] is None else numeric_defaults["TotalCharges"]
    total_charges = st.number_input("TotalCharges", min_value=0.0, max_value=1e7, value=float(total_default))

    submitted = st.form_submit_button("Predizer")

if submitted:
    input_dict = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_prot,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total_charges
    }

    df_input = pd.DataFrame([input_dict])
    # Ensure numeric types
    df_input["SeniorCitizen"] = pd.to_numeric(df_input["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    df_input["tenure"] = pd.to_numeric(df_input["tenure"], errors="coerce").fillna(0)
    df_input["MonthlyCharges"] = pd.to_numeric(df_input["MonthlyCharges"], errors="coerce").fillna(0.0)
    df_input["TotalCharges"] = pd.to_numeric(df_input["TotalCharges"], errors="coerce").fillna(0.0)

    try:
        proba = model.predict_proba(df_input)[:, 1][0]
        pred = int(proba >= threshold)
        st.metric("Probabilidade de churn", f"{proba:.4f}")
        st.metric("Predição (0 = stay, 1 = churn)", pred)
    except Exception as e:
        st.error(f"Erro ao prever. Verifique se as colunas e tipos de entrada batem com o treinamento. Erro: {e}")

# ---- Batch prediction ----
if uploaded is not None:
    st.write("### Avaliação em lote (arquivo enviado)")
    df = pd.read_csv(uploaded)
    st.write("Amostra do arquivo:")
    st.write(df.head())

    # Check for required columns
    required_cols = [
        "gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService",
        "MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
        "TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling",
        "PaymentMethod","MonthlyCharges","TotalCharges"
    ]
    missing = set(required_cols) - set(df.columns)
    if missing:
        st.error(f"Arquivo enviado não contém colunas necessárias: {missing}")
    else:
        X_batch = df[required_cols].copy()
        # coerce numeric
        for c in ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]:
            X_batch[c] = pd.to_numeric(X_batch[c], errors="coerce").fillna(0)
        # predict
        try:
            y_proba = model.predict_proba(X_batch)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)
            out = X_batch.copy()
            out["churn_proba"] = y_proba
            out["churn_pred"] = y_pred
            st.write(out.head())

            # if 'Churn' present, compute metrics
            if "Churn" in df.columns:
                y_true = df["Churn"].map({"No":0,"Yes":1}) if df["Churn"].dtype==object else pd.to_numeric(df["Churn"], errors="coerce")
                st.write("AUC:", roc_auc_score(y_true, y_proba))
                st.text(classification_report(y_true, y_pred, digits=4))

                precision, recall, _ = precision_recall_curve(y_true, y_proba)
                ap = average_precision_score(y_true, y_proba)
                fig, ax = plt.subplots()
                ax.plot(recall, precision, label=f"AP={ap:.4f}")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.legend()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Erro durante predição em lote: {e}")

st.write("---")
st.caption("Atenção: este app assume que best_model.joblib é a pipeline completa (pré-processamento + modelo).")