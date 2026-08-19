import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Clasificador de Minerales")

st.title(" Clasificador Inteligente de Minerales")
st.write("Introduce las propiedades físicas observadas en la muestra para identificar el mineral.")

@st.cache_resource
def load_data_and_model():
    df = pd.read_excel("minerales_core_100.xlsx")
    features = ["Dureza", "Brillo", "Color", "Exfoliacion", "Reaccion_HCl", "Magnetismo", "Densidad", "Sistema"]
    X = df[features].copy()
    y = df["Nombre_ES"]

    encoders = {}
    for col in ["Brillo", "Color", "Exfoliacion", "Reaccion_HCl", "Magnetismo", "Sistema"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf, encoders

clf, encoders = load_data_and_model()

col1, col2 = st.columns(2)

with col1:
    dureza = st.slider("Dureza (Mohs)", 1.0, 10.0, 5.0, step=0.1)
    brillo = st.selectbox("Brillo", encoders["Brillo"].classes_)
    color = st.selectbox("Color", encoders["Color"].classes_)
    exfol = st.selectbox("Exfoliación", encoders["Exfoliacion"].classes_)

with col2:
    hcl = st.selectbox("Reacción a HCl", encoders["Reaccion_HCl"].classes_)
    mag = st.selectbox("Magnetismo", encoders["Magnetismo"].classes_)
    densidad = st.number_input("Densidad (g/cm³)", value=2.7, step=0.1)
    sistema = st.selectbox("Sistema Cristalino", encoders["Sistema"].classes_)

if st.button(" Identificar Mineral", use_container_width=True):
    entrada = {
        "Dureza": [dureza],
        "Brillo": [encoders["Brillo"].transform([brillo])[0]],
        "Color": [encoders["Color"].transform([color])[0]],
        "Exfoliacion": [encoders["Exfoliacion"].transform([exfol])[0]],
        "Reaccion_HCl": [encoders["Reaccion_HCl"].transform([hcl])[0]],
        "Magnetismo": [encoders["Magnetismo"].transform([mag])[0]],
        "Densidad": [densidad],
        "Sistema": [encoders["Sistema"].transform([sistema])[0]]
    }
    
    df_in = pd.DataFrame(entrada)
    probs = clf.predict_proba(df_in)[0]
    clases = clf.classes_
    
    top3_idx = np.argsort(probs)[::-1][:3]
    
    st.subheader("Resultados:")
    for idx in top3_idx:
        if probs[idx] > 0:
            st.write(f"**{clases[idx]}**: {probs[idx]*100:.1f}% de coincidencia")
            st.progress(float(probs[idx]))
