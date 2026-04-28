import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Cargar el pipeline y las columnas
pipeline = joblib.load('pipeline_modelo_primas.pkl')
features_model = joblib.load('features_list.pkl')

st.set_page_config(page_title="Predicción de Primas de Seguros", layout="wide")

st.title("💰 Calculador Inteligente de Primas")
st.markdown("Ingrese los datos demográficos del cliente para calcular el valor de la prima sugerida.")

# 2. Crear el formulario de entrada
with st.form("datos_cliente"):
    col1, col2 = st.columns(2)
    
    with col1:
        edad = st.number_input("Edad", min_value=18, max_value=100, value=30)
        imc = st.number_input("IMC", min_value=10.0, max_value=50.0, value=25.0)
        genero = st.selectbox("Género", ["Masculino", "Femenino"])
        fumador = st.selectbox("¿Es fumador?", ["Si", "No"])
        
    with col2:
        ciudad = st.selectbox("Ciudad", ["Cali", "Bogotá", "Medellín", "Otra"]) # Ajusta según tu top 10
        antecedente = st.selectbox("Antecedente Familiar", ["Si", "No"])
        siniestro = st.number_input("Valor Siniestro Pagado (COL)", min_value=0.0, value=0.0)
    
    submit = st.form_submit_button("Calcular Prima Sugerida")

if submit:
    # 3. Recrear la lógica de ingeniería de variables de tu modelo
    data = {
        "Edad": edad,
        "IMC": imc,
        "Fumador": fumador,
        "Ciudad": ciudad,
        "Antecedente Familiar": antecedente,
        "VALOR SINIESTRO PAGADO COL": siniestro,
        "Género": genero,
        "log_siniestro": np.log1p(siniestro),
        "Edad_2": edad ** 2,
        "Interaccion_Edad_IMC": edad * imc
    }
    
    input_df = pd.DataFrame([data])
    
    # Asegurar que el orden de las columnas sea el mismo que en el entrenamiento
    input_df = input_df[features_model]
    
    # 4. Predicción
    prediccion = pipeline.predict(input_df)[0]
    
    st.success(f"### Valor de la Prima Sugerida: ${prediccion:,.2f} COP")
    st.info("Este valor busca el equilibrio entre el beneficio técnico de la aseguradora y la competitividad para el cliente.")
