import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# Configuración de página
st.set_page_config(page_title="Seguros - Cotizador de Primas", layout="centered")

# --- CONSTANTES ---
# Definimos el error MAE que obtuviste en tu modelo
MAE_VALOR = 71322.57

# --- CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    try:
        # Aseguramos que el pipeline cargue correctamente
        model = joblib.load('pipeline_modelo_primas.pkl')
        features = joblib.load('features_list.pkl')
        return model, features
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None, None

pipeline, features_model = load_model()

# --- INTERFAZ ---
st.title("🛡️ Sistema de Cotización de Primas")
st.markdown("""
Esta herramienta utiliza un modelo de inteligencia artificial (XGBoost) para sugerir el valor de la prima. 
El cálculo incluye un rango de variación basado en el error medio del modelo ($MAE$).
""")

if pipeline is not None:
    with st.form("form_cliente"):
        st.subheader("Datos Demográficos y de Riesgo")
        c1, c2 = st.columns(2)
        
        with c1:
            edad = st.number_input("Edad", 18, 100, 30)
            imc = st.number_input("Índice de Masa Corporal (IMC)", 10.0, 60.0, 25.0)
            genero = st.selectbox("Género", ["Masculino", "Femenino"])
            fumador = st.selectbox("¿Fumador?", ["Si", "No"])
            
        with c2:
            ciudad = st.selectbox("Ciudad", ["Bogotá", "Medellín", "Cali", "Barranquilla", "Cartagena", "Bucaramanga", "Pereira", "Manizales", "Cúcuta", "Ibagué", "Otra"])
            antecedente = st.selectbox("Antecedentes Familiares", ["Si", "No"])
            siniestro = st.number_input("Valor Siniestro Histórico (COP)", min_value=0.0, value=0.0)

        enviar = st.form_submit_button("Generar Predicción")

    if enviar:
        # --- REPLICAR INGENIERÍA DE VARIABLES ---
        data = {
            "Edad": edad,
            "IMC": imc,
            "Fumador": fumador,
            "Ciudad": ciudad,
            "Antecedente Familiar": antecedente,
            "VALOR SINIESTRO PAGADO COL": siniestro,
            "Género": genero
        }
        
        data["log_siniestro"] = np.log1p(siniestro)
        data["Edad_2"] = edad ** 2
        data["Interaccion_Edad_IMC"] = edad * imc
        
        input_df = pd.DataFrame([data])
        
        # Asegurar columnas
        for col in features_model:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[features_model]
        
        # --- PREDICCIÓN Y RANGO ---
        with st.spinner('Analizando perfil de riesgo...'):
            try:
                prediccion = pipeline.predict(input_df)[0]
                
                # Calcular rango basado en el MAE
                valor_min = max(0, prediccion - MAE_VALOR)
                valor_max = prediccion + MAE_VALOR
                
                # --- MOSTRAR RESULTADOS ---
                st.divider()
                st.subheader("Resultado de la Cotización")
                
                # Métrica principal
                st.metric(label="Prima Sugerida (Base)", value=f"${prediccion:,.0f} COP")
                
                # Rango de oscilación
                st.info(f"""
                **Rango Estimado de Ajuste:** Debido a la variabilidad estadística, el valor puede oscilar entre:  
                ### **${valor_min:,.0f}** y **${valor_max:,.0f} COP**
                """)
                
                # Explicación técnica
                with st.expander("Ver detalles del cálculo"):
                    st.write(f"- **Predicción Central:** ${prediccion:,.2f}")
                    st.write(f"- **Error Medio Aplicado (MAE):** ±${MAE_VALOR:,.2f}")
                    st.write("- **Modelo:** XGBoost Regressor v3.2.0")
                    st.write("- **Variables clave:** El IMC y el historial de siniestros tienen el mayor peso en este resultado.")
                
                st.balloons()
                
            except Exception as e:
                st.error(f"Error durante la predicción: {e}")
else:
    st.warning("⚠️ El archivo del modelo no se encontró o es incompatible.")

st.markdown("---")
st.caption("Desarrollado para la optimización de beneficios Aseguradora-Cliente.")
