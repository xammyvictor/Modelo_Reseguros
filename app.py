import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import os

# Configuración de página
st.set_page_config(page_title="Seguros - Cotizador de Primas", layout="centered")

# --- CONSTANTES ---
MAE_VALOR = 71322.57
MODEL_FILE = 'pipeline_modelo_primas.pkl'
FEATURES_FILE = 'features_list.pkl'

# --- CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    # 1. Verificar si los archivos existen en el directorio actual
    if not os.path.exists(MODEL_FILE):
        return None, f"Archivo no encontrado: {MODEL_FILE}"
    if not os.path.exists(FEATURES_FILE):
        return None, f"Archivo no encontrado: {FEATURES_FILE}"
    
    try:
        # 2. Intentar cargar los archivos
        model = joblib.load(MODEL_FILE)
        features = joblib.load(FEATURES_FILE)
        return (model, features), None
    except Exception as e:
        return None, f"Error técnico al cargar: {str(e)}"

# Intentar cargar
model_data, error_msg = load_model()

# --- INTERFAZ ---
st.title("🛡️ Sistema de Cotización de Primas")
st.markdown("""
Esta herramienta utiliza un modelo de inteligencia artificial (XGBoost) para sugerir el valor de la prima. 
El cálculo incluye un rango de variación basado en el error medio del modelo ($MAE$).
""")

if error_msg:
    st.error(f"### ❌ Error de Configuración")
    st.warning(error_msg)
    st.info("""
    **Instrucciones para solucionar:**
    1. Asegúrate de que los archivos `pipeline_modelo_primas.pkl` y `features_list.pkl` estén subidos a la raíz de tu repositorio en GitHub.
    2. Verifica que los nombres coincidan exactamente (mayúsculas y minúsculas).
    3. Si el error persiste, vuelve a generar los archivos en Google Colab con `joblib.dump` y súbelos de nuevo.
    """)
else:
    pipeline, features_model = model_data
    
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
        try:
            # --- INGENIERÍA DE VARIABLES ---
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
            
            # Asegurar columnas y orden
            for col in features_model:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            input_df = input_df[features_model]
            
            # --- PREDICCIÓN ---
            with st.spinner('Analizando perfil de riesgo...'):
                prediccion = pipeline.predict(input_df)[0]
                valor_min = max(0, prediccion - MAE_VALOR)
                valor_max = prediccion + MAE_VALOR
                
                st.divider()
                st.subheader("Resultado de la Cotización")
                
                # Métrica principal de Streamlit
                st.metric(label="Prima Sugerida (Valor Central)", value=f"${prediccion:,.0f} COP")
                
                # --- DISEÑO MEJORADO DEL RANGO (Sin asteriscos) ---
                st.markdown(f"""
                <div style="
                    background-color: #f0f7ff; 
                    padding: 24px; 
                    border-radius: 12px; 
                    border-left: 6px solid #007bff; 
                    margin-top: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                ">
                    <p style="margin: 0; color: #0056b3; font-weight: 600; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 0.5px;">
                        Rango Estimado de Ajuste
                    </p>
                    <p style="margin: 8px 0; color: #444; font-size: 0.95rem;">
                        Debido a la variabilidad estadística, el valor puede oscilar entre:
                    </p>
                    <div style="display: flex; align-items: baseline; gap: 10px; margin-top: 5px;">
                        <span style="font-size: 1.8rem; font-weight: 800; color: #222;">
                            ${valor_min:,.0f}
                        </span>
                        <span style="font-size: 1.2rem; color: #888; font-weight: 400;">y</span>
                        <span style="font-size: 1.8rem; font-weight: 800; color: #222;">
                            ${valor_max:,.0f}
                        </span>
                        <span style="font-size: 1.1rem; font-weight: 600; color: #444; margin-left: 5px;">COP</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("Ver detalles técnicos del modelo"):
                    st.write(f"**Predicción Central:** ${prediccion:,.2f}")
                    st.write(f"**Margen de Error (MAE):** ±${MAE_VALOR:,.2f}")
                    st.write("**Confiabilidad:** El modelo se ajusta al perfil demográfico máximo registrado para este grupo.")
                
                st.balloons()
                
        except Exception as e:
            st.error(f"Error durante la predicción: {e}")

st.markdown("---")
st.caption("Desarrollado para la optimización de beneficios Aseguradora-Cliente.")
