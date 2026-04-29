import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import os

# Configuración de página
st.set_page_config(page_title="Seguros - Cotizador de Primas", layout="centered")

# --- CONSTANTES ---
# Actualizado según la nueva métrica del modelo
MAE_VALOR = 70312.0
MODEL_FILE = 'pipeline_modelo_primas.pkl'
FEATURES_FILE = 'features_list.pkl'

# --- CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None, f"Archivo no encontrado: {MODEL_FILE}"
    if not os.path.exists(FEATURES_FILE):
        return None, f"Archivo no encontrado: {FEATURES_FILE}"
    
    try:
        model = joblib.load(MODEL_FILE)
        features = joblib.load(FEATURES_FILE)
        return (model, features), None
    except Exception as e:
        return None, f"Error técnico al cargar: {str(e)}"

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
    1. Asegúrate de haber re-entrenado el modelo con las nuevas opciones y haber generado los nuevos archivos `.pkl`.
    2. Verifica que los nombres de los archivos en GitHub sean exactos.
    """)
else:
    pipeline, features_model = model_data
    
    with st.form("form_cliente"):
        st.subheader("Datos Demográficos y de Salud")
        c1, c2 = st.columns(2)
        
        with c1:
            edad = st.number_input("Edad", 18, 100, 30)
            peso = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0)
            altura = st.number_input("Altura (cm)", min_value=100.0, max_value=250.0, value=170.0)
            genero = st.selectbox("Género", ["Masculino", "Femenino"])
            fumador = st.selectbox("¿Fumador?", ["No", "Si"])
            # Opciones actualizadas según solicitud
            actividad = st.selectbox("Actividad Física", ["Activo", "Moderado", "Sedentario"])
            
        with c2:
            ciudad = st.selectbox("Ciudad", ["Bogotá", "Medellín", "Cali", "Barranquilla", "Cartagena", "Bucaramanga", "Pereira", "Manizales", "Cúcuta", "Ibagué", "Otra"])
            antecedente = st.selectbox("Antecedentes Familiares", ["No", "Si"])
            # Opciones de Patología actualizadas
            patologia = st.selectbox("Patología", ["Asma", "Cardiopatía", "Diabetes", "Hipertensión", "Ninguna"])
            # Opciones de Alergias actualizadas
            alergias = st.selectbox("Alergias", ["Gluten", "Lácteos", "Medicamentos", "Ninguna", "Polen"])
            siniestro = st.number_input("Valor Siniestro Histórico (COP)", min_value=0.0, value=0.0)

        enviar = st.form_submit_button("Generar Predicción")

    if enviar:
        try:
            # --- CÁLCULO DE IMC ---
            altura_m = altura / 100
            imc_calculado = peso / (altura_m ** 2)
            
            # --- INGENIERÍA DE VARIABLES ---
            data = {
                "Edad": edad,
                "IMC": imc_calculado,
                "Fumador": fumador,
                "Ciudad": ciudad,
                "Antecedente Familiar": antecedente,
                "VALOR SINIESTRO PAGADO COL": siniestro,
                "Género": genero,
                "Patología": patologia,
                "Alergias": alergias,
                "Actividad Física": actividad
            }
            
            # Variables derivadas del entrenamiento
            data["log_siniestro"] = np.log1p(siniestro)
            data["Edad_2"] = edad ** 2
            data["Interaccion_Edad_IMC"] = edad * imc_calculado
            
            input_df = pd.DataFrame([data])
            
            # Asegurar columnas y orden según el modelo entrenado
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
                
                st.metric(label="Prima Sugerida (Valor Central)", value=f"${prediccion:,.0f} COP")
                
                st.write(f"**IMC Calculado:** {imc_calculado:.2f}")

                # Diseño del rango optimizado (sin asteriscos)
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
                        Basado en el perfil de salud y riesgo, el valor oscila entre:
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
                
                with st.expander("Ver detalles técnicos"):
                    st.write(f"**Predicción Central:** ${prediccion:,.2f}")
                    st.write(f"**Margen de Error (MAE):** ±${MAE_VALOR:,.2f}")
                    st.write(f"**Perfil:** {patologia} | {alergias} | {actividad}")
                
                st.balloons()
                
        except Exception as e:
            st.error(f"Error durante la predicción: {e}")
            st.info("Asegúrate de que las opciones seleccionadas coincidan con las categorías del modelo entrenado.")

st.markdown("---")
st.caption("Desarrollado para la optimización de beneficios Aseguradora-Cliente.")
