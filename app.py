import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="FinForecast Enterprise", layout="wide", page_icon="📈")

# Estilos CSS para dar un look más "Fintech"
st.markdown("""
    <style>
    .big-font { font-size:20px !important; }
    .metric-container { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

st.title("FinForecast: Suite de Análisis Financiero Predictivo")
st.markdown("""
**Módulo de Proyección Incremental:** Esta interfaz permite la **incorporación de registros de cierre semanal** al dataset histórico. 
El algoritmo ajusta sus coeficientes dinámicamente con cada nuevo registro para refinar la tendencia del Flujo de Caja Operativo.
""")

# --- 1. GESTIÓN DE MEMORIA DE SESIÓN ---
if 'scenario_data' not in st.session_state:
    st.session_state['scenario_data'] = pd.DataFrame(columns=['date', 'sales', 'expenses'])

def clear_scenario():
    st.session_state['scenario_data'] = pd.DataFrame(columns=['date', 'sales', 'expenses'])

# --- 2. MOTOR DE DATOS Y MODELADO ---
@st.cache_data
def load_historical_data():
    try:
        df_main = pd.read_csv('checking_account_main.csv')
        df_sec = pd.read_csv('checking_account_secondary.csv')
        df_cc = pd.read_csv('credit_card_account.csv')
    except FileNotFoundError:
        return None
    
    # Preprocesamiento ETL
    for df in [df_main, df_sec, df_cc]:
        df['date'] = pd.to_datetime(df['date'])
    
    # Ingresos (Sales Revenue)
    daily_sales = df_main[df_main['category'] == 'Sales Revenue'].groupby('date')['amount'].sum().reset_index()
    daily_sales.columns = ['date', 'sales']
    
    # Egresos Operativos (OpEx + Nómina)
    exp_main = df_main[(df_main['type'] == 'Debit') & (df_main['category'] != 'Transfer')]
    exp_sec = df_sec[df_sec['category'] == 'Payroll']
    exp_cc = df_cc[df_cc['type'] == 'Debit']
    all_expenses = pd.concat([exp_main, exp_sec, exp_cc])
    daily_expenses = all_expenses.groupby('date')['amount'].sum().reset_index()
    daily_expenses.columns = ['date', 'expenses']
    
    # Consolidación Semanal
    df_combined = pd.merge(daily_sales, daily_expenses, on='date', how='outer').fillna(0)
    df_combined = df_combined.sort_values('date')
    df_weekly = df_combined.set_index('date').resample('W-MON').sum().reset_index()
    
    return df_weekly

def calibrate_model(df):
    """
    Entrena el modelo de Regresión Lineal con los datos consolidados.
    Retorna los objetos del modelo calibrados.
    """
    df_lags = df.copy()
    # Ingeniería de Características (Lags Temporales)
    for i in [1, 2, 4]:
        df_lags[f'sales_lag_{i}'] = df_lags['sales'].shift(i)
        df_lags[f'expenses_lag_{i}'] = df_lags['expenses'].shift(i)
    
    df_train = df_lags.dropna()
    
    if df_train.empty:
        return None, None

    features = [f'sales_lag_{i}' for i in [1, 2, 4]] + [f'expenses_lag_{i}' for i in [1, 2, 4]]
    X = df_train[features]
    
    # Ajuste de Modelos
    model_s = LinearRegression().fit(X, df_train['sales'])
    model_e = LinearRegression().fit(X, df_train['expenses'])
    
    return model_s, model_e

# Carga Inicial
df_historical = load_historical_data()
if df_historical is None:
    st.error("Error Crítico: No se detectaron los archivos fuente en el directorio raíz.")
    st.stop()

# --- 3. PANEL DE GESTIÓN (SIDEBAR) ---
st.sidebar.header("Gestión de Periodos")
st.sidebar.markdown("---")

# Consolidación de Datos (Histórico + Escenario Actual)
df_consolidated = pd.concat([df_historical, st.session_state['scenario_data']], ignore_index=True)
df_consolidated['date'] = pd.to_datetime(df_consolidated['date'])
df_consolidated = df_consolidated.sort_values('date').reset_index(drop=True)

# Lógica de Fechas
if not df_consolidated.empty:
    last_record_date = df_consolidated['date'].iloc[-1]
    next_period_date = last_record_date + pd.Timedelta(days=7)
else:
    next_period_date = pd.Timestamp.today()

st.sidebar.subheader("Registro de Cierre Semanal")
st.sidebar.caption(f"Periodo sugerido: {next_period_date.date()}")

input_sales = st.sidebar.number_input("Ingresos Totales ($)", min_value=0.0, value=6500.0, step=100.0)
input_exp = st.sidebar.number_input("Egresos Operativos ($)", min_value=0.0, value=2800.0, step=100.0)

col_btn1, col_btn2 = st.sidebar.columns(2)

if col_btn1.button("Registrar Cierre"):
    new_record = pd.DataFrame({'date': [next_period_date], 'sales': [input_sales], 'expenses': [input_exp]})
    st.session_state['scenario_data'] = pd.concat([st.session_state['scenario_data'], new_record], ignore_index=True)
    st.rerun()

if col_btn2.button("Limpiar Escenario"):
    clear_scenario()
    st.rerun()

# --- 4. DASHBOARD DE RENDIMIENTO ---
st.subheader("Estado Financiero Actual")

if not df_consolidated.empty:
    # Cálculo de Variaciones (Delta)
    curr_sales = df_consolidated['sales'].iloc[-1]
    curr_exp = df_consolidated['expenses'].iloc[-1]
    
    prev_sales = df_consolidated['sales'].iloc[-2] if len(df_consolidated) > 1 else curr_sales
    prev_exp = df_consolidated['expenses'].iloc[-2] if len(df_consolidated) > 1 else curr_exp
    
    delta_sales = curr_sales - prev_sales
    delta_exp = curr_exp - prev_exp

    col1, col2, col3 = st.columns(3)
    col1.metric("Ingresos (Cierre Reciente)", f"${curr_sales:,.2f}", f"{delta_sales:,.2f}")
    col2.metric("Egresos (Cierre Reciente)", f"${curr_exp:,.2f}", f"{delta_exp:,.2f}", delta_color="inverse")
    col3.metric("Periodos Proyectados Adicionales", len(st.session_state['scenario_data']))

st.divider()

# Gráfica de Tendencia
st.subheader("Análisis de Tendencia Histórica y Escenarios")

if not df_consolidated.empty:
    df_consolidated['sales'] = df_consolidated['sales'].astype(float)
    df_consolidated['expenses'] = df_consolidated['expenses'].astype(float)
    
    # Visualización
    chart_data = df_consolidated.set_index('date')[['sales', 'expenses']]
    chart_data.columns = ['Ingresos', 'Egresos'] # Etiquetas en español profesional
    st.line_chart(chart_data, color=["#0000FF", "#FF0000"])

# Visualización Tabular Diferenciada
with st.expander("Desglose de Registros (Auditoría de Datos)"):
    df_audit = df_consolidated.copy()
    # Etiquetado de origen de datos
    df_audit['Origen'] = ['Histórico'] * len(df_historical) + ['Escenario Manual'] * len(st.session_state['scenario_data'])
    
    # Formato condicional para identificar registros manuales
    st.dataframe(df_audit.sort_values('date', ascending=False).style.format({
        "sales": "${:,.2f}", "expenses": "${:,.2f}"
    }).applymap(
        lambda x: 'background-color: #fffdc1' if x == 'Escenario Manual' else '', subset=['Origen']
    ))

# --- 5. MÓDULO DE PREDICCIÓN (FORECASTING) ---
st.divider()
st.subheader("Proyección de Flujo de Caja (Q+1)")
st.markdown("Generación de pronóstico basado en la calibración más reciente del algoritmo.")

if st.button("Ejecutar Modelo Predictivo", type="primary"):
    
    with st.spinner('Procesando vectores y calibrando coeficientes...'):
        # 1. Calibración en Tiempo Real
        model_s_live, model_e_live = calibrate_model(df_consolidated)
        
        if model_s_live is None:
            st.warning("Insuficiencia de datos históricos para generar una proyección fiable.")
        else:
            # 2. Forecasting Recursivo
            history_sales = list(df_consolidated['sales'].values)
            history_expenses = list(df_consolidated['expenses'].values)
            
            future_dates = pd.date_range(start=df_consolidated['date'].iloc[-1] + pd.Timedelta(days=7), periods=4, freq='W-MON')
            forecast_data = []
            
            for date in future_dates:
                # Construcción del vector de características (Lags)
                features = [
                    history_sales[-1], history_sales[-2], history_sales[-4],
                    history_expenses[-1], history_expenses[-2], history_expenses[-4]
                ]
                
                # Inferencia
                pred_s = model_s_live.predict([features])[0]
                pred_e = model_e_live.predict([features])[0]
                
                forecast_data.append({
                    'Fecha Proyectada': date.date(), 
                    'Ingresos Estimados': float(pred_s), 
                    'Egresos Estimados': float(pred_e),
                    'Flujo Neto': float(pred_s - pred_e)
                })
                
                # Recursividad (Alimentar el futuro con la predicción actual)
                history_sales.append(pred_s)
                history_expenses.append(pred_e)
            
            # 3. Presentación de Resultados
            df_forecast = pd.DataFrame(forecast_data)
            
            c1, c2 = st.columns([1, 1.5])
            with c1:
                st.success("Proyección generada exitosamente.")
                st.table(df_forecast.style.format({
                    "Ingresos Estimados": "${:,.2f}", 
                    "Egresos Estimados": "${:,.2f}",
                    "Flujo Neto": "${:,.2f}"
                }))
                
            with c2:
                st.markdown("##### Visualización de Tendencia Futura")
                st.line_chart(df_forecast.set_index('Fecha Proyectada')[['Ingresos Estimados', 'Egresos Estimados']])