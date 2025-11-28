import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="FinForecast Pro", layout="wide")

st.title("FinForecast: Simulador Evolutivo de Flujo de Caja")
st.markdown("""
Esta herramienta permite **inyectar datos secuenciales** al modelo. 
Cada vez que agregas una semana, el sistema **se re-entrena automáticamente** para aprender de la nueva tendencia sin alterar los datos originales.
""")

# --- 1. GESTIÓN DE ESTADO (MEMORIA TEMPORAL) ---
if 'new_data' not in st.session_state:
    st.session_state['new_data'] = pd.DataFrame(columns=['date', 'sales', 'expenses'])

def reset_simulation():
    st.session_state['new_data'] = pd.DataFrame(columns=['date', 'sales', 'expenses'])

# --- 2. FUNCIONES DE CARGA Y ENTRENAMIENTO ---
@st.cache_data
def load_original_data():
    # Cargar CSVs
    try:
        df_main = pd.read_csv('checking_account_main.csv')
        df_sec = pd.read_csv('checking_account_secondary.csv')
        df_cc = pd.read_csv('credit_card_account.csv')
    except FileNotFoundError:
        return None
    
    # Preprocesar
    for df in [df_main, df_sec, df_cc]:
        df['date'] = pd.to_datetime(df['date'])
    
    # Agrupar
    daily_sales = df_main[df_main['category'] == 'Sales Revenue'].groupby('date')['amount'].sum().reset_index()
    daily_sales.columns = ['date', 'sales']
    
    exp_main = df_main[(df_main['type'] == 'Debit') & (df_main['category'] != 'Transfer')]
    exp_sec = df_sec[df_sec['category'] == 'Payroll']
    exp_cc = df_cc[df_cc['type'] == 'Debit']
    all_expenses = pd.concat([exp_main, exp_sec, exp_cc])
    daily_expenses = all_expenses.groupby('date')['amount'].sum().reset_index()
    daily_expenses.columns = ['date', 'expenses']
    
    df_combined = pd.merge(daily_sales, daily_expenses, on='date', how='outer').fillna(0)
    df_combined = df_combined.sort_values('date')
    df_weekly = df_combined.set_index('date').resample('W-MON').sum().reset_index()
    
    return df_weekly

def train_model(df):
    # Crear Lags
    df_lags = df.copy()
    for i in [1, 2, 4]:
        df_lags[f'sales_lag_{i}'] = df_lags['sales'].shift(i)
        df_lags[f'expenses_lag_{i}'] = df_lags['expenses'].shift(i)
    
    df_train = df_lags.dropna()
    
    if df_train.empty:
        return None, None

    X = df_train[[f'sales_lag_{i}' for i in [1, 2, 4]] + [f'expenses_lag_{i}' for i in [1, 2, 4]]]
    y_s = df_train['sales']
    y_e = df_train['expenses']
    
    model_s = LinearRegression().fit(X, y_s)
    model_e = LinearRegression().fit(X, y_e)
    
    return model_s, model_e

# Cargar datos base
df_original = load_original_data()
if df_original is None:
    st.error("No se encuentran los archivos CSV originales (checking_account_main.csv, etc).")
    st.stop()

# --- 3. BARRA LATERAL: INYECCIÓN DE DATOS ---
st.sidebar.header("Panel de Control")

# Combinar Histórico Real + Datos Simulados
df_total = pd.concat([df_original, st.session_state['new_data']], ignore_index=True)
df_total['date'] = pd.to_datetime(df_total['date'])
df_total = df_total.sort_values('date').reset_index(drop=True)

# Calcular fecha sugerida (Siguiente Lunes)
if not df_total.empty:
    last_date = df_total['date'].iloc[-1]
    next_date = last_date + pd.Timedelta(days=7)
else:
    next_date = pd.Timestamp.today()

st.sidebar.subheader("Agregar Nueva Semana")
st.sidebar.info(f"Fecha sugerida: {next_date.date()}")

input_sales = st.sidebar.number_input("Ventas ($)", min_value=0.0, value=6000.0, step=500.0)
input_exp = st.sidebar.number_input("Gastos ($)", min_value=0.0, value=2500.0, step=500.0)

if st.sidebar.button("Agregar Dato y Re-Entrenar"):
    new_row = pd.DataFrame({'date': [next_date], 'sales': [input_sales], 'expenses': [input_exp]})
    # Guardar en Session State (Memoria)
    st.session_state['new_data'] = pd.concat([st.session_state['new_data'], new_row], ignore_index=True)
    st.rerun() # Recargar la página para mostrar cambios

if st.sidebar.button("Borrar Simulación (Reset)"):
    reset_simulation()
    st.rerun()

# --- 4. VISUALIZACIÓN DE DATOS ---
# KPIs
if not df_total.empty:
    last_sales = df_total['sales'].iloc[-1]
    last_expenses = df_total['expenses'].iloc[-1]
    prev_sales = df_total['sales'].iloc[-2] if len(df_total) > 1 else last_sales
    sales_delta = last_sales - prev_sales
    expenses_delta = last_expenses - (df_total['expenses'].iloc[-2] if len(df_total) > 1 else last_expenses)

    col1, col2, col3 = st.columns(3)
    col1.metric("Ventas (Última Semana)", f"${last_sales:,.2f}", f"{sales_delta:,.2f}")
    col2.metric("Gastos (Última Semana)", f"${last_expenses:,.2f}", f"{expenses_delta:,.2f}", delta_color="inverse")
    col3.metric("Semanas Simuladas", len(st.session_state['new_data']))

st.divider()

st.subheader("Evolución de Ventas y Gastos")

# Gráfico interactivo
if not df_total.empty:
    # Asegurar tipos de datos correctos para evitar errores de Streamlit
    df_total['sales'] = df_total['sales'].astype(float)
    df_total['expenses'] = df_total['expenses'].astype(float)
    
    st.line_chart(df_total, x='date', y=['sales', 'expenses'])

# Tabla de datos en expander
with st.expander("Ver datos detallados"):
    df_display = df_total.copy()
    df_display['Tipo'] = ['Real'] * len(df_original) + ['Simulado'] * len(st.session_state['new_data'])
    st.dataframe(df_display.sort_values('date', ascending=False).style.applymap(
        lambda x: 'background-color: #d1e7dd' if x == 'Simulado' else '', subset=['Tipo']
    ))

# --- 5. PREDICCIÓN FUTURA (FORECAST) ---
st.divider()
st.subheader("Proyección a Futuro (Próximas 4 Semanas)")

if st.button("Entrenar IA con Datos Actuales y Predecir", type="primary"):
    
    with st.spinner('Re-entrenando modelo con los nuevos datos...'):
        # 1. Re-entrenar modelo en vivo con TODOS los datos (Real + Simulado)
        model_s_live, model_e_live = train_model(df_total)
        
        if model_s_live is None:
            st.error("No hay suficientes datos para entrenar el modelo.")
        else:
            # 2. Predecir recursivamente
            history_sales = list(df_total['sales'].values)
            history_expenses = list(df_total['expenses'].values)
            
            future_dates = pd.date_range(start=df_total['date'].iloc[-1] + pd.Timedelta(days=7), periods=4, freq='W-MON')
            predictions = []
            
            for date in future_dates:
                features = [
                    history_sales[-1], history_sales[-2], history_sales[-4],
                    history_expenses[-1], history_expenses[-2], history_expenses[-4]
                ]
                
                # Predecir con el modelo nuevo
                pred_s = model_s_live.predict([features])[0]
                pred_e = model_e_live.predict([features])[0]
                
                predictions.append({'Fecha': date, 'Ventas Proyectadas': float(pred_s), 'Gastos Proyectados': float(pred_e)})
                
                history_sales.append(pred_s)
                history_expenses.append(pred_e)
            
            # 3. Mostrar Resultados
            df_pred = pd.DataFrame(predictions)
            
            c1, c2 = st.columns(2)
            with c1:
                st.success("Modelo actualizado correctamente.")
                st.dataframe(df_pred)
                
            with c2:
                # Gráfico de proyección
                st.line_chart(df_pred, x='Fecha', y=['Ventas Proyectadas', 'Gastos Proyectados'])