# FinForecast

FinForecast es una suite de análisis financiero predictivo diseñada para la proyección de flujos de caja operativos. Esta herramienta utiliza algoritmos de Regresión Lineal con Forecasting Recursivo para transformar datos financieros históricos en predicciones accionables, permitiendo la simulación de escenarios y la calibración de modelos en tiempo real.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

## Descripción

El proyecto aborda la problemática de la incertidumbre financiera en pequeñas y medianas empresas. A través de una interfaz web interactiva, FinForecast permite unificar fuentes de datos heterogéneas (cuentas bancarias, nóminas, tarjetas de crédito) para modelar el comportamiento futuro de ingresos y egresos.

La solución se diferencia por su capacidad de "Proyección Incremental", permitiendo a los usuarios incorporar cierres semanales recientes de forma manual y re-entrenar el modelo instantáneamente para ajustar las proyecciones a las tendencias de mercado más actuales.

## Características Principales

- **Motor de Predicción Recursiva**
  Genera proyecciones a 4 semanas (Q+1) utilizando una ventana deslizante donde las estimaciones previas alimentan las predicciones futuras.

- **Ingeniería de Características**
  Implementación de variables de retardo (Lags) en t-1, t-2 y t-4 para capturar la inercia y estacionalidad mensual de los flujos financieros.

- **Simulación de Escenarios**
  Interfaz para el registro manual de cierres semanales, permitiendo observar el impacto inmediato de nuevos datos en la tendencia futura sin alterar el dataset original permanentemente.

- **Auditoría de Datos**
  Visualización diferenciada entre datos históricos consolidados y datos de escenarios simulados para mantener la integridad del análisis.

## Tecnologías Utilizadas

- **Lenguaje:** Python
- **Interfaz:** Streamlit
- **Modelado:** Scikit-Learn (Linear Regression)
- **Manipulación de Datos:** Pandas, NumPy
- **Visualización:** Matplotlib

## Estructura del Proyecto

FinForecast_App/

├── app.py                      # Aplicación principal (Interfaz Streamlit)

├── FinForecast.ipynb           # Notebook de análisis y entrenamiento

├── requirements.txt            # Dependencias del entorno

├── checking_account_main.csv   # Dataset: Ingresos y gastos operativos

├── checking_account_secondary.csv # Dataset: Nómina y transferencias

└── credit_card_account.csv     # Dataset: Gastos variables

## Instalación y Despliegue

Para ejecutar este proyecto en un entorno local, siga los siguientes pasos:

1. Clonar el repositorio
   ```bash
   git clone [https://github.com/mckimssongg/FinForecast_App.git](https://github.com/mckimssongg/FinForecast_App.git)
   cd FinForecast_App
   ```


2. Instalar las dependencias Se recomienda utilizar un entorno virtual.
   ```bash
   pip install -r requirements.txt
   ```

   
3. Ejecutar la aplicación
   ```bash
   python -m streamlit run app.py
   ```

## Metodología
El núcleo del sistema se basa en un modelo de Regresión Lineal Supervisado. El proceso de ETL (Extracción, Transformación y Carga) unifica las transacciones diarias en una serie temporal semanal.
Durante la inferencia, el sistema re-entrena el modelo utilizando la totalidad de los datos disponibles (históricos + simulados) para asegurar que la proyección refleje la información más reciente disponible.

Métricas de Validación:

MAE Ventas: ~$410 USD (Margen de error <7%)

MAE Gastos: ~$1,200 USD
