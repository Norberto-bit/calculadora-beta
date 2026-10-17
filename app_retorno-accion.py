import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Configuración de la página web
st.set_page_config(
    page_title="Calculadora de Rentabilidad y Beta - NR Advisory",
    page_icon="📈",
    layout="wide",
)

# Estilo visual moderno y limpio
st.title("📊 Análisis de Rentabilidad y Beta de Mercado")
st.markdown(
    "Herramienta académica y profesional para evaluar el rendimiento de una acción frente al S&P 500."
)

# --- PANEL LATERAL (CONTROLES) ---
st.sidebar.header("⚙️ Parámetros de Análisis")

# Selección de Tickers
ticker_accion = st.sidebar.text_input(
    "Ticker de la Acción (Ej: AAPL, MELI, GGAL)", value="AAPL"
).upper()
ticker_mercado = st.sidebar.text_input(
    "Ticker de Referencia (Benchmark)", value="SPY"
).upper()

# Rango de fechas
col_f1, col_f2 = st.sidebar.columns(2)
with col_f1:
    fecha_inicio = st.date_input(
        "Fecha de Inicio", datetime.date(2023, 1, 1)
    )
with col_f2:
    fecha_fin = st.date_input("Fecha de Fin", datetime.date.today())

# Tasa libre de riesgo estimada para el costo de capital / CAPM
tasa_libre_riesgo = (
    st.sidebar.number_input(
        "Tasa Libre de Riesgo Anual (%)", value=4.0, step=0.25
    )
    / 100
)


# Función para descargar y procesar datos de forma segura
@st.cache_data
def cargar_datos(acc, mkt, inicio, fin):
  try:
    # Descargar precios de cierre ajustados
    datos = yf.download([acc, mkt], start=inicio, end=fin, progress=False)[
        "Close"
    ]
    if isinstance(datos, pd.Series):
      return None, "No se pudieron descargar suficientes datos para los símbolos."
    datos = datos.dropna()
    return datos, None
  except Exception as e:
    return None, str(e)


# Ejecutar descarga
datos_crudos, error = cargar_datos(
    ticker_accion, ticker_mercado, fecha_inicio, fecha_fin
)

if error or datos_crudos is None or datos_crudos.empty:
  st.error(
     f"Ocurrió un error al obtener los datos. Verifica los símbolos ingresados ({ticker_accion}, {ticker_mercado}). Detalle: {error}"
  )
else:
  # Asegurar orden de columnas
  if ticker_accion in datos_crudos.columns and ticker_mercado in datos_crudos.columns:
    df = datos_crudos[[ticker_accion, ticker_mercado]].copy()
    df.columns = ["Accion", "Mercado"]

    # Calcular retornos logarítmicos o diarios porcentuales
    df["Ret_Accion"] = df["Accion"].pct_change()
    df["Ret_Mercado"] = df["Mercado"].pct_change()
    df = df.dropna()

    # --- CÁLCULOS CLAVE (Replicando la planilla) ---
    # 1. Rentabilidad acumulada en el período
    rentabilidad_acum_accion = (
        df["Accion"].iloc[-1] / df["Accion"].iloc[0]
    ) - 1
    rentabilidad_acum_mercado = (
        df["Mercado"].iloc[-1] / df["Mercado"].iloc[0]
    ) - 1

    # 2. Cálculo de la Beta (Covarianza / Varianza del mercado)
    cov_mat = np.cov(df["Ret_Accion"], df["Ret_Mercado"])
    beta = cov_mat[0, 1] / cov_mat[1, 1]

    # 3. Volatilidad anualizada (Desvío estándar * sqrt(252 ruedas))
    vol_accion = df["Ret_Accion"].std() * np.sqrt(252)
    vol_mercado = df["Ret_Mercado"].std() * np.sqrt(252)

    # 4. CAPM (Cost of Equity esperado)
    # Retorno anualizado aproximado del mercado
    dias = (df.index[-1] - df.index[0]).days
    anos = dias / 365.25 if dias > 0 else 1
    ret_anual_mercado = (
        (1 + rentabilidad_acum_mercado) ** (1 / anos)
    ) - 1 if anos > 0 else 0
    costo_capital_capm = tasa_libre_riesgo + beta * (
        ret_anual_mercado - tasa_libre_riesgo
    )

    # --- VISUALIZACIÓN EN PANTALLA (KPIs) ---
    st.markdown("### 📈 Resumen Ejecutivo de Desempeño")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
      st.metric(
          label=f"Rentabilidad Acum. ({ticker_accion})",
          value=f"{rentabilidad_acum_accion:.2%}",
      )
    with kpi2:
      st.metric(
          label=f"Rentabilidad Acum. ({ticker_mercado})",
          value=f"{rentabilidad_acum_mercado:.2%}",
      )
    with kpi3:
      st.metric(label="Coeficiente Beta ($\beta$)", value=f"{beta:.2f}")
    with kpi4:
      st.metric(
          label="Costo de Capital (CAPM est.)",
          value=f"{costo_capital_capm:.2%}",
      )

    st.markdown("---")

    # --- PESTAÑAS DE GRÁFICOS Y DETALLE ---
    tab1, tab2, tab3 = st.tabs([
        "📉 Evolución Normalizada (Base 100)",
        "📐 Análisis de Regresión y Beta",
        "📋 Datos Históricos",
    ])

    with tab1:
      st.subheader("Comparativa de Precios Normalizados (Base 100 al inicio)")
      df_normalizado = (df[["Accion", "Mercado"]] / df[["Accion", "Mercado"]].iloc[0]) * 100
      df_normalizado.columns = [ticker_accion, ticker_mercado]
      
      fig_line = px.line(
          df_normalizado,
          labels={"value": "Precio Normalizado (Base 100)", "index": "Fecha"},
          title=f"Desempeño Relativo: {ticker_accion} vs {ticker_mercado}",
      )
      fig_line.update_layout(hovermode="x unified")
      st.plotly_chart(fig_line, use_container_width=True)

    with tab2:
      st.subheader("Dispersión de Retornos Diarios (Estimación de Beta)")
      st.markdown(
          "Cada punto representa el retorno diario de la acción frente al mercado en una rueda bursátil."
      )
      
      fig_scatter = px.scatter(
          df,
          x="Ret_Mercado",
          y="Ret_Accion",
          opacity=0.6,
          labels={
              "Ret_Mercado": f"Retorno Diario Mercado ({ticker_mercado})",
              "Ret_Accion": f"Retorno Diario Acción ({ticker_accion})",
          },
          title=f"Regresión Lineal — Beta: {beta:.2f}",
      )
      
      # Línea de tendencia (Beta)
      m, b = np.polyfit(df["Ret_Mercado"], df["Ret_Accion"], 1)
      df["Tendencia"] = m * df["Ret_Mercado"] + b
      
      fig_scatter.add_traces(
          go.Scatter(
              x=df["Ret_Mercado"],
              y=df["Tendencia"],
              mode="lines",
              name=f"Línea Beta ({beta:.2f})",
              line=dict(color="red", width=2),
          )
      )
      st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
      st.subheader("Tabla de Datos Diarios Procesados")
      st.dataframe(df.tail(100), use_container_width=True)
