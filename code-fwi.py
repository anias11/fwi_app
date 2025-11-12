import base64
import geopandas as gpd
import pandas as pd
import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
from functions import *
from cartopy.feature import OCEAN


# ================== CONFIG ==================
st.set_page_config(
    page_title="FWI Dashboard",
    page_icon="./static/logos/TWP-circle-white.svg",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ================== PATHS ===================
logo_path = "./static/logos/TWP-circle-white.svg"
data_path = "./data/ds_forecast.nc"
# csv_path = 

#  ================== ESTILOS ===================
st.markdown(f"""
<style>
/* ===== Tipografía: local (./static) con fallback a Google ===== */
@font-face {{
  font-family: 'PoppinsLocal';
  src: url('./static/Poppins-Regular.woff2') format('woff2'),
       url('./static/Poppins-Regular.ttf') format('truetype');
  font-weight: 300;
  font-style: normal;
  font-display: swap;
}}

/* ===== Fondo y contenedor principal ===== */
[data-testid="stAppViewContainer"] {{
  background: linear-gradient(90deg, #175CA1, #07A9E0 140%);
  background-attachment: fixed;
}}

/* ===== Cabecera (logo + título) ===== */
.header-row {{ display:flex; align-items:center; gap:12px; }}
.header-row h1 {{ margin:0; font-size:4vh; font-weight:500; color:#fff; }}
.header-row img {{ height:5vh; width:auto; }}


/* ===== Ajustes generales ===== */
.block-container label:empty {{ margin:0; padding:0; }}
footer {{ visibility: hidden; }}
section[data-testid="stSidebar"] {{ display:none !important; }}
header[data-testid="stHeader"] {{ display:none !important; }}
MainMenu {{ visibility: hidden; }}
main blockquote, .block-container {{ padding-top: 0.6rem; padding-bottom: 0.6rem; }}

/* ======================================================================================= */
/* ===== A PARTIR D'AQUÍ AFEGIU ELS ESTILS DEL GRAFICS Y ELEMENS QUE ANEU CONSTRUINT ===== */
/* ======================================================================================= */

</style>
""", unsafe_allow_html=True)


# ================== DATA (Aqui va la carga y procesado de datos) ==================
logo_data_uri = img_to_data_uri(logo_path)   #logo

# ========= Dades inventades =========

# ================== CABECERA ==================
st.markdown(
    f"""
    <div class="header-box">
      <div class="header-row">
        <img src="{logo_data_uri}" alt="TDP Logo" />
        <div>
          <h1 style="font-family: 'PoppinsLocal', 'Poppins', sans-serif; font-weight: 700; margin: 0;">Componentes, índices y clasificación de riesgo FWI</h1>
          <p style="font-family: 'PoppinsLocal', 'Poppins', sans-serif; font-size: 2vh; color: #fff; margin: 0; opacity: 0.9;">Estimación del riesgo de incendios calculada con el pronóstico meteorológico GFS del 31 de julio del 2025</p>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ================== TABLERO ==================

# --- Obrim el dataset ---
ds = xr.open_dataset(data_path)

# --- Variables disponibles ---
variables = list(ds.data_vars)

# --- Estilo para los selectbox con Poppins ---
st.markdown("""
<style>
/* Aplicar Poppins a todos los selectbox */
div[data-baseweb="select"] > div,
div[data-baseweb="select"] span,
div[data-baseweb="select"] input,
.stSelectbox label,
.stSelectbox div {
    font-family: 'PoppinsLocal', 'Poppins', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# --- Coord temporal ---
if "time" in ds.coords:
    times = ds["time"].values
    # Convertim a string per mostrar al selector
    time_labels = [str(np.datetime_as_string(t, unit='D')) for t in times]
    selected_time_label = st.selectbox("Selecciona fecha:", time_labels)
    # Index de temps
    time_index = time_labels.index(selected_time_label)
    ds_sel = ds.isel(time=time_index)
else:
    ds_sel = ds
    time_index = None
    selected_time_label = None

# --- Diccionaris per noms i colormaps ---
variable_display_names = {
    't2m': 'Temperatura',
    'rh': 'Humedad',
    'wind10m': 'Velocidad del Viento',
    'rain_24h': 'Precipitación',
    'FWI_risk': 'Riesgo de Incendio',
    'FWI_anomalies': 'Anomalías del FWI',
    'FFMC': 'FFMC',
    'DMC': 'DMC',
    'DC': 'DC',
    'ISI': 'ISI',
    'BUI': 'BUI',
    'FWI': 'FWI',
}

variable_cmaps = {
    't2m': 'coolwarm',
    'rh': 'PuBuGn',
    'wind10m': 'viridis',
    'rain_24h': 'Blues',
    'FWI': 'hot_r',
    'FFMC': 'plasma',
    'DMC': 'cividis',
    'DC': 'YlOrBr',
    'ISI': 'OrRd',
    'BUI': 'YlOrBr',
    'FWI_anomalies': None,
    'FWI_risk': None  # mapa discret
}

# --- Noms amables per al selector ---
display_variables = [variable_display_names.get(v, v) for v in variables]
display_to_original = {display_variables[i]: variables[i] for i in range(len(variables))}

# --- Selector de variable ---
default_label = 'Riesgo de Incendio' if 'FWI_risk' in variables else display_variables[0]
selected_label = st.selectbox(
    "Selecciona variable:",
    display_variables,
    index=display_variables.index(default_label)
)
original_variable = display_to_original[selected_label]

# --- Títol dinàmic ---
if selected_time_label:
    title = f"{variable_display_names.get(original_variable, original_variable)} — {selected_time_label}"
else:
    title = variable_display_names.get(original_variable, original_variable)

# --- Cridar la funció de ploteig ---
fig = plot_variable_cartopy(
    ds_sel,                         # dataset ja filtrat pel temps seleccionat
    original_variable,
    variable_cmaps,
    variable_display_names,
    title=title
)
# --- Aplicar máscara de océano ---
# Obtener el eje de la figura
ax = fig.gca()

# Aplicar máscara de océano
ax.add_feature(OCEAN, zorder=10, facecolor='white', edgecolor='none', alpha=1.00)

st.pyplot(fig, bbox_inches='tight')
