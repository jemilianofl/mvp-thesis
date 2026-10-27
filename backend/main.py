import os
import json
import pandas as pd
import numpy as np
import geopandas as gpd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

app = FastAPI(title="Sistema de Alerta Temprana de Degradación de Suelos")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DIR_ACTUAL = Path(__file__).resolve().parent
DIR_RAIZ = DIR_ACTUAL if (DIR_ACTUAL / "Datos").exists() else DIR_ACTUAL.parent

RUTA_MALLA = DIR_RAIZ / "Datos" / "01_Crudos" / "malla_regional_yucatan.geojson"
RUTA_CSV_RIESGO = DIR_RAIZ / "Datos" / "06_Inferencia" / "01_Reporte_Riesgo_Regional.csv"
RUTA_POLIGONOS = DIR_RAIZ / "Datos" / "01_Crudos" / "poligonos_maestros.geojson"
RUTA_HISTORICO = DIR_RAIZ / "Datos" / "04_Resultados" / "baseline_ensamble"

cache = {}

# Las 12 variables físicas exactas que necesita tu matriz SOM
ESTANDARES_SOM = [
    'NDVI', 'BSI', 'GCI', 'RADAR_VH', 'RADAR_VV', 
    'TMAX_MEDIAN', 'PRECIP_SUM_MES', 'PRECIP_ACUM_3M', 
    'ESTRES_HIDRICO', 'ELEVACION', 'NDVI_WAVELET_DETALLE', 'PRECIP_WAVELET_DETALLE'
]

@app.on_event("startup")
def inicializar_motor():
    print("🚀 Levantando Backend de Monitoreo Forestal (Despliegue de Datos Crudos)...")
    
    if RUTA_MALLA.exists() and RUTA_CSV_RIESGO.exists():
        gdf_malla = gpd.read_file(RUTA_MALLA)
        gdf_malla['ID_POLIGONO'] = gdf_malla['ID_POLIGONO'].astype(str)
        
        df_preds = pd.read_csv(RUTA_CSV_RIESGO)
        df_preds['ID_POLIGONO'] = df_preds['ID_POLIGONO'].astype(str)
        col_pred = 'FECHA_DIA' if 'FECHA_DIA' in df_preds.columns else 'FECHA_MES'
        df_preds['FECHA_CRUCE'] = df_preds[col_pred].astype(str)
        
        cache['malla_regional'] = gdf_malla
        cache['datos_regionales'] = df_preds

    if RUTA_POLIGONOS.exists():
        gdf_pol = gpd.read_file(RUTA_POLIGONOS)
        gdf_pol['ID_POLIGONO'] = gdf_pol['id'].astype(str) if 'id' in gdf_pol.columns else gdf_pol.index.astype(str)
        
        archivos = list(RUTA_HISTORICO.glob("*.parquet"))
        if archivos:
            df_hist = pd.concat([pd.read_parquet(f) for f in archivos], ignore_index=True)
            df_hist['ID_POLIGONO'] = df_hist['ID_POLIGONO'].astype(str)
            cache['poligonos_control'] = gdf_pol
            cache['historial_control'] = df_hist

    # Forzar el calendario hasta el día actual para evitar bloqueos
    fechas_totales = set()
    if 'datos_regionales' in cache:
        fechas_totales.update(cache['datos_regionales']['FECHA_CRUCE'].unique())
    
    if fechas_totales:
        fecha_min = sorted(list(fechas_totales))[0]
        fecha_hoy = datetime.now().strftime('%Y-%m-%d')
        rango_calendario = pd.date_range(start=fecha_min, end=fecha_hoy, freq='D')
        cache['fechas_disponibles'] = rango_calendario.strftime('%Y-%m-%d').tolist()[::-1]
        
        df_preds_global = cache.get('datos_regionales', pd.DataFrame())
        if not df_preds_global.empty and 'PROBABILIDAD_COLAPSO' in df_preds_global.columns:
            alertas = df_preds_global[df_preds_global['PROBABILIDAD_COLAPSO'] >= 80]
            cache['fechas_alerta'] = set(alertas['FECHA_CRUCE'].unique())
    else:
        cache['fechas_disponibles'] = []
        cache['fechas_alerta'] = set()

def sanear_dataframe(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if col == 'geometry': continue 
        if df[col].dtype == 'object': df[col] = df[col].fillna('ESTABLE')
        else: df[col] = df[col].fillna(0)
    return df

@app.get("/api/fechas")
def obtener_linea_tiempo():
    fechas_disp = cache.get('fechas_disponibles', [])
    fechas_alerta = cache.get('fechas_alerta', set())
    return {"fechas": [{"fecha": f, "alerta": f in fechas_alerta} for f in fechas_disp]}

@app.get("/api/capas/regional")
def obtener_capa_regional(fecha: str = None):
    if 'malla_regional' not in cache: return {"type": "FeatureCollection", "features": []}
    gdf = cache['malla_regional'].copy()
    df = cache['datos_regionales']
    
    if df.empty: return json.loads(gdf.to_json())
    
    if fecha:
        df_past = df[df['FECHA_CRUCE'] <= fecha]
        df_filtrado = df_past.sort_values('FECHA_CRUCE').groupby('ID_POLIGONO').last().reset_index()
    else:
        df_filtrado = df.sort_values('FECHA_CRUCE').groupby('ID_POLIGONO').last().reset_index()
        
    resultado = gdf.merge(df_filtrado, on='ID_POLIGONO', how='left')
    if 'PROBABILIDAD_COLAPSO' not in resultado.columns: resultado['PROBABILIDAD_COLAPSO'] = 0
    resultado['PROBABILIDAD_COLAPSO'] = resultado['PROBABILIDAD_COLAPSO'].fillna(0).round(1)
    
    return json.loads(resultado.to_json())

@app.get("/api/capas/control")
def obtener_capa_control(fecha: str = None):
    if 'poligonos_control' not in cache: return {"type": "FeatureCollection", "features": []}
    gdf = cache['poligonos_control'].copy()
    df = cache['historial_control']
    
    if not df.empty and 'ENSAMBLE_PRED' in df.columns:
        df_audit = df.groupby('ID_POLIGONO').last().reset_index()
        resultado = gdf.merge(df_audit, on='ID_POLIGONO', how='left')
    else:
        resultado = gdf.copy()
        resultado['ENSAMBLE_PRED'] = 0
        
    resultado['ENSAMBLE_PRED'] = resultado['ENSAMBLE_PRED'].fillna(0)
    return json.loads(resultado.to_json())

@app.get("/api/serie_tiempo/{id_poligono}")
def obtener_serie_tiempo(id_poligono: str, fecha: str = None):
    df_preds = cache.get('datos_regionales', pd.DataFrame())
    df_hist = cache.get('historial_control', pd.DataFrame())
    
    # 1. Definir la ventana de los últimos 7 días
    fecha_fin_str = fecha if fecha else datetime.now().strftime('%Y-%m-%d')
    fecha_fin = pd.to_datetime(fecha_fin_str)
    fecha_inicio = fecha_fin - pd.Timedelta(days=7)
    rango_semanal = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
    
    # Base diaria obligatoria (Evita las gráficas invisibles)
    df_merged = pd.DataFrame({'FECHA_CRUCE_DT': rango_semanal})
    df_merged['FECHA_CRUCE_STR'] = df_merged['FECHA_CRUCE_DT'].dt.strftime('%Y-%m-%d')
    
    # 2. Rescatar valores CRUDOS (Sin escalar) del histórico Parquet
    pol_hist = df_hist[df_hist['ID_POLIGONO'] == id_poligono] if not df_hist.empty else pd.DataFrame()
    
    if not pol_hist.empty:
        # Tomar la última foto del bosque y arrastrarla toda la semana
        col_temporal = 'FECHA_CRUCE' if 'FECHA_CRUCE' in pol_hist.columns else pol_hist.columns[0]
        last_raw = pol_hist.sort_values(col_temporal).iloc[-1]
        
        for col in ESTANDARES_SOM:
            val = 0.0
            if col in last_raw: val = last_raw[col]
            elif col == 'NDVI' and 'NDVI_ACTUAL' in last_raw: val = last_raw['NDVI_ACTUAL']
            elif col == 'ESTRES_HIDRICO' and 'ESTRES' in last_raw: val = last_raw['ESTRES']
            
            df_merged[col] = float(val) if not pd.isna(val) else 0.0
    else:
        for col in ESTANDARES_SOM: df_merged[col] = 0.0
    
    # 3. Inyectar el riesgo de la IA del día exacto
    pol_preds = df_preds[df_preds['ID_POLIGONO'] == id_poligono] if not df_preds.empty else pd.DataFrame()
    
    if not pol_preds.empty:
        pol_preds['FECHA_CRUCE_DT'] = pd.to_datetime(pol_preds['FECHA_CRUCE'], errors='coerce')
        df_merged = pd.merge(df_merged, pol_preds[['FECHA_CRUCE_DT', 'PROBABILIDAD_COLAPSO', 'NIVEL_RIESGO']], on='FECHA_CRUCE_DT', how='left')
        df_merged['PROBABILIDAD_COLAPSO'] = df_merged['PROBABILIDAD_COLAPSO'].ffill().fillna(0.0)
    else:
        df_merged['PROBABILIDAD_COLAPSO'] = 0.0
        df_merged['NIVEL_RIESGO'] = 'ESTABLE'
        
    resultados = []
    for _, row in df_merged.iterrows():
        dict_row = {"fecha": row['FECHA_CRUCE_STR']}
        for col in ESTANDARES_SOM:
            # Ahora sí, el valor crudo puro se envía a la interfaz
            dict_row[col] = round(float(row[col]), 3)
            
        dict_row['Riesgo_IA'] = round(float(row['PROBABILIDAD_COLAPSO']), 1)
        dict_row['NIVEL_RIESGO'] = str(row['NIVEL_RIESGO']) if not pd.isna(row.get('NIVEL_RIESGO')) else 'ESTABLE'
        resultados.append(dict_row)
        
    return resultados

@app.get("/api/fechas/recargar")
def recargar_linea_tiempo():
    inicializar_motor()
    return {"fechas": cache.get('fechas_disponibles', [])}