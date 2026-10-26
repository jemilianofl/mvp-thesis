import os
import json
import pandas as pd
import numpy as np
import geopandas as gpd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
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

@app.on_event("startup")
def inicializar_motor():
    print("🚀 Levantando Backend de Monitoreo Forestal (Resolución de 7 Días)...")
    
    if RUTA_MALLA.exists() and RUTA_CSV_RIESGO.exists():
        gdf_malla = gpd.read_file(RUTA_MALLA)
        gdf_malla['ID_POLIGONO'] = gdf_malla['ID_POLIGONO'].astype(str)
        
        df_preds = pd.read_csv(RUTA_CSV_RIESGO)
        df_preds['ID_POLIGONO'] = df_preds['ID_POLIGONO'].astype(str)
        
        col_pred = 'FECHA_DIA' if 'FECHA_DIA' in df_preds.columns else 'FECHA_MES'
        df_preds['FECHA_CRUCE'] = df_preds[col_pred].astype(str)
        
        cache['malla_regional'] = gdf_malla
        cache['datos_regionales'] = df_preds
        cache['fechas_regionales'] = set(df_preds['FECHA_CRUCE'].unique())

    if RUTA_POLIGONOS.exists():
        gdf_pol = gpd.read_file(RUTA_POLIGONOS)
        gdf_pol['ID_POLIGONO'] = gdf_pol['id'].astype(str) if 'id' in gdf_pol.columns else gdf_pol.index.astype(str)
        
        archivos = list(RUTA_HISTORICO.glob("*.parquet"))
        if archivos:
            df_hist = pd.concat([pd.read_parquet(f) for f in archivos], ignore_index=True)
            df_hist['ID_POLIGONO'] = df_hist['ID_POLIGONO'].astype(str)
            
            col_hist = 'FECHA_DIA' if 'FECHA_DIA' in df_hist.columns else 'FECHA_MES'
            df_hist['FECHA_CRUCE'] = df_hist[col_hist].astype(str)
            
            ids_con_datos = df_hist['ID_POLIGONO'].unique()
            gdf_pol = gdf_pol[gdf_pol['ID_POLIGONO'].isin(ids_con_datos)]
            
            cache['poligonos_control'] = gdf_pol
            cache['historial_control'] = df_hist
            cache['fechas_control'] = set(df_hist['FECHA_CRUCE'].unique())

    fechas_totales = cache.get('fechas_regionales', set()) | cache.get('fechas_control', set())
    cache['fechas_disponibles'] = sorted(list(fechas_totales), reverse=True)

def sanear_dataframe(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if col == 'geometry': 
            continue 
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('ESTABLE')
        else:
            df[col] = df[col].fillna(0)
    return df

@app.get("/api/fechas")
def obtener_linea_tiempo():
    return {"fechas": cache.get('fechas_disponibles', [])}

@app.get("/api/capas/regional")
def obtener_capa_regional(fecha: str = None):
    if 'malla_regional' not in cache: return {"type": "FeatureCollection", "features": []}
    gdf = cache['malla_regional'].copy()
    df = cache['datos_regionales']
    
    if fecha:
        df_past = df[df['FECHA_CRUCE'] <= fecha]
        df_filtrado = df_past.sort_values('FECHA_CRUCE').groupby('ID_POLIGONO').last().reset_index()
    else:
        df_filtrado = df.sort_values('FECHA_CRUCE').groupby('ID_POLIGONO').last().reset_index()
        
    resultado = gdf.merge(df_filtrado, on='ID_POLIGONO', how='left')
    
    if 'PROBABILIDAD_COLAPSO' not in resultado.columns:
        resultado['PROBABILIDAD_COLAPSO'] = 0
    if 'NIVEL_RIESGO' not in resultado.columns:
        resultado['NIVEL_RIESGO'] = 'ESTABLE'
        
    resultado = sanear_dataframe(resultado)
    resultado['PROBABILIDAD_COLAPSO'] = resultado['PROBABILIDAD_COLAPSO'].round(1)
    
    return json.loads(resultado.to_json())

@app.get("/api/capas/control")
def obtener_capa_control(fecha: str = None):
    if 'poligonos_control' not in cache: return {"type": "FeatureCollection", "features": []}
    gdf = cache['poligonos_control'].copy()
    df = cache['historial_control']
    
    if fecha:
        df_past = df[df['FECHA_CRUCE'] <= fecha]
        df_audit = df_past.sort_values(by=['ID_POLIGONO', 'ENSAMBLE_PRED']).groupby('ID_POLIGONO').last().reset_index()
    else:
        df_audit = df.sort_values(by=['ID_POLIGONO', 'ENSAMBLE_PRED']).groupby('ID_POLIGONO').last().reset_index()
        
    resultado = gdf.merge(df_audit, on='ID_POLIGONO', how='left')
    
    if 'ENSAMBLE_PRED' not in resultado.columns:
        resultado['ENSAMBLE_PRED'] = 0
        
    resultado = sanear_dataframe(resultado)
    return json.loads(resultado.to_json())

@app.get("/api/serie_tiempo/{id_poligono}")
def obtener_serie_tiempo(id_poligono: str, fecha: str = None):
    """Devuelve exactamente los últimos 7 días desde la fecha seleccionada para evitar líneas planas."""
    if 'historial_control' not in cache or 'datos_regionales' not in cache: return []
    
    df_hist = cache['historial_control']
    df_preds = cache['datos_regionales']
    
    hist_pol = df_hist[df_hist['ID_POLIGONO'] == id_poligono].copy()
    pred_pol = df_preds[df_preds['ID_POLIGONO'] == id_poligono].copy()
    
    if hist_pol.empty and pred_pol.empty: return []

    if not hist_pol.empty and not pred_pol.empty:
        df_merged = pd.merge(hist_pol, pred_pol, on=['ID_POLIGONO', 'FECHA_CRUCE'], how='outer', suffixes=('', '_drop'))
        df_merged = df_merged[[c for c in df_merged.columns if not c.endswith('_drop')]]
    elif not hist_pol.empty:
        df_merged = hist_pol.copy()
        df_merged['PROBABILIDAD_COLAPSO'] = 0
    else:
        df_merged = pred_pol.copy()
    
    df_merged = df_merged.sort_values('FECHA_CRUCE')
    
    # -------------------------------------------------------------
    # RECORTAR A UNA SEMANA Y PROPAGAR LAS 12 VARIABLES
    # -------------------------------------------------------------
    if not df_merged.empty:
        df_merged['FECHA_CRUCE'] = pd.to_datetime(df_merged['FECHA_CRUCE'], errors='coerce')
        df_merged = df_merged.dropna(subset=['FECHA_CRUCE']).set_index('FECHA_CRUCE').sort_index()
        
        # Propagamos datos hacia atrás y adelante para que las 12 variables existan en todo momento
        df_merged = df_merged.ffill().bfill().fillna(0)
        
        fecha_fin = pd.to_datetime(fecha) if fecha else df_merged.index.max()
        fecha_inicio = fecha_fin - pd.Timedelta(days=7) # Ventana estricta de 1 semana
        
        rango_semanal = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
        
        df_merged = df_merged.reindex(rango_semanal)
        df_merged = df_merged.ffill().bfill().fillna(0)
        
        df_merged = df_merged.reset_index().rename(columns={'index': 'FECHA_CRUCE'})
        df_merged['FECHA_CRUCE'] = df_merged['FECHA_CRUCE'].dt.strftime('%Y-%m-%d')
    
    df_merged = sanear_dataframe(df_merged)
    
    resultados = []
    cols_numericas = df_merged.select_dtypes(include=['number']).columns
    
    for _, row in df_merged.iterrows():
        dict_row = {"fecha": row['FECHA_CRUCE']}
        for col in cols_numericas:
            if col not in ['ID_POLIGONO', 'Unnamed: 0', 'index']:
                dict_row[col] = round(float(row[col]), 3)
                
        dict_row['Riesgo_IA'] = dict_row.get('PROBABILIDAD_COLAPSO', 0)
        resultados.append(dict_row)
        
    return resultados

@app.get("/api/fechas/recargar")
def recargar_linea_tiempo():
    inicializar_motor()
    return {"fechas": cache.get('fechas_disponibles', [])}