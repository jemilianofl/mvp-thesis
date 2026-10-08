import os
import json
import pandas as pd
import geopandas as gpd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

app = FastAPI(title="Motor MLOps - Deforestación Yucatán")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DIR_ACTUAL = Path(__file__).resolve().parent
# Si la carpeta Datos está al lado de main.py (Docker), úsala. Si no, sube un nivel (Local).
DIR_RAIZ = DIR_ACTUAL if (DIR_ACTUAL / "Datos").exists() else DIR_ACTUAL.parent

RUTA_MALLA = DIR_RAIZ / "Datos" / "01_Crudos" / "malla_regional_yucatan.geojson"
RUTA_CSV_LSTM = DIR_RAIZ / "Datos" / "06_Inferencia" / "01_Reporte_Riesgo_Regional.csv"
RUTA_POLIGONOS = DIR_RAIZ / "Datos" / "01_Crudos" / "poligonos_maestros.geojson"
RUTA_ENSAMBLE = DIR_RAIZ / "Datos" / "04_Resultados" / "baseline_ensamble"

cache = {}

@app.on_event("startup")
def inicializar_motor():
    print("🚀 Levantando Sistema de Capas (Backend)...")
    
    if RUTA_MALLA.exists() and RUTA_CSV_LSTM.exists():
        gdf_malla = gpd.read_file(RUTA_MALLA)
        gdf_malla['ID_POLIGONO'] = gdf_malla['ID_POLIGONO'].astype(str)
        
        df_preds = pd.read_csv(RUTA_CSV_LSTM)
        df_preds['ID_POLIGONO'] = df_preds['ID_POLIGONO'].astype(str)
        
        cache['malla_regional'] = gdf_malla
        cache['datos_regionales'] = df_preds
        
        # Leer columna de tiempo dinámica
        col = 'FECHA_DIA' if 'FECHA_DIA' in df_preds.columns else 'FECHA_MES'
        cache['col_fecha_lstm'] = col
        cache['fechas_regionales'] = set(df_preds[col].unique()) if col in df_preds.columns else set()

    if RUTA_POLIGONOS.exists():
        gdf_pol = gpd.read_file(RUTA_POLIGONOS)
        gdf_pol['ID_POLIGONO'] = gdf_pol['id'].astype(str) if 'id' in gdf_pol.columns else gdf_pol.index.astype(str)
        
        archivos = list(RUTA_ENSAMBLE.glob("*.parquet"))
        if archivos:
            df_hist = pd.concat([pd.read_parquet(f) for f in archivos], ignore_index=True)
            df_hist['ID_POLIGONO'] = df_hist['ID_POLIGONO'].astype(str)
            df_hist['FECHA_MES'] = df_hist['FECHA_MES'].astype(str)
            
            # FILTRO CRUCIAL: Eliminar polígonos que nunca tuvieron datos satelitales
            ids_con_datos = df_hist['ID_POLIGONO'].unique()
            gdf_pol = gdf_pol[gdf_pol['ID_POLIGONO'].isin(ids_con_datos)]
            
            cache['poligonos_control'] = gdf_pol
            cache['historial_control'] = df_hist
            cache['fechas_control'] = set(df_hist['FECHA_MES'].unique())

    fechas_totales = cache.get('fechas_regionales', set()) | cache.get('fechas_control', set())
    cache['fechas_disponibles'] = sorted(list(fechas_totales))

@app.get("/api/fechas")
def obtener_linea_tiempo():
    return {"fechas": cache.get('fechas_disponibles', [])}

@app.get("/api/capas/regional")
def obtener_capa_regional(fecha: str = None):
    if 'malla_regional' not in cache: return {"type": "FeatureCollection", "features": []}
    
    gdf = cache['malla_regional']
    df = cache['datos_regionales']
    col = cache['col_fecha_lstm']
    
    if fecha and col in df.columns:
        df_filtrado = df[df[col] == fecha]
        if df_filtrado.empty:
            df_filtrado = df[df[col].str.startswith(fecha[:7])]
    else:
        df_filtrado = df

    resultado = gdf.merge(df_filtrado, on='ID_POLIGONO', how='left')
    resultado['PROBABILIDAD_COLAPSO'] = resultado['PROBABILIDAD_COLAPSO'].fillna(0).round(1)
    resultado['NIVEL_RIESGO'] = resultado['NIVEL_RIESGO'].fillna('Selva Sana')
    
    return json.loads(resultado.to_json())

@app.get("/api/capas/control")
def obtener_capa_control(fecha: str = None):
    if 'poligonos_control' not in cache: return {"type": "FeatureCollection", "features": []}
    
    gdf = cache['poligonos_control']
    df = cache['historial_control']
    
    if fecha:
        # Cruza el "2026-06-01" de LSTM con el "2026-06" del Ensamble
        df_mes = df[df['FECHA_MES'].str.startswith(fecha[:7])]
        df_audit = df_mes.sort_values(by=['ID_POLIGONO', 'ENSAMBLE_PRED']).groupby('ID_POLIGONO').first().reset_index()
    else:
        df_audit = df.sort_values(by=['ID_POLIGONO', 'ENSAMBLE_PRED']).groupby('ID_POLIGONO').first().reset_index()

    resultado = gdf.merge(df_audit, on='ID_POLIGONO', how='left')
    resultado['ENSAMBLE_PRED'] = resultado['ENSAMBLE_PRED'].fillna(-999) 
    
    for v in ['NDVI', 'RADAR_VV', 'RADAR_VH', 'ESTRES_HIDRICO']:
        if v in resultado.columns: 
            resultado[v] = pd.to_numeric(resultado[v], errors='coerce').fillna(0).round(3)
            
    return json.loads(resultado.to_json())