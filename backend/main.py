import os
import json
import pandas as pd
import geopandas as gpd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

app = FastAPI(title="Plataforma de Monitoreo Forestal - Yucatán")
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
    print("🚀 Levantando Backend de Monitoreo Forestal...")
    
    if RUTA_MALLA.exists() and RUTA_CSV_RIESGO.exists():
        gdf_malla = gpd.read_file(RUTA_MALLA)
        gdf_malla['ID_POLIGONO'] = gdf_malla['ID_POLIGONO'].astype(str)
        
        df_preds = pd.read_csv(RUTA_CSV_RIESGO)
        df_preds['ID_POLIGONO'] = df_preds['ID_POLIGONO'].astype(str)
        
        cache['malla_regional'] = gdf_malla
        cache['datos_regionales'] = df_preds
        
        col = 'FECHA_DIA' if 'FECHA_DIA' in df_preds.columns else 'FECHA_MES'
        cache['col_fecha_riesgo'] = col
        cache['fechas_regionales'] = set(df_preds[col].unique()) if col in df_preds.columns else set()

    if RUTA_POLIGONOS.exists():
        gdf_pol = gpd.read_file(RUTA_POLIGONOS)
        gdf_pol['ID_POLIGONO'] = gdf_pol['id'].astype(str) if 'id' in gdf_pol.columns else gdf_pol.index.astype(str)
        
        archivos = list(RUTA_HISTORICO.glob("*.parquet"))
        if archivos:
            df_hist = pd.concat([pd.read_parquet(f) for f in archivos], ignore_index=True)
            df_hist['ID_POLIGONO'] = df_hist['ID_POLIGONO'].astype(str)
            col_hist = 'FECHA_DIA' if 'FECHA_DIA' in df_hist.columns else 'FECHA_MES'
            cache['col_fecha_hist'] = col_hist
            df_hist[col_hist] = df_hist[col_hist].astype(str)
            
            ids_con_datos = df_hist['ID_POLIGONO'].unique()
            gdf_pol = gdf_pol[gdf_pol['ID_POLIGONO'].isin(ids_con_datos)]
            
            cache['poligonos_control'] = gdf_pol
            cache['historial_control'] = df_hist
            cache['fechas_control'] = set(df_hist[col_hist].unique())

    fechas_totales = cache.get('fechas_regionales', set()) | cache.get('fechas_control', set())
    cache['fechas_disponibles'] = sorted(list(fechas_totales), reverse=True)

@app.get("/api/fechas")
def obtener_linea_tiempo():
    return {"fechas": cache.get('fechas_disponibles', [])}

@app.get("/api/capas/regional")
def obtener_capa_regional(fecha: str = None):
    if 'malla_regional' not in cache: return {"type": "FeatureCollection", "features": []}
    gdf = cache['malla_regional']
    df = cache['datos_regionales']
    col = cache['col_fecha_riesgo']
    
    df_filtrado = df[df[col].astype(str).str.startswith(fecha[:7])] if fecha and col in df.columns else df
    resultado = gdf.merge(df_filtrado, on='ID_POLIGONO', how='left')
    resultado['PROBABILIDAD_COLAPSO'] = resultado['PROBABILIDAD_COLAPSO'].fillna(0).round(1)
    
    return json.loads(resultado.to_json())

@app.get("/api/capas/control")
def obtener_capa_control(fecha: str = None):
    if 'poligonos_control' not in cache: return {"type": "FeatureCollection", "features": []}
    gdf = cache['poligonos_control']
    df = cache['historial_control']
    col = cache.get('col_fecha_hist', 'FECHA_MES')
    
    df_mes = df[df[col].astype(str).str.startswith(fecha[:7])] if fecha else df
    df_audit = df_mes.sort_values(by=['ID_POLIGONO', 'ENSAMBLE_PRED']).groupby('ID_POLIGONO').first().reset_index()
    resultado = gdf.merge(df_audit, on='ID_POLIGONO', how='left')
    
    return json.loads(resultado.to_json())

@app.get("/api/serie_tiempo/{id_poligono}")
def obtener_serie_tiempo(id_poligono: str):
    if 'historial_control' not in cache or 'datos_regionales' not in cache: return []
    
    df_hist = cache['historial_control']
    df_preds = cache['datos_regionales']
    
    hist_pol = df_hist[df_hist['ID_POLIGONO'] == id_poligono].copy()
    pred_pol = df_preds[df_preds['ID_POLIGONO'] == id_poligono].copy()
    
    col_hist = cache.get('col_fecha_hist', 'FECHA_MES')
    col_pred = cache.get('col_fecha_riesgo', 'FECHA_MES')
    
    # Estandarizar al formato YYYY-MM para alinear correctamente los registros satelitales con las predicciones
    if not hist_pol.empty: hist_pol['mes_cruce'] = hist_pol[col_hist].astype(str).str[:7]
    if not pred_pol.empty: pred_pol['mes_cruce'] = pred_pol[col_pred].astype(str).str[:7]
    
    if hist_pol.empty and pred_pol.empty: return []

    if not hist_pol.empty and not pred_pol.empty:
        df_merged = pd.merge(hist_pol, pred_pol, on=['ID_POLIGONO', 'mes_cruce'], how='outer', suffixes=('', '_desc'))
        df_merged = df_merged[[c for c in df_merged.columns if not c.endswith('_desc')]]
    elif not hist_pol.empty:
        df_merged = hist_pol.copy()
        df_merged['PROBABILIDAD_COLAPSO'] = 0
    else:
        df_merged = pred_pol.copy()
    
    df_merged = df_merged.sort_values('mes_cruce').fillna(0)
    
    resultados = []
    cols_numericas = df_merged.select_dtypes(include=['number']).columns
    
    for _, row in df_merged.iterrows():
        dict_row = {"fecha": row['mes_cruce']}
        for col in cols_numericas:
            if col not in ['ID_POLIGONO', 'Unnamed: 0', 'index']:
                dict_row[col] = round(float(row[col]), 3)
        dict_row['NDVI_Graph'] = dict_row.get('NDVI', 0)
        dict_row['Estres_Graph'] = dict_row.get('ESTRES_HIDRICO', 0)
        dict_row['Riesgo_Graph'] = dict_row.get('PROBABILIDAD_COLAPSO', 0)
        resultados.append(dict_row)
        
    return resultados

@app.get("/api/fechas/recargar")
def recargar_linea_tiempo():
    inicializar_motor()
    return {"fechas": cache.get('fechas_disponibles', [])}