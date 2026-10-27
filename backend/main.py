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
ESTANDARES_SOM = ['NDVI', 'BSI', 'GCI', 'RADAR_VH', 'RADAR_VV', 'TMAX_MEDIAN', 'PRECIP_SUM_MES', 'PRECIP_ACUM_3M', 'ESTRES_HIDRICO', 'ELEVACION', 'NDVI_WAVELET_DETALLE', 'PRECIP_WAVELET_DETALLE']

@app.on_event("startup")
def inicializar_motor():
    print("🚀 Levantando Backend de Inferencia Diaria...")
    _cargar_datos_regionales()
    _cargar_datos_control()
    _consolidar_fechas()

def _cargar_datos_regionales():
    if RUTA_MALLA.exists() and RUTA_CSV_RIESGO.exists():
        gdf_malla = gpd.read_file(RUTA_MALLA)
        gdf_malla['ID_POLIGONO'] = gdf_malla['ID_POLIGONO'].astype(str)
        
        df_preds = pd.read_csv(RUTA_CSV_RIESGO)
        df_preds['ID_POLIGONO'] = df_preds['ID_POLIGONO'].astype(str)
        col_pred = 'FECHA_DIA' if 'FECHA_DIA' in df_preds.columns else 'FECHA_MES'
        df_preds['FECHA_CRUCE'] = df_preds[col_pred].astype(str)
        
        cache['malla_regional'] = gdf_malla
        cache['datos_regionales'] = df_preds

def _cargar_datos_control():
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
            cache['poligonos_control'] = gdf_pol[gdf_pol['ID_POLIGONO'].isin(ids_con_datos)]
            cache['historial_control'] = df_hist

def _consolidar_fechas():
    fechas_totales = set()
    if 'datos_regionales' in cache:
        fechas_totales.update(cache['datos_regionales']['FECHA_CRUCE'].unique())
    if 'historial_control' in cache:
        fechas_totales.update(cache['historial_control']['FECHA_CRUCE'].unique())
    
    if fechas_totales:
        fecha_min = sorted(list(fechas_totales))[0]
        fecha_hoy = datetime.now().strftime('%Y-%m-%d')
        rango_calendario = pd.date_range(start=fecha_min, end=fecha_hoy, freq='D')
        cache['fechas_disponibles'] = rango_calendario.strftime('%Y-%m-%d').tolist()[::-1]
        
        df_preds_global = cache.get('datos_regionales', pd.DataFrame())
        if not df_preds_global.empty and 'PROBABILIDAD_COLAPSO' in df_preds_global.columns:
            alertas = df_preds_global[df_preds_global['PROBABILIDAD_COLAPSO'] >= 50]
            cache['fechas_alerta'] = set(alertas['FECHA_CRUCE'].unique())
        else:
            cache['fechas_alerta'] = set()
    else:
        cache['fechas_disponibles'] = []
        cache['fechas_alerta'] = set()

def sanear_dataframe(df: pd.DataFrame) -> pd.DataFrame:
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
    
    df_filtrado = df[df['FECHA_CRUCE'] <= fecha].sort_values('FECHA_CRUCE').groupby('ID_POLIGONO').last().reset_index() if fecha else df.sort_values('FECHA_CRUCE').groupby('ID_POLIGONO').last().reset_index()
    resultado = gdf.merge(df_filtrado, on='ID_POLIGONO', how='left')
    
    if 'PROBABILIDAD_COLAPSO' not in resultado.columns: resultado['PROBABILIDAD_COLAPSO'] = 0
    if 'NIVEL_RIESGO' not in resultado.columns: resultado['NIVEL_RIESGO'] = 'ESTABLE'
        
    resultado = sanear_dataframe(resultado)
    resultado['PROBABILIDAD_COLAPSO'] = resultado['PROBABILIDAD_COLAPSO'].round(1)
    return json.loads(resultado.to_json())

@app.get("/api/capas/control")
def obtener_capa_control(fecha: str = None):
    if 'poligonos_control' not in cache: return {"type": "FeatureCollection", "features": []}
    gdf = cache['poligonos_control'].copy()
    df = cache['historial_control']
    
    df_filtrado = df[df['FECHA_CRUCE'] <= fecha].sort_values('FECHA_CRUCE').groupby('ID_POLIGONO').last().reset_index() if fecha else df.sort_values('FECHA_CRUCE').groupby('ID_POLIGONO').last().reset_index()
    resultado = gdf.merge(df_filtrado, on='ID_POLIGONO', how='left')
    
    if 'ENSAMBLE_PRED' not in resultado.columns: resultado['ENSAMBLE_PRED'] = 0
    resultado = sanear_dataframe(resultado)
    return json.loads(resultado.to_json())

@app.get("/api/serie_tiempo/{id_poligono}")
def obtener_serie_tiempo(id_poligono: str, fecha: str = None):
    df_preds = cache.get('datos_regionales', pd.DataFrame())
    df_hist = cache.get('historial_control', pd.DataFrame())
    
    hist_pol = df_hist[df_hist['ID_POLIGONO'] == id_poligono].copy() if not df_hist.empty else pd.DataFrame()
    pred_pol = df_preds[df_preds['ID_POLIGONO'] == id_poligono].copy() if not df_preds.empty else pd.DataFrame()
    
    fecha_fin_dt = pd.to_datetime(fecha) if fecha else datetime.now()
    fecha_inicio_dt = fecha_fin_dt - pd.Timedelta(days=7)
    rango_semanal = pd.date_range(start=fecha_inicio_dt, end=fecha_fin_dt, freq='D')
    
    # FIX TOTAL: Construimos un DataFrame base garantizado de 7 días. Así NUNCA devolvemos un array vacío al frontend.
    df_base = pd.DataFrame({'FECHA_CRUCE': rango_semanal})
    
    if not pred_pol.empty:
        cols_ia = [c for c in ['ID_POLIGONO', 'FECHA_CRUCE', 'PROBABILIDAD_COLAPSO', 'NIVEL_RIESGO'] + ESTANDARES_SOM if c in pred_pol.columns]
        pred_pol = pred_pol[cols_ia]

    if not hist_pol.empty and not pred_pol.empty:
        df_merged = pd.merge(hist_pol, pred_pol, on=['ID_POLIGONO', 'FECHA_CRUCE'], how='outer')
    elif not pred_pol.empty:
        df_merged = pred_pol.copy()
    elif not hist_pol.empty:
        df_merged = hist_pol.copy()
    else:
        df_merged = pd.DataFrame()
        
    if not df_merged.empty:
        df_merged['FECHA_CRUCE'] = pd.to_datetime(df_merged['FECHA_CRUCE'], errors='coerce')
        df_merged = df_merged.dropna(subset=['FECHA_CRUCE']).set_index('FECHA_CRUCE').sort_index()
        
        fecha_arranque_segura = min(df_merged.index.min(), fecha_inicio_dt)
        rango_completo = pd.date_range(start=fecha_arranque_segura, end=fecha_fin_dt, freq='D')
        
        df_merged = df_merged.reindex(rango_completo).ffill().bfill().fillna(0)
        df_merged = df_merged.loc[fecha_inicio_dt:fecha_fin_dt].reset_index().rename(columns={'index': 'FECHA_CRUCE'})
    else:
        # Si el polígono es un punto ciego total, devolvemos el esqueleto limpio
        df_merged = df_base.copy()
        
    # Asegurar que todas las métricas existan
    if 'PROBABILIDAD_COLAPSO' not in df_merged.columns: df_merged['PROBABILIDAD_COLAPSO'] = 0.0
    if 'NIVEL_RIESGO' not in df_merged.columns: df_merged['NIVEL_RIESGO'] = 'ESTABLE'
    if 'ENSAMBLE_PRED' not in df_merged.columns: df_merged['ENSAMBLE_PRED'] = 0
    for col in ESTANDARES_SOM:
        if col not in df_merged.columns: df_merged[col] = 0.0
        
    df_merged['FECHA_CRUCE'] = pd.to_datetime(df_merged['FECHA_CRUCE']).dt.strftime('%Y-%m-%d')
    df_merged = sanear_dataframe(df_merged)
    
    resultados = []
    cols_relevantes = ESTANDARES_SOM + ['PROBABILIDAD_COLAPSO', 'ENSAMBLE_PRED']
    
    for _, row in df_merged.iterrows():
        dict_row = {"fecha": row['FECHA_CRUCE']}
        for col in cols_relevantes:
            if col in df_merged.columns:
                dict_row[col] = round(float(row[col]), 4)
                
        dict_row['Riesgo_IA'] = round(float(row.get('PROBABILIDAD_COLAPSO', 0)), 1)
        nr = str(row.get('NIVEL_RIESGO', 'ESTABLE')).strip().upper()
        dict_row['NIVEL_RIESGO'] = 'ESTABLE' if nr == 'NAN' else nr
        resultados.append(dict_row)
        
    return resultados

@app.get("/api/fechas/recargar")
def recargar_linea_tiempo():
    inicializar_motor()
    return {"fechas": cache.get('fechas_disponibles', [])}