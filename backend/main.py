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

cache = {}
ESTANDARES_SOM = ['NDVI', 'BSI', 'GCI', 'RADAR_VH', 'RADAR_VV', 'TMAX_MEDIAN', 'PRECIP_SUM_MES', 'PRECIP_ACUM_3M', 'ESTRES_HIDRICO', 'ELEVACION', 'NDVI_WAVELET_DETALLE', 'PRECIP_WAVELET_DETALLE']

@app.on_event("startup")
def inicializar_motor():
    print("🚀 Levantando Backend de Inferencia Diaria (Datos Seguros)...")
    if RUTA_MALLA.exists() and RUTA_CSV_RIESGO.exists():
        gdf_malla = gpd.read_file(RUTA_MALLA)
        # Limpieza robusta de IDs para evitar desajustes como "4899" vs "4899.0"
        gdf_malla['ID_POLIGONO'] = gdf_malla['ID_POLIGONO'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        
        df_preds = pd.read_csv(RUTA_CSV_RIESGO)
        df_preds['ID_POLIGONO'] = df_preds['ID_POLIGONO'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        
        # FIX DEFINITIVO: Forzar el uso exclusivo de FECHA_DIA para la línea de tiempo del Frontend
        if 'FECHA_DIA' in df_preds.columns:
            df_preds['FECHA_CRUCE'] = df_preds['FECHA_DIA'].astype(str)
        else:
            # Plan de contingencia extremo si algo falla en el ensamble
            col_pred = 'FECHA_MES' if 'FECHA_MES' in df_preds.columns else df_preds.columns[1]
            df_preds['FECHA_CRUCE'] = df_preds[col_pred].astype(str)
            
        col_pred = 'FECHA_DIA' if 'FECHA_DIA' in df_preds.columns else 'FECHA_MES'
        df_preds['FECHA_CRUCE'] = df_preds[col_pred].astype(str)
        
        # Depuración visual en consola
        print(f"Columnas cargadas en el reporte: {df_preds.columns.tolist()}")
        
        cache['malla_regional'] = gdf_malla
        cache['datos_regionales'] = df_preds

    fechas_totales = set()
    if 'datos_regionales' in cache:
        fechas_totales.update(cache['datos_regionales']['FECHA_CRUCE'].unique())
        
    if fechas_totales:
        fechas_ordenadas = sorted(list(fechas_totales))
        rango_calendario = pd.date_range(start=fechas_ordenadas[0], end=fechas_ordenadas[-1], freq='D')
        cache['fechas_disponibles'] = rango_calendario.strftime('%Y-%m-%d').tolist()[::-1]
        
        df_preds = cache['datos_regionales']
        alertas = df_preds[df_preds['PROBABILIDAD_COLAPSO'] >= 50]
        cache['fechas_alerta'] = set(alertas['FECHA_CRUCE'].unique())
    else:
        cache['fechas_disponibles'] = []
        cache['fechas_alerta'] = set()

def sanear_dataframe(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if col == 'geometry': continue 
        if df[col].dtype == 'object': df[col] = df[col].fillna('SIN DATOS')
        else: df[col] = df[col].fillna(0.0)
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
    df = cache.get('datos_regionales', pd.DataFrame())
    
    if not df.empty:
        df_filtrado = df[df['FECHA_CRUCE'] <= fecha].sort_values('FECHA_CRUCE').groupby('ID_POLIGONO').last().reset_index() if fecha else df.sort_values('FECHA_CRUCE').groupby('ID_POLIGONO').last().reset_index()
        resultado = gdf.merge(df_filtrado, on='ID_POLIGONO', how='left')
    else:
        resultado = gdf.copy()
            
    if 'PROBABILIDAD_COLAPSO' not in resultado.columns: resultado['PROBABILIDAD_COLAPSO'] = 0.0
    if 'NIVEL_RIESGO' not in resultado.columns: resultado['NIVEL_RIESGO'] = 'SIN DATOS'
        
    resultado = sanear_dataframe(resultado)
    return json.loads(resultado.to_json())

@app.get("/api/capas/control")
def obtener_capa_control(fecha: str = None):
    return {"type": "FeatureCollection", "features": []}

@app.get("/api/serie_tiempo/{id_poligono}")
def obtener_serie_tiempo(id_poligono: str, fecha: str = None):
    df_preds = cache.get('datos_regionales', pd.DataFrame())
    pred_pol = df_preds[df_preds['ID_POLIGONO'] == id_poligono].copy() if not df_preds.empty else pd.DataFrame()
    
    fecha_fin = pd.to_datetime(fecha) if fecha else datetime.now()
    fecha_inicio = fecha_fin - pd.Timedelta(days=7)
    rango_semanal = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
    df_base = pd.DataFrame({'FECHA_CRUCE': rango_semanal})
    
    # Si hacemos clic en un polígono gris, lo declaramos abiertamente sin datos en lugar de quebrar
    if pred_pol.empty:
        df_merged = df_base.copy()
        df_merged['PROBABILIDAD_COLAPSO'] = 0.0
        df_merged['NIVEL_RIESGO'] = 'SIN DATOS'
    else:
        df_merged = pred_pol.copy()
        df_merged['FECHA_CRUCE'] = pd.to_datetime(df_merged['FECHA_CRUCE'], errors='coerce')
        df_merged = df_merged.dropna(subset=['FECHA_CRUCE']).set_index('FECHA_CRUCE').sort_index()
        
        fecha_arranque = min(df_merged.index.min(), fecha_inicio)
        rango_completo = pd.date_range(start=fecha_arranque, end=fecha_fin, freq='D')
        
        df_merged = df_merged.reindex(rango_completo).ffill().bfill().fillna(0.0)
        df_merged = df_merged.loc[fecha_inicio:fecha_fin].reset_index().rename(columns={'index': 'FECHA_CRUCE'})
        
    df_merged['FECHA_CRUCE'] = pd.to_datetime(df_merged['FECHA_CRUCE']).dt.strftime('%Y-%m-%d')
    
    resultados = []
    for _, row in df_merged.iterrows():
        dict_row = {"fecha": row['FECHA_CRUCE']}
        for col in ESTANDARES_SOM:
            valor = row[col] if col in df_merged.columns else 0.0
            dict_row[col] = round(float(valor), 3) if pd.notna(valor) else 0.0
                
        riesgo_ia = row.get('PROBABILIDAD_COLAPSO', 0.0)
        dict_row['Riesgo_IA'] = round(float(riesgo_ia), 1) if pd.notna(riesgo_ia) else 0.0
        
        nr = str(row.get('NIVEL_RIESGO', 'SIN DATOS')).strip().upper()
        dict_row['NIVEL_RIESGO'] = 'SIN DATOS' if nr in ['NAN', '0', '0.0', 'NULL', 'NONE'] or pd.isna(row.get('NIVEL_RIESGO')) else nr
                
        dict_row['Riesgo_IA'] = round(float(row.get('PROBABILIDAD_COLAPSO', 0.0)), 1)
        nr = str(row.get('NIVEL_RIESGO', 'SIN DATOS')).strip().upper()
        dict_row['NIVEL_RIESGO'] = 'SIN DATOS' if nr in ['NAN', '0', '0.0'] else nr
        
        resultados.append(dict_row)
        
    return resultados

@app.get("/api/fechas/recargar")
def recargar_linea_tiempo():
    inicializar_motor()
    return {"fechas": cache.get('fechas_disponibles', [])}