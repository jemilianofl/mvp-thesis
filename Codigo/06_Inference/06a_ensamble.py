import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def ejecutar_inferencia():
    print("⚙️ INICIANDO ENSAMBLE (VINCULANDO IA CON DATOS FÍSICOS CRUDOS)...")
    
    dir_base = os.path.dirname(os.path.abspath(__file__))
    dir_dl = os.path.abspath(os.path.join(dir_base, "..", "..", "Datos", "05_Deep_Learning"))
    dir_filtrados = os.path.abspath(os.path.join(dir_base, "..", "..", "Datos", "03_Model_Ready", "tensores_filtrados"))
    dir_salida = os.path.abspath(os.path.join(dir_base, "..", "..", "Datos", "06_Inferencia"))
    os.makedirs(dir_salida, exist_ok=True)

    # 1. Cargar Red Neuronal
    ruta_modelo = os.path.join(dir_dl, "modelo_lstm_deforestacion.keras")
    modelo_lstm = tf.keras.models.load_model(ruta_modelo, compile=False)
    
    # 2. Cargar Tensor (X) y Predecir
    archivos_npy = glob.glob(os.path.join(dir_dl, "*.npy"))
    ruta_tensor = max([f for f in archivos_npy if len(np.load(f, mmap_mode='r').shape) == 3], key=os.path.getmtime)
    X_tensor = np.load(ruta_tensor)
    
    esperadas = modelo_lstm.input_shape[-1]
    X_inferencia = X_tensor[:, :, :esperadas] if X_tensor.shape[-1] > esperadas else X_tensor
    
    print(f"   ► Evaluando secuencias temporales ({X_inferencia.shape[0]} muestras)...")
    probabilidades = modelo_lstm.predict(X_inferencia, verbose=0).ravel() * 100
    
    # 3. Cargar Metadata
    ruta_meta = max(glob.glob(os.path.join(dir_dl, "*.csv")), key=os.path.getmtime)
    df_meta = pd.read_csv(ruta_meta).iloc[:len(probabilidades)].copy()
    df_meta['PROBABILIDAD_COLAPSO'] = probabilidades
    df_meta['NIVEL_RIESGO'] = np.where(df_meta['PROBABILIDAD_COLAPSO'] > 80, 'CRITICO', 
                                np.where(df_meta['PROBABILIDAD_COLAPSO'] > 50, 'ALERTA', 'ESTABLE'))
    
    # FIX: Limpieza de IDs y Fechas en Metadata
    df_meta['ID_POLIGONO'] = df_meta['ID_POLIGONO'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    col_fecha_meta = 'FECHA_DIA' if 'FECHA_DIA' in df_meta.columns else 'FECHA_MES'
    df_meta[col_fecha_meta] = pd.to_datetime(df_meta[col_fecha_meta]).dt.strftime('%Y-%m-%d')

    # 4. Cargar DATOS FÍSICOS ORIGINALES
    print("   ► Recuperando variables físicas originales...")
    archivos_pq = glob.glob(os.path.join(dir_filtrados, "*.parquet"))
    if not archivos_pq:
        raise FileNotFoundError("No se encontraron tensores filtrados.")
        
    df_fisico = pd.concat([pd.read_parquet(f) for f in archivos_pq], ignore_index=True)
    
    # FIX: Limpieza de IDs y Fechas en Datos Físicos
    df_fisico['ID_POLIGONO'] = df_fisico['ID_POLIGONO'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    col_fecha_fis = 'FECHA_DIA' if 'FECHA_DIA' in df_fisico.columns else 'FECHA_MES'
    df_fisico[col_fecha_fis] = pd.to_datetime(df_fisico[col_fecha_fis]).dt.strftime('%Y-%m-%d')

   # 5. Fusionar IA con Datos Físicos
    print("   ► Sincronizando variables (Ignorando desfase temporal, forzando coincidencia espacial)...")
    
    # 1. Obtener la ÚLTIMA lectura física disponible por polígono (sin importar de qué mes sea)
    df_fisico_latest = df_fisico.sort_values(col_fecha_fis).groupby('ID_POLIGONO').last().reset_index()
    
    # 2. Cruzar exclusivamente por ID de polígono
    df_final = pd.merge(df_meta, df_fisico_latest, on='ID_POLIGONO', how='left')
    
    # 3. Limpiar duplicados de nombres de columnas que deja el cruce
    if 'FECHA_DIA_y' in df_final.columns: df_final = df_final.drop(columns=['FECHA_DIA_y'])
    if 'FECHA_DIA_x' in df_final.columns: df_final = df_final.rename(columns={'FECHA_DIA_x': 'FECHA_DIA'})
    if 'FECHA_MES_y' in df_final.columns: df_final = df_final.drop(columns=['FECHA_MES_y'])
    if 'FECHA_MES_x' in df_final.columns: df_final = df_final.rename(columns={'FECHA_MES_x': 'FECHA_MES'})

    # 4. Calibración Híbrida: Si el sensor físico detecta deforestación, forzar la Alerta de la IA
    if 'ENSAMBLE_PRED' in df_final.columns:
        # Si la IA dijo 0% pero hay evidencia física en campo, elevamos el riesgo a 85% (Crítico)
        df_final['PROBABILIDAD_COLAPSO'] = np.where(
            (df_final['ENSAMBLE_PRED'] == 1) & (df_final['PROBABILIDAD_COLAPSO'] < 80),
            85.0, 
            df_final['PROBABILIDAD_COLAPSO']
        )
        # Recalcular el nivel de texto
        df_final['NIVEL_RIESGO'] = np.where(df_final['PROBABILIDAD_COLAPSO'] > 80, 'CRITICO', 
                                     np.where(df_final['PROBABILIDAD_COLAPSO'] > 50, 'ALERTA', 'ESTABLE'))

    # Exportar Reporte Final
    ruta_csv = os.path.join(dir_salida, "01_Reporte_Riesgo_Regional.csv")
    df_final.to_csv(ruta_csv, index=False)
    print(f"✅ Reporte calibrado exportado a: {ruta_csv}")
    
if __name__ == "__main__":
    ejecutar_inferencia()