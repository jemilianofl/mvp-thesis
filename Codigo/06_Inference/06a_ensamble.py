import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def ejecutar_inferencia_global():
    print("⚙️ INICIANDO INFERENCIA GLOBAL DIARIA...")
    
    dir_base = os.path.dirname(os.path.abspath(__file__))
    ruta_modelo = os.path.abspath(os.path.join(dir_base, "..", "..", "Datos", "05_Deep_Learning", "modelo_lstm_deforestacion.keras"))

    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError("❌ No hay modelo entrenado en la ruta especificada.")
        
    modelo_lstm = tf.keras.models.load_model(ruta_modelo, compile=False)
    dir_salida = os.path.abspath(os.path.join(dir_base, "..", "..", "Datos", "06_Inferencia"))
    os.makedirs(dir_salida, exist_ok=True)

    dir_features = os.path.abspath(os.path.join(dir_base, "..", "..", "Datos", "02_Feature_Engineering"))
    dir_resultados = os.path.abspath(os.path.join(dir_base, "..", "..", "Datos", "04_Resultados"))
    dir_deep = os.path.abspath(os.path.join(dir_base, "..", "..", "Datos", "05_Deep_Learning"))
    
    archivos_npy = glob.glob(os.path.join(dir_features, "*.npy")) + glob.glob(os.path.join(dir_resultados, "*.npy")) + glob.glob(os.path.join(dir_deep, "*.npy"))
    archivos_csv = glob.glob(os.path.join(dir_features, "*.csv")) + glob.glob(os.path.join(dir_resultados, "*.csv")) + glob.glob(os.path.join(dir_deep, "*.csv"))

    tensores_3d = []
    for f in archivos_npy:
        try:
            forma = np.load(f, mmap_mode='r').shape
            if len(forma) == 3: tensores_3d.append((f, forma[0]))
        except: pass

    if not tensores_3d:
        raise FileNotFoundError("❌ No se encontró ningún tensor 3D válido para la inferencia.")
        
    # FIX: Forzar la selección del tensor original masivo, no la versión enmascarada
    ruta_tensor, max_muestras = max(tensores_3d, key=lambda x: x[1])
    X_tensor = np.load(ruta_tensor)
    
    esperadas = modelo_lstm.input_shape[-1]
    X_tensor_inferencia = X_tensor[:, :, :esperadas] if X_tensor.shape[-1] > esperadas else X_tensor

    print(f"   ► Evaluando la región completa ({X_tensor_inferencia.shape[0]} muestras)...")
    probabilidades_lstm = modelo_lstm.predict(X_tensor_inferencia, verbose=0).ravel() * 100 
    
    # FIX: Buscar exactamente el CSV de metadatos que coincida con las 185,000 muestras
    ruta_metadatos = None
    for f in archivos_csv:
        try:
            df_temp = pd.read_csv(f)
            if len(df_temp) == len(probabilidades_lstm):
                ruta_metadatos = f
                break
        except: pass

    # Si no lo encuentra por largo exacto, busca el CSV más pesado
    if not ruta_metadatos:
        print("   ⚠️ Buscando archivo maestro alternativo...")
        archivos_csv_validos = []
        for f in archivos_csv:
            try: archivos_csv_validos.append((f, len(pd.read_csv(f))))
            except: pass
        if archivos_csv_validos:
            ruta_metadatos = max(archivos_csv_validos, key=lambda x: x[1])[0]

    if ruta_metadatos:
        df_resultados = pd.read_csv(ruta_metadatos)
        min_len = min(len(df_resultados), len(probabilidades_lstm))
        df_resultados = df_resultados.iloc[:min_len].copy()
        
        df_resultados['PROBABILIDAD_COLAPSO'] = probabilidades_lstm[:min_len]
        df_resultados['NIVEL_RIESGO'] = np.where(df_resultados['PROBABILIDAD_COLAPSO'] > 80, 'CRITICO', 
                                        np.where(df_resultados['PROBABILIDAD_COLAPSO'] > 50, 'ALERTA', 'ESTABLE'))
        
        if 'FECHA_DIA' not in df_resultados.columns and 'FECHA_MES' not in df_resultados.columns:
            df_resultados['FECHA_DIA'] = datetime.now().strftime('%Y-%m-%d')
            
        # Extracción de las 12 Variables Físicas Crudas para TODA la región
        ultima_lectura = X_tensor_inferencia[:min_len, -1, :]
        etiquetas = ['NDVI', 'BSI', 'GCI', 'RADAR_VH', 'RADAR_VV', 'TMAX_MEDIAN', 'PRECIP_SUM_MES', 'PRECIP_ACUM_3M', 'ESTRES_HIDRICO', 'ELEVACION', 'NDVI_WAVELET_DETALLE', 'PRECIP_WAVELET_DETALLE']
        
        for i in range(min(ultima_lectura.shape[1], len(etiquetas))):
            df_resultados[etiquetas[i]] = ultima_lectura[:, i]
        
        ruta_csv_salida = os.path.join(dir_salida, "01_Reporte_Riesgo_Regional.csv")
        
        # Sobreescritura limpia para evitar duplicados
        df_resultados.to_csv(ruta_csv_salida, index=False)
        print(f"✅ Inferencia regional masiva ({min_len} polígonos) guardada exitosamente en:\n   {ruta_csv_salida}")

if __name__ == "__main__":
    ejecutar_inferencia_global()