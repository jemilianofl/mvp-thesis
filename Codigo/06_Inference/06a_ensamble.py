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
    
    archivos_npy = (glob.glob(os.path.join(dir_features, "*.npy")) + 
                    glob.glob(os.path.join(dir_resultados, "*.npy")) + 
                    glob.glob(os.path.join(dir_deep, "*.npy")))
    
    archivos_csv = (glob.glob(os.path.join(dir_features, "*.csv")) + 
                    glob.glob(os.path.join(dir_resultados, "*.csv")) + 
                    glob.glob(os.path.join(dir_deep, "*.csv")))

    tensores_3d = []
    for f in archivos_npy:
        try:
            forma = np.load(f, mmap_mode='r').shape
            if len(forma) == 3: tensores_3d.append(f)
        except:
            pass

    if not tensores_3d:
        raise FileNotFoundError("❌ No se encontró ningún tensor 3D válido para la inferencia.")
        
    ruta_tensor = max(tensores_3d, key=os.path.getmtime)
    X_tensor = np.load(ruta_tensor)
    
    esperadas = modelo_lstm.input_shape[-1]
    if X_tensor.shape[-1] > esperadas:
        X_tensor_inferencia = X_tensor[:, :, :esperadas]
    elif X_tensor.shape[-1] < esperadas:
        raise ValueError(f"❌ Faltan datos: El modelo exige {esperadas} variables.")
    else:
        X_tensor_inferencia = X_tensor

    print(f"   ► Evaluando la región completa ({X_tensor_inferencia.shape[0]} muestras)...")
    probabilidades_lstm = modelo_lstm.predict(X_tensor_inferencia, verbose=0).ravel() * 100 
    
    posibles_csv = [f for f in archivos_csv if "meta" in f.lower() or "tensor" in f.lower() or "dataset" in f.lower()]
    ruta_metadatos = max(posibles_csv, key=os.path.getmtime) if posibles_csv else (max(archivos_csv, key=os.path.getmtime) if archivos_csv else None)

    if ruta_metadatos:
        # FIX CLAVE: Leemos los DATOS CRUDOS originales y no volvemos a inyectarles el tensor escalado
        df_resultados = pd.read_csv(ruta_metadatos)
        min_len = min(len(df_resultados), len(probabilidades_lstm))
        df_resultados = df_resultados.iloc[:min_len].copy()
        
        # Agregamos UNICAMENTE la respuesta de la Inteligencia Artificial
        df_resultados['PROBABILIDAD_COLAPSO'] = probabilidades_lstm[:min_len]
        df_resultados['NIVEL_RIESGO'] = np.where(df_resultados['PROBABILIDAD_COLAPSO'] > 80, 'CRITICO', 
                                        np.where(df_resultados['PROBABILIDAD_COLAPSO'] > 50, 'ALERTA', 'ESTABLE'))
        
        if 'FECHA_DIA' not in df_resultados.columns and 'FECHA_MES' not in df_resultados.columns:
            df_resultados['FECHA_DIA'] = datetime.now().strftime('%Y-%m-%d')
        
        ruta_csv_salida = os.path.join(dir_salida, "01_Reporte_Riesgo_Regional.csv")
        
        # Guardamos preservando todas las variables físicas originales
        df_resultados.to_csv(ruta_csv_salida, index=False)
            
        print(f"✅ Inferencia diaria vinculada a variables físicas guardada en:\n   {ruta_csv_salida}")

if __name__ == "__main__":
    ejecutar_inferencia_global()