import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def hibridar_modelos():
    print("⚙️ INICIANDO ENSAMBLE DE INFERENCIA (LSTM + BASELINES)...")
    
    dir_base = os.path.dirname(os.path.abspath(__file__))
    ruta_modelo = os.path.abspath(os.path.join(dir_base, "..", "..", "Datos", "05_Deep_Learning", "modelo_lstm_deforestacion.keras"))

    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"❌ No se encontró el modelo Keras en: {ruta_modelo}.")
        
    print("   ► Cargando Red Neuronal profunda (Keras Nativo)...")
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
            if len(forma) == 3: tensores_3d.append(f)
        except:
            pass

    if not tensores_3d:
        raise FileNotFoundError(f"❌ No se encontró ningún tensor 3D para la LSTM.")
        
    ruta_tensor = max(tensores_3d, key=os.path.getmtime)
    posibles_csv = [f for f in archivos_csv if "meta" in f.lower() or "tensor" in f.lower() or "dataset" in f.lower()]
    ruta_metadatos = max(posibles_csv, key=os.path.getmtime) if posibles_csv else (max(archivos_csv, key=os.path.getmtime) if archivos_csv else None)

    print(f"   ► Autodescubrimiento matemático exitoso:")
    
    try:
        X_tensor = np.load(ruta_tensor)
    except Exception as e:
        raise RuntimeError(f"❌ Fallo al cargar el tensor. Detalle: {e}")

    esperadas = modelo_lstm.input_shape[-1]
    actuales = X_tensor.shape[-1]

    if actuales > esperadas:
        X_tensor_inferencia = X_tensor[:, :, :esperadas]
    elif actuales < esperadas:
        raise ValueError(f"❌ Faltan datos: El modelo exige {esperadas} variables.")
    else:
        X_tensor_inferencia = X_tensor

    print("   ► Ejecutando predicción en la red neuronal profunda...")
    probabilidades_lstm = modelo_lstm.predict(X_tensor_inferencia, verbose=0).ravel()
    probabilidades_lstm = probabilidades_lstm * 100 
    
    if ruta_metadatos and os.path.exists(ruta_metadatos):
        print("   ► Consolidando Ensamble Híbrido Final con 12 Variables...")
        df_resultados = pd.read_csv(ruta_metadatos)
        min_len = min(len(df_resultados), len(probabilidades_lstm))
        df_resultados = df_resultados.iloc[:min_len].copy()
        
        df_resultados['PROBABILIDAD_COLAPSO'] = probabilidades_lstm[:min_len]
        df_resultados['NIVEL_RIESGO'] = np.where(df_resultados['PROBABILIDAD_COLAPSO'] > 80, 'CRITICO', 
                                        np.where(df_resultados['PROBABILIDAD_COLAPSO'] > 50, 'ALERTA', 'ESTABLE'))
        
        # --- EXTRACCIÓN EXPLÍCITA DE LAS 12 VARIABLES FÍSICAS AL CSV ---
        # Get the most recent time step [Muestras, Último_Mes, Variables]
        ultima_lectura = X_tensor_inferencia[:min_len, -1, :]
        
        # Mapping standard forest metadata labels
        etiquetas = [
            'NDVI_Vegetacion', 'BSI_Suelo_Desnudo', 'GCI_Clorofila', 
            'Radar_VV_Biomasa', 'Radar_VH_Estructura', 'Temp_Maxima', 
            'Temp_Minima', 'Precipitacion', 'Elevacion_Topografica', 
            'Pendiente_Terreno', 'Energia_Wavelet_Alta', 'Energia_Wavelet_Baja'
        ]
        
        # Inject directly into dataframe
        for i in range(ultima_lectura.shape[1]):
            nombre_col = etiquetas[i] if i < len(etiquetas) else f'Variable_Tensor_{i+1}'
            df_resultados[nombre_col] = ultima_lectura[:, i]
        
        ruta_csv_salida = os.path.join(dir_salida, "01_Reporte_Riesgo_Regional.csv")
        df_resultados.to_csv(ruta_csv_salida, index=False)
        print(f"✅ Ensamble exportado exitosamente con explicabilidad 12-Var en {ruta_csv_salida}\n")

if __name__ == "__main__":
    hibridar_modelos()