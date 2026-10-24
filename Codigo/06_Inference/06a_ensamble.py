import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings

# Ocultar warnings de librerías y de C++ de TensorFlow para una terminal limpia
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def hibridar_modelos():
    print("⚙️ INICIANDO ENSAMBLE DE INFERENCIA (LSTM + BASELINES)...")
    
    dir_base = os.path.dirname(os.path.abspath(__file__))
    # Apuntamos al modelo nativo de Keras en lugar del TFLite
    ruta_modelo = os.path.abspath(os.path.join(dir_base, "..", "..", "Datos", "05_Deep_Learning", "modelo_lstm_deforestacion.keras"))

    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"❌ No se encontró el modelo Keras en: {ruta_modelo}.")
        
    print("   ► Cargando Red Neuronal profunda (Keras Nativo)...")
    modelo_lstm = tf.keras.models.load_model(ruta_modelo, compile=False)
    
    dir_salida = os.path.abspath(os.path.join(dir_base, "..", "..", "Datos", "06_Inferencia"))
    os.makedirs(dir_salida, exist_ok=True)

    # =====================================================================
    # LÓGICA MLOPS: AUTO-DISCOVERY MATEMÁTICO DE TENSORES
    # =====================================================================
    dir_features = os.path.abspath(os.path.join(dir_base, "..", "..", "Datos", "02_Feature_Engineering"))
    dir_resultados = os.path.abspath(os.path.join(dir_base, "..", "..", "Datos", "04_Resultados"))
    dir_deep = os.path.abspath(os.path.join(dir_base, "..", "..", "Datos", "05_Deep_Learning"))
    
    archivos_npy = (
        glob.glob(os.path.join(dir_features, "*.npy")) + 
        glob.glob(os.path.join(dir_resultados, "*.npy")) +
        glob.glob(os.path.join(dir_deep, "*.npy"))
    )
    archivos_csv = (
        glob.glob(os.path.join(dir_features, "*.csv")) + 
        glob.glob(os.path.join(dir_resultados, "*.csv")) +
        glob.glob(os.path.join(dir_deep, "*.csv"))
    )

    # Escaneamos la "forma matemática" (shape) de todos los .npy. 
    tensores_3d = []
    for f in archivos_npy:
        try:
            forma = np.load(f, mmap_mode='r').shape
            if len(forma) == 3: 
                tensores_3d.append(f)
        except:
            pass

    if not tensores_3d:
        raise FileNotFoundError(f"❌ No se encontró ningún tensor 3D para la LSTM en las carpetas de datos.")
        
    ruta_tensor = max(tensores_3d, key=os.path.getmtime)
    
    posibles_csv = [f for f in archivos_csv if "meta" in f.lower() or "tensor" in f.lower() or "dataset" in f.lower()]
    ruta_metadatos = max(posibles_csv, key=os.path.getmtime) if posibles_csv else (max(archivos_csv, key=os.path.getmtime) if archivos_csv else None)

    print(f"   ► Autodescubrimiento matemático exitoso:")
    print(f"      - Tensor 3D detectado: {os.path.basename(ruta_tensor)} con forma {np.load(ruta_tensor, mmap_mode='r').shape}")
    if ruta_metadatos: print(f"      - Metadatos detectados: {os.path.basename(ruta_metadatos)}")
    # =====================================================================
    
    try:
        X_tensor = np.load(ruta_tensor)
    except Exception as e:
        raise RuntimeError(f"❌ Fallo al cargar el tensor. Detalle: {e}")

    # LÓGICA MLOPS: EVALUACIÓN COMPARATIVA (SHAPE-AWARE)
    esperadas = modelo_lstm.input_shape[-1]
    actuales = X_tensor.shape[-1]

    if actuales > esperadas:
        print(f"   ⚠️ Adaptación Activa: El Modelo fue entrenado con {esperadas} variables.")
        print(f"   ⚠️ El Tensor actual tiene {actuales} variables.")
        print(f"   ✂️ Aplicando Opción 1 (Modelo Base): Recortando variables extras...")
        X_tensor_inferencia = X_tensor[:, :, :esperadas]
    elif actuales < esperadas:
        raise ValueError(f"❌ Faltan datos: El modelo exige {esperadas} variables pero se enviaron {actuales}.")
    else:
        print(f"   🧠 Aplicando Opción 2 (Modelo Enriquecido): Inferencia utilizando el 100% de las {actuales} variables...")
        X_tensor_inferencia = X_tensor

    print("   ► Ejecutando predicción en la red neuronal profunda...")
    probabilidades_lstm = modelo_lstm.predict(X_tensor_inferencia, verbose=0).ravel()
    probabilidades_lstm = probabilidades_lstm * 100 # Escalar a Porcentaje
    
    print(f"✅ Inferencia completada. Se generaron {len(probabilidades_lstm)} predicciones de riesgo.")
    
    # ENSAMBLE Y EXPORTACIÓN
    if ruta_metadatos and os.path.exists(ruta_metadatos):
        print("   ► Consolidando Ensamble Híbrido Final...")
        df_resultados = pd.read_csv(ruta_metadatos)
        
        min_len = min(len(df_resultados), len(probabilidades_lstm))
        df_resultados = df_resultados.iloc[:min_len].copy()
        
        df_resultados['PROBABILIDAD_COLAPSO'] = probabilidades_lstm[:min_len]
        df_resultados['NIVEL_RIESGO'] = np.where(df_resultados['PROBABILIDAD_COLAPSO'] > 80, 'CRITICO', 
                                        np.where(df_resultados['PROBABILIDAD_COLAPSO'] > 50, 'ALERTA', 'ESTABLE'))
        
        ruta_csv_salida = os.path.join(dir_salida, "01_Reporte_Riesgo_Regional.csv")
        df_resultados.to_csv(ruta_csv_salida, index=False)
        print(f"✅ Ensamble exportado exitosamente a {ruta_csv_salida}\n")
    else:
        print(f"   ⚠ No se encontraron metadatos CSV válidos. Las predicciones no se pudieron acoplar a sus coordenadas.")

if __name__ == "__main__":
    hibridar_modelos()