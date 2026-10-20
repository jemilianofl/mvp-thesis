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
    # ------------------ REEMPLAZAR BLOQUE DE CARGA ------------------
    ruta_modelo = os.path.abspath(os.path.join(dir_base, "..", "..", "Datos", "05_Deep_Learning", "modelo_compacto.tflite"))

    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"❌ No se encontró el modelo TFLite en: {ruta_modelo}.")
        
    print("   ► Inicializando Intérprete TFLite ultraligero...")
    with open(ruta_modelo, "rb") as f:
        modelo_bytes = f.read()

    # Habilita Select TF Ops (Flex Delegate) para inferencia
    interpreter = tf.lite.Interpreter(
        model_content=modelo_bytes,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # ---------------------------------------------------------
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

    # FIX MLOPS DEFINITIVO: No dependemos del nombre del archivo.
    # Escaneamos la "forma matemática" (shape) de todos los .npy. 
    # El tensor de la LSTM SIEMPRE tiene 3 dimensiones (muestras, meses, variables).
    tensores_3d = []
    for f in archivos_npy:
        try:
            # mmap_mode='r' lee los metadatos sin saturar la memoria RAM
            forma = np.load(f, mmap_mode='r').shape
            if len(forma) == 3: 
                tensores_3d.append(f)
        except:
            pass

    if not tensores_3d:
        raise FileNotFoundError(f"❌ No se encontró ningún tensor 3D para la LSTM en las carpetas de datos.")
        
    # Tomamos el tensor 3D más reciente que haya generado el pipeline
    ruta_tensor = max(tensores_3d, key=os.path.getmtime)
    
    posibles_csv = [f for f in archivos_csv if "meta" in f.lower() or "tensor" in f.lower() or "dataset" in f.lower()]
    ruta_metadatos = max(posibles_csv, key=os.path.getmtime) if posibles_csv else (max(archivos_csv, key=os.path.getmtime) if archivos_csv else None)

    print(f"   ► Autodescubrimiento matemático exitoso:")
    print(f"      - Tensor 3D detectado: {os.path.basename(ruta_tensor)} con forma {np.load(ruta_tensor, mmap_mode='r').shape}")
    if ruta_metadatos: print(f"      - Metadatos detectados: {os.path.basename(ruta_metadatos)}")
    # =====================================================================
    
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"❌ No se encontró el modelo en: {ruta_modelo}. Asegúrate de haber entrenado la red.")
        
    print("   ► Cargando Red Neuronal LSTM...")
    modelo_lstm = tf.keras.models.load_model(ruta_modelo, compile=False) 
    
    try:
        X_tensor = np.load(ruta_tensor)
    except Exception as e:
        raise RuntimeError(f"❌ Fallo al cargar el tensor. Detalle: {e}")

    # LÓGICA MLOPS: EVALUACIÓN COMPARATIVA (SHAPE-AWARE)
    esperadas = modelo_lstm.input_shape[-1]
    actuales = X_tensor.shape[-1]

    if actuales > esperadas:
        print(f"   ⚠️ Adaptación Activa: El Modelo fue entrenado con {esperadas} variables.")
        print(f"   ⚠️ El Tensor actual tiene {actuales} variables (Edafología + Wavelets inyectados).")
        print(f"   ✂️ Aplicando Opción 1 (Modelo Base): Recortando variables extras para evaluar el Baseline...")
        X_tensor_inferencia = X_tensor[:, :, :esperadas]
        
        # ------------------ REEMPLAZAR LÓGICA DE INFERENCIA ------------------
        print("   ► Ejecutando predicción en el tensor cuantizado...")
        X_tensor_inferencia = X_tensor_inferencia.astype(np.float32) # TFLite exige Float32 estricto
        probabilidades_lstm = []

        for muestra in X_tensor_inferencia:
            # TFLite procesa de 1 en 1 (o requiere reajustar el tamaño del tensor base)
            tensor_entrada = np.expand_dims(muestra, axis=0) 
            interpreter.set_tensor(input_details[0]['index'], tensor_entrada)
            interpreter.invoke()
            prob = interpreter.get_tensor(output_details[0]['index'])
            probabilidades_lstm.append(prob[0][0])
            
        probabilidades_lstm = np.array(probabilidades_lstm)
        probabilidades_lstm = probabilidades_lstm * 100 # Escalar a Porcentaje
        # ---------------------------------------------------------
        
    elif actuales < esperadas:
        raise ValueError(f"❌ Faltan datos: El modelo exige {esperadas} variables pero se enviaron {actuales}.")
        
    else:
        print(f"   🧠 Aplicando Opción 2 (Modelo Enriquecido): Inferencia utilizando el 100% de las {actuales} variables...")
        X_tensor_inferencia = X_tensor

    print("   ► Ejecutando predicción en la red neuronal profunda...")
    probabilidades_lstm = modelo_lstm.predict(X_tensor_inferencia, verbose=0).ravel()
    
    print(f"✅ Inferencia completada. Se generaron {len(probabilidades_lstm)} predicciones de riesgo.")
    
    # ENSAMBLE Y EXPORTACIÓN
    if ruta_metadatos and os.path.exists(ruta_metadatos):
        print("   ► Consolidando Ensamble Híbrido Final...")
        df_resultados = pd.read_csv(ruta_metadatos)
        
        # Ajustar si las longitudes no coinciden exactamente
        # ------------------ REEMPLAZAR EXPORTACIÓN FINAL ------------------
        min_len = min(len(df_resultados), len(probabilidades_lstm))
        df_resultados = df_resultados.iloc[:min_len].copy()
        
        df_resultados['PROBABILIDAD_COLAPSO'] = probabilidades_lstm[:min_len]
        df_resultados['NIVEL_RIESGO'] = np.where(df_resultados['PROBABILIDAD_COLAPSO'] > 80, 'CRITICO', 
                                        np.where(df_resultados['PROBABILIDAD_COLAPSO'] > 50, 'ALERTA', 'ESTABLE'))
        
        ruta_csv_salida = os.path.join(dir_salida, "01_Reporte_Riesgo_Regional.csv")
        df_resultados.to_csv(ruta_csv_salida, index=False)
        print(f"✅ Ensamble exportado exitosamente a {ruta_csv_salida}\n")
    # ---------------------------------------------------------
    else:
        print(f"   ⚠️️ No se encontraron metadatos CSV válidos. Las predicciones no se pudieron acoplar a sus coordenadas.")

if __name__ == "__main__":
    hibridar_modelos()