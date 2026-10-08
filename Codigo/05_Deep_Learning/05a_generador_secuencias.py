import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# --- RUTAS MLOPS ---
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
# Ruta a los tensores completos (los que tienen 12D)
DIR_TENSORES = os.path.abspath(os.path.join(DIR_BASE, "..", "..", "Datos", "03_Model_Ready", "tensores_finales"))
# Ruta a los resultados del ensamble
DIR_ENSAMBLE = os.path.abspath(os.path.join(DIR_BASE, "..", "..", "Datos", "04_Resultados", "baseline_ensamble"))
DIR_SALIDA = os.path.abspath(os.path.join(DIR_BASE, "..", "..", "Datos", "05_Deep_Learning"))
os.makedirs(DIR_SALIDA, exist_ok=True)

VENTANA_TIEMPO = 6  

def generar_secuencias_lstm():
    print(f"🕰️ 1/4 Cargando 12 dimensiones físicas y frecuenciales (Memoria = {VENTANA_TIEMPO} meses)...")
    
    # 1. Cargamos el dataset completo con todas las 12 dimensiones
    archivos_tensores = glob.glob(os.path.join(DIR_TENSORES, "*.parquet"))
    if not archivos_tensores:
        print(f"❌ No se encontraron tensores en {DIR_TENSORES}. ¿Ejecutaste la fase 3?")
        return
        
    df_tensores = pd.concat([pd.read_parquet(f) for f in archivos_tensores], ignore_index=True)
    
    # Aseguramos formato de fecha estándar para el cruce
    df_tensores['FECHA_MES'] = pd.to_datetime(df_tensores['FECHA_MES']).dt.strftime('%Y-%m')
    
    # 2. Cargamos el diagnóstico del ensamble (Las etiquetas Y)
    print("🤖 2/4 Cargando etiquetas de anomalías del Ensamble...")
    archivos_ensamble = glob.glob(os.path.join(DIR_ENSAMBLE, "*.parquet"))
    if not archivos_ensamble:
        print(f"❌ No se encontraron predicciones en {DIR_ENSAMBLE}. ¿Ejecutaste la fase 4?")
        return
        
    df_ensamble = pd.concat([pd.read_parquet(f) for f in archivos_ensamble], ignore_index=True)
    df_ensamble['FECHA_MES'] = pd.to_datetime(df_ensamble['FECHA_MES']).dt.strftime('%Y-%m')
    
    # Cruzamos la data maestra (12D) con el diagnóstico del ensamble
    # Forzamos ID_POLIGONO a string en ambos para evitar errores de tipo al cruzar
    df_tensores['ID_POLIGONO'] = df_tensores['ID_POLIGONO'].astype(str)
    df_ensamble['ID_POLIGONO'] = df_ensamble['ID_POLIGONO'].astype(str)
    
    # Hacemos el cruce (merge). Solo tomamos 'ENSAMBLE_PRED' del ensamble.
    try:
         df = pd.merge(df_tensores, df_ensamble[['ID_POLIGONO', 'FECHA_MES', 'ENSAMBLE_PRED']], 
                      on=['ID_POLIGONO', 'FECHA_MES'], how='inner')
    except KeyError as e:
        print(f"❌ Error en el cruce. Falta columna clave en los datos del ensamble: {e}")
        print(f"Columnas en ensamble: {df_ensamble.columns.tolist()}")
        return
                  
    df = df.sort_values(by=['ID_POLIGONO', 'FECHA_MES'])
    
    # Verificamos que queden datos después del cruce
    if df.empty:
        print("❌ El cruce de datos resultó vacío. Verifica que ID_POLIGONO y FECHA_MES coincidan en ambos datasets.")
        return

    features = [
        'NDVI', 'BSI', 'GCI', 'RADAR_VH', 'RADAR_VV', 
        'TMAX_MEDIAN', 'PRECIP_SUM_MES', 'PRECIP_ACUM_3M', 
        'ESTRES_HIDRICO', 'ELEVACION',
        'NDVI_WAVELET_DETALLE', 'PRECIP_WAVELET_DETALLE'
    ]
    
    # Última comprobación de columnas
    faltantes = [f for f in features if f not in df.columns]
    if faltantes:
        print(f"❌ Aún faltan columnas después del merge: {faltantes}")
        return

    print("📐 3/4 Escalando tensores para la Red Neuronal...")
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    X_secuencias = []
    Y_etiquetas = []
    IDs_info = []

    print("🎞️ 4/4 Transformando geometría plana a Tensores 3D [Muestras, Tiempo, Variables]...")
    
    for poligono_id, datos_pol in df.groupby('ID_POLIGONO'):
        valores_x = datos_pol[features].values
        
        # Objetivo (Y): 1 = Deforestación/Colapso (asumiendo -1 en el ensamble), 0 = Sano
        # Ajusta esto si tu ensamble usa convenciones diferentes
        valores_y = np.where(datos_pol['ENSAMBLE_PRED'].values == -1, 1, 0)
        fechas = datos_pol['FECHA_MES'].values
        
        for i in range(len(datos_pol) - VENTANA_TIEMPO):
            secuencia_x = valores_x[i : i + VENTANA_TIEMPO]
            etiqueta_y = valores_y[i + VENTANA_TIEMPO]
            info = [poligono_id, fechas[i + VENTANA_TIEMPO]]
            
            X_secuencias.append(secuencia_x)
            Y_etiquetas.append(etiqueta_y)
            IDs_info.append(info)

    if not X_secuencias:
        print("❌ No se pudieron generar secuencias. ¿Son las series de tiempo más cortas que la ventana?")
        return

    X_tensor = np.array(X_secuencias)
    Y_tensor = np.array(Y_etiquetas)
    Info_tensor = np.array(IDs_info)
    
    print(f"\n✅ Transformación completada con éxito.")
    print(f"   ► Dimensiones de X (Historia): {X_tensor.shape} -> [Muestras, Meses, Variables]")
    print(f"   ► Dimensiones de Y (Predicción): {Y_tensor.shape} -> [Muestras]")
    
    ruta_x = os.path.join(DIR_SALIDA, "X_tensor_lstm.npy")
    ruta_y = os.path.join(DIR_SALIDA, "Y_tensor_lstm.npy")
    ruta_info = os.path.join(DIR_SALIDA, "Info_tensor_lstm.npy")
    
    np.save(ruta_x, X_tensor)
    np.save(ruta_y, Y_tensor)
    np.save(ruta_info, Info_tensor)
    
    print(f"   💾 Tensores exportados a: {DIR_SALIDA}\n")

if __name__ == "__main__":
    generar_secuencias_lstm()