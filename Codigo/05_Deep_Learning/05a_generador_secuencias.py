import os
import polars as pl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# --- RUTAS MLOPS ---
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_ENSAMBLE = os.path.abspath(os.path.join(DIR_BASE, "..", "..", "Datos", "03_Modelos", "resultados_ensamble.parquet"))
DIR_SALIDA = os.path.abspath(os.path.join(DIR_BASE, "..", "..", "Datos", "05_Deep_Learning"))
os.makedirs(DIR_SALIDA, exist_ok=True)

# Hiperparámetro crucial: ¿Cuántos meses debe "recordar" la red neuronal para predecir el estado actual?
VENTANA_TIEMPO = 6  

def generar_secuencias_lstm():
    print(f"🕰️ 1/3 Cargando datos e inicializando ventanas de tiempo (T={VENTANA_TIEMPO} meses)...")
    df = pl.read_parquet(RUTA_ENSAMBLE).to_pandas()
    df['FECHA_MES'] = pd.to_datetime(df['FECHA_MES'])
    df = df.sort_values(by=['ID_POLIGONO', 'FECHA_MES'])
    
    # Seleccionamos las variables que la red neuronal debe aprender
    features = ['NDVI', 'BSI', 'GCI', 'RADAR_VH', 'TMAX_MEDIAN', 'PRECIP_MAX_DIA']
    
    print("📐 2/3 Escalando variables físicas para la Red Neuronal...")
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    X_secuencias = []
    Y_etiquetas = []
    IDs_info = []

    print("🎞️ 3/3 Transformando geometría plana a Tensores 3D [Muestras, Tiempo, Variables]...")
    
    # Agrupamos por polígono para no mezclar la historia de un asentamiento con otro
    for poligono_id, datos_pol in df.groupby('ID_POLIGONO'):
        # Convertimos los datos a arrays de numpy para hacer los cortes
        valores_x = datos_pol[features].values
        
        # El objetivo (Y) será predecir si el mes actual es una anomalía (-1) o normal (1)
        # Convertimos la anomalía a formato Binario Clásico (0 = Sano, 1 = Degradado) para TensorFlow
        valores_y = np.where(datos_pol['ANOMALIA_ENSAMBLE'].values == -1, 1, 0)
        fechas = datos_pol['FECHA_MES'].dt.strftime('%Y-%m').values
        
        # Deslizamiento de la ventana
        for i in range(len(datos_pol) - VENTANA_TIEMPO):
            # X: La historia de los 6 meses anteriores
            secuencia_x = valores_x[i : i + VENTANA_TIEMPO]
            
            # Y: El estado de degradación en el mes 7
            etiqueta_y = valores_y[i + VENTANA_TIEMPO]
            
            # Info extra para saber a qué polígono y fecha corresponde esta predicción
            info = [poligono_id, fechas[i + VENTANA_TIEMPO]]
            
            X_secuencias.append(secuencia_x)
            Y_etiquetas.append(etiqueta_y)
            IDs_info.append(info)

    # Convertimos las listas a Tensores Reales (Numpy Arrays 3D)
    X_tensor = np.array(X_secuencias)
    Y_tensor = np.array(Y_etiquetas)
    Info_tensor = np.array(IDs_info)
    
    print(f"\n✅ Transformación completada con éxito.")
    print(f"   ► Dimensiones de X (Historia): {X_tensor.shape} -> [Muestras, Meses, Variables]")
    print(f"   ► Dimensiones de Y (Predicción): {Y_tensor.shape} -> [Muestras]")
    
    # Guardamos los tensores para que el script de la red neuronal los absorba
    ruta_x = os.path.join(DIR_SALIDA, "X_tensor_lstm.npy")
    ruta_y = os.path.join(DIR_SALIDA, "Y_tensor_lstm.npy")
    ruta_info = os.path.join(DIR_SALIDA, "Info_tensor_lstm.npy")
    
    np.save(ruta_x, X_tensor)
    np.save(ruta_y, Y_tensor)
    np.save(ruta_info, Info_tensor)
    
    print(f"   💾 Tensores exportados a: {DIR_SALIDA}\n")

if __name__ == "__main__":
    generar_secuencias_lstm()