import os
import polars as pl
import pandas as pd
import numpy as np
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")

# --- RUTAS MLOPS ---
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_ENSAMBLE = os.path.abspath(os.path.join(DIR_BASE, "..", "..", "Datos", "03_Modelos", "resultados_ensamble.parquet"))
DIR_DL = os.path.abspath(os.path.join(DIR_BASE, "..", "..", "Datos", "05_Deep_Learning"))
RUTA_MODELO_LSTM = os.path.join(DIR_DL, "modelo_lstm_deforestacion.h5")
RUTA_X = os.path.join(DIR_DL, "X_tensor_lstm.npy")
RUTA_INFO = os.path.join(DIR_DL, "Info_tensor_lstm.npy")
DIR_SALIDA = os.path.abspath(os.path.join(DIR_BASE, "..", "..", "Datos", "06_Inferencia"))
os.makedirs(DIR_SALIDA, exist_ok=True)

def hibridar_modelos():
    print("🧬 1/3 Cargando Súper-Ensamble (Modelo Genético Espacial + Tensores Temporales)...")
    df_base = pl.read_parquet(RUTA_ENSAMBLE).to_pandas()
    X_tensor = np.load(RUTA_X)
    Info_tensor = np.load(RUTA_INFO)
    
    print("🧠 2/3 Despertando Red Neuronal (LSTM) para predicción global...")
    modelo_lstm = tf.keras.models.load_model(RUTA_MODELO_LSTM)
    
    # Predecir la probabilidad de colapso para todas las secuencias
    probabilidades_lstm = modelo_lstm.predict(X_tensor, verbose=0).ravel()
    
    # Crear un DataFrame temporal con las predicciones de la LSTM
    df_lstm = pd.DataFrame({
        'ID_POLIGONO': Info_tensor[:, 0],
        'FECHA_MES_STR': Info_tensor[:, 1], # Este es un string 'YYYY-MM'
        'PROBABILIDAD_LSTM': probabilidades_lstm
    })
    
    # Preparar el dataframe base para el cruce exacto
    df_base['ID_POLIGONO'] = df_base['ID_POLIGONO'].astype(str)
    df_lstm['ID_POLIGONO'] = df_lstm['ID_POLIGONO'].astype(str)
    # Convertimos la fecha del parquet (datetime) al mismo formato string de la LSTM
    df_base['FECHA_MES_STR'] = pd.to_datetime(df_base['FECHA_MES']).dt.strftime('%Y-%m')
    
    # Fusionar el conocimiento Espacial con el Temporal
    df_hibrido = pd.merge(df_base, df_lstm, on=['ID_POLIGONO', 'FECHA_MES_STR'], how='inner')
    
    print("⚖️ 3/3 Aplicando Reglas de Validación Cruzada (Filtro de Falsos Positivos)...")
    
    # CORRECCIÓN METODOLÓGICA: Calculamos el promedio de probabilidad de la red
    # y pedimos que el evento esté por encima del promedio, NO un número mágico
    umbral_dinamico = df_hibrido['PROBABILIDAD_LSTM'].mean() + df_hibrido['PROBABILIDAD_LSTM'].std()
    print(f"   ► Umbral dinámico de alarma LSTM calculado en: {umbral_dinamico:.3f}")
    
    # REGLA DE ORO ACTUALIZADA: Anómalo espacialmente Y colapso temporal inminente
    condicion_critica = (df_hibrido['ANOMALIA_ENSAMBLE'] == -1) & (df_hibrido['PROBABILIDAD_LSTM'] >= umbral_dinamico)
    
    df_hibrido['ESTADO_HIBRIDO'] = np.where(condicion_critica, 'DEFORESTACION_CONFIRMADA', 'Sano / Ruido')
    
    deforestacion_real = df_hibrido[df_hibrido['ESTADO_HIBRIDO'] == 'DEFORESTACION_CONFIRMADA']
    
    print(f"\n✅ SÚPER-ENSAMBLE COMPLETADO.")
    print(f"   ► Quedan {len(deforestacion_real)} incidentes con DEGRADACIÓN TOTALMENTE CONFIRMADA (Espacial + Temporal).")
    
    ruta_export = os.path.join(DIR_SALIDA, "01_Alerta_Deforestacion_Hibrida.csv")
    deforestacion_real.to_csv(ruta_export, index=False)
    print(f"   💾 Reporte de incidentes críticos guardado en: {ruta_export}")

if __name__ == "__main__":
    hibridar_modelos()