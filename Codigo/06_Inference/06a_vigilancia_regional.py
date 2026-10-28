import os

import numpy as np
import pandas as pd


def suavizar_predicciones():
    print("🛰 Generando Progresión Diaria de Alerta Temprana (Bajo Ruido)...")
    
    dir_base = os.path.dirname(os.path.abspath(__file__))
    ruta_csv = os.path.abspath(os.path.join(dir_base, "..", "..", "Datos", "06_Inferencia", "01_Reporte_Riesgo_Regional.csv"))

    if not os.path.exists(ruta_csv):
        print(f"   ⚠️ No se encontró el archivo de riesgo en: {ruta_csv}")
        return

    df = pd.read_csv(ruta_csv)
    col_fecha = 'FECHA_DIA' if 'FECHA_DIA' in df.columns else 'FECHA_MES'
    
    if col_fecha not in df.columns or 'PROBABILIDAD_COLAPSO' not in df.columns:
        print("   ⚠️ El archivo CSV no tiene la estructura esperada.")
        return

    df[col_fecha] = pd.to_datetime(df[col_fecha])
    df = df.sort_values(['ID_POLIGONO', col_fecha])
    
    # FIX: Suavizado ultra-ligero (alpha = 0.8) para preservar los picos verdaderos de riesgo
    df['PROBABILIDAD_COLAPSO'] = df.groupby('ID_POLIGONO')['PROBABILIDAD_COLAPSO'].transform(
        lambda x: x.ewm(alpha=0.8, adjust=False).mean()
    )
    
    df['NIVEL_RIESGO'] = np.where(df['PROBABILIDAD_COLAPSO'] > 80, 'CRITICO', 
                         np.where(df['PROBABILIDAD_COLAPSO'] > 50, 'ALERTA', 'ESTABLE'))

    df[col_fecha] = df[col_fecha].dt.strftime('%Y-%m-%d')
    
    # Esto sobrescribe el archivo manteniendo todas las columnas originales intactas (incluyendo las variables)
    df.to_csv(ruta_csv, index=False)
    print(f"✅ Secuencia ejecutiva diaria exportada y calibrada en {ruta_csv}")


if __name__ == "__main__":
    suavizar_predicciones()