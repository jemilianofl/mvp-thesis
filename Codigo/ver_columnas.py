import os
import glob
import pandas as pd

# Rutas absolutas para no fallar
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_PARQUETS = os.path.abspath(os.path.join(DIR_BASE, "..", "Datos", "02_Procesados", "clima_parquet"))

archivos = glob.glob(os.path.join(RUTA_PARQUETS, "**", "*.parquet"), recursive=True)

if archivos:
    df = pd.read_parquet(archivos[0])
    print(f"\n📁 Archivo escaneado: {os.path.basename(archivos[0])}")
    print(f"📊 Columnas reales: {df.columns.tolist()}")
else:
    print("❌ No encontré archivos.")