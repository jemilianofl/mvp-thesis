import os
import geopandas as gpd
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

class AlertaTempranaStakeholders:
    def __init__(self):
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.ruta_malla = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "malla_regional_yucatan.geojson"))
        self.dir_salida = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "06_Inferencia"))

    def ejecutar(self):
        print("🛰️️ Generando Progresión Diaria de Alerta Temprana (Bajo Ruido)...")
        gdf_malla = gpd.read_file(self.ruta_malla)

        fecha_inicio = datetime(2026, 6, 1)
        fechas_diarias = [(fecha_inicio + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(15)]
        
        resultados = []
        np.random.seed(101) 
        
        # Seleccionamos SOLO 50 focos de degradación en toda la península
        focos = np.random.choice(len(gdf_malla), size=50, replace=False)
        
        for dia_idx, fecha in enumerate(fechas_diarias):
            # Base: Toda la selva está muy sana (0% a 15% de riesgo)
            probs = np.random.uniform(0, 15, size=len(gdf_malla))
            ndvi = np.random.uniform(0.7, 0.9, size=len(gdf_malla))
            
            # Evolución progresiva de los 50 focos:
            # Dia 0: 20% (Sano) -> Dia 7: 55% (Alerta) -> Dia 14: 85% (Crítico)
            riesgo_diario = 20 + (dia_idx * 4.5) 
            probs[focos] = np.random.uniform(riesgo_diario - 2, riesgo_diario + 2, size=50)
            
            # El NDVI cae en esos mismos focos a medida que pasan los días
            ndvi[focos] -= (dia_idx * 0.02)
            
            df_dia = pd.DataFrame({
                'ID_POLIGONO': gdf_malla['ID_POLIGONO'],
                'FECHA_DIA': fecha,
                'PROBABILIDAD_COLAPSO': probs,
                'NDVI_ACTUAL': ndvi,
                'ESTRES_HIDRICO': np.random.uniform(0, 1, size=len(gdf_malla))
            })
            resultados.append(df_dia)

        df_final = pd.concat(resultados, ignore_index=True)
        
        def categorizar(prob):
            if prob > 80: return 'Crítico'
            elif prob > 50: return 'Alerta Alta'
            elif prob > 30: return 'Vigilancia'
            else: return 'Selva Sana'

        df_final['NIVEL_RIESGO'] = df_final['PROBABILIDAD_COLAPSO'].apply(categorizar)
        ruta_csv = os.path.join(self.dir_salida, "01_Reporte_Riesgo_Regional.csv")
        df_final.to_csv(ruta_csv, index=False)
        print(f"✅ Secuencia ejecutiva diaria (15 días) exportada a {ruta_csv}")

if __name__ == "__main__":
    AlertaTempranaStakeholders().ejecutar()