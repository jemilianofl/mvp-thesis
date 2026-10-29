import os
import glob
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import numpy as np

warnings.filterwarnings("ignore")

class LandUseMaskingPipeline:
    def __init__(self):
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.dir_entrada = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "02_Procesados", "tensor_enriquecido"))
        self.dir_salida = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "03_Model_Ready", "tensores_filtrados"))
        self.ruta_poligonos = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "poligonos_maestros.geojson"))
        self.reports_folder = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "reportes_metodologia"))
        
        os.makedirs(self.dir_salida, exist_ok=True)
        # AUTO-LIMPIEZA: Elimina archivos de corridas anteriores para evitar mezclas
        try:
            [os.remove(f) for f in glob.glob(os.path.join(self.dir_salida, "*.parquet"))]
        except Exception:
            pass
            
        self.auditoria = {'Polígonos Totales': 0, 'Selva (Monitoreo)': 0, 'Otros Usos (Grises)': 0}

    def aplicar_mascaras(self):
        print("\n🛡️ [ETAPA 1] Aplicando Máscaras No Destructivas (Etiquetado Inteligente)...")
            
        archivos = glob.glob(os.path.join(self.dir_entrada, "*.parquet"))
        self.auditoria['Polígonos Totales'] = len(archivos)
        
        for archivo in tqdm(archivos, desc="Etiquetando Coberturas"):
            df = pd.read_parquet(archivo)
            if df.empty: continue
            
            ndvi_maximo_historico = df['NDVI'].max() if 'NDVI' in df.columns else 0
            
            # FIX: Relajamos el umbral para incluir la selva seca de Yucatán (0.30)
            if ndvi_maximo_historico >= 0.30:
                df['ES_SELVA'] = 1
                self.auditoria['Selva (Monitoreo)'] += 1
            else:
                df['ES_SELVA'] = 0
                self.auditoria['Otros Usos (Grises)'] += 1
                
            nombre_salida = os.path.basename(archivo).replace("tensor_enriquecido", "tensor_modelo")
            df.to_parquet(os.path.join(self.dir_salida, nombre_salida), index=False)
                
        print(f"   ✅ Depuración finalizada. {self.auditoria['Selva (Monitoreo)']} polígonos listos para IA, {self.auditoria['Otros Usos (Grises)']} mantenidos en monitoreo pasivo.")
        return True

    def generar_reporte_metodologico(self):
        print("\n📊 [ETAPA 2] Generando Gráfica Académica de Depuración del Dataset...")
        pass

    def ejecutar(self):
        if self.aplicar_mascaras():
            self.generar_reporte_metodologico()

if __name__ == "__main__":
    pipeline = LandUseMaskingPipeline()
    pipeline.ejecutar()