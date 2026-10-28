import glob
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
from datetime import datetime
from dateutil.relativedelta import relativedelta
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")

class EarthEnginePipeline:
    def __init__(self):
        self.project_id = "flash-physics-471715-s8"
        try:
            ee.Initialize(project=self.project_id)
            print("🌍 Google Earth Engine inicializado correctamente.")
        except ee.EEException:
            print("Autenticando en Google Earth Engine...")
            ee.Authenticate(auth_mode='localhost')
            ee.Initialize(project=self.project_id)
        
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        # FIX: Ahora extraemos contra TODA LA REGIÓN, no solo las zonas de control
        self.ruta_malla = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "malla_regional_yucatan.geojson"))
        self.dir_salida = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "02_Procesados", "gee_parquet"))
        self.reports_folder = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "reportes_metodologia"))
        
        os.makedirs(self.dir_salida, exist_ok=True)
        os.makedirs(self.reports_folder, exist_ok=True)
        
        self.fecha_actual = datetime.now()
        self.fecha_inicio = '2017-01-01'
        self.fecha_fin = self.fecha_actual.strftime('%Y-%m-%d')
        
        print(f"📡 Rango temporal configurado: {self.fecha_inicio} a {self.fecha_fin}")
        self.max_workers = 5

    def _calcular_indices_landsat(self, image):
        ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        bsi = image.expression(
            '((B6 + B4) - (B5 + B2)) / ((B6 + B4) + (B5 + B2))', {
                'B6': image.select('SR_B6'), 'B4': image.select('SR_B4'),
                'B5': image.select('SR_B5'), 'B2': image.select('SR_B2')
            }).rename('BSI')
        gci = image.expression(
            '(B5 / B3) - 1', {
                'B5': image.select('SR_B5'), 'B3': image.select('SR_B3')
            }).rename('GCI')
        return image.addBands([ndvi, bsi, gci])

    def extraer_datos_lote(self, batch_df, batch_idx):
        """Descarga información para un lote de la malla"""
        try:
            # Crear un FeatureCollection de este lote en GEE
            features = []
            for _, row in batch_df.iterrows():
                pol_id = str(row['ID_POLIGONO'])
                coords = list(row.geometry.exterior.coords)
                features.append(ee.Feature(ee.Geometry.Polygon([coords]), {'ID_POLIGONO': pol_id}))
            
            ee_fc = ee.FeatureCollection(features)
            
            landsat = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                       .filterBounds(ee_fc)
                       .filterDate(self.fecha_inicio, self.fecha_fin)
                       .filter(ee.Filter.lt('CLOUD_COVER', 30))
                       .map(self._calcular_indices_landsat))
                       
            # Reducción espacial masiva de una sola vez
            # Se usa el primer día del mes actual para una inferencia rápida y ligera
            img_landsat = landsat.median()
            stats = img_landsat.select(['NDVI', 'BSI', 'GCI']).reduceRegions(
                collection=ee_fc,
                reducer=ee.Reducer.mean(),
                scale=30
            )
            
            resultados = stats.getInfo().get('features', [])
            df_opt = pd.DataFrame([f['properties'] for f in resultados])
            
            if df_opt.empty: return False
                
            
            df_opt['FECHA_DIA'] = self.fecha_actual.strftime('%Y-%m-%d')
            
            # Guardamos cada polígono del lote en el formato que espera el resto del pipeline
            for _, row in df_opt.iterrows():
                pol_id = row['ID_POLIGONO']
                ruta_pq = os.path.join(self.dir_salida, f"gee_poligono_{pol_id}.parquet")
                
                # Combinamos con histórico si existe, si no, creamos nuevo
                df_pol = pd.DataFrame([row])
                if os.path.exists(ruta_pq):
                    try:
                        df_hist = pd.read_parquet(ruta_pq)
                        df_pol = pd.concat([df_hist, df_pol]).drop_duplicates(subset=['FECHA_DIA'], keep='last')
                    except: pass
                
                df_pol.to_parquet(ruta_pq, index=False)
                
            return True
        except Exception as e:
            return False

    def procesar_lote_masivo(self):
        print("\n🛰️ [ETAPA 1] Procesando Extracción Multiespectral desde GEE (Malla Completa)...")
        if not os.path.exists(self.ruta_malla): 
            print("❌ No se encontró la malla regional.")
            return
            
        gdf = gpd.read_file(self.ruta_malla)
        if 'ID_POLIGONO' not in gdf.columns:
            gdf['ID_POLIGONO'] = gdf.index.astype(str)
            
        # Segmentar la malla gigante en lotes manejables (ej. 1000 polígonos por lote)
        lotes = [gdf.iloc[i:i + 1000] for i in range(0, len(gdf), 1000)]
        
        print(f"   ► Conectando a servidores de Google para actualizar {len(gdf)} cuadrantes en {len(lotes)} bloques...")
        exitos = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futuros = {executor.submit(self.extraer_datos_lote, lote, idx): idx for idx, lote in enumerate(lotes)}
            for fut in tqdm(as_completed(futuros), total=len(lotes)):
                if fut.result(): exitos += 1
                
        print(f"\n   ✅ Extracción finalizada. Lotes procesados: {exitos}/{len(lotes)}.")

    def generar_reporte_metodologico(self):
        print("\n📊 [ETAPA 2] Reporte Académico Omitido en Malla Masiva para optimizar tiempo.")

    def ejecutar(self):
        self.procesar_lote_masivo()
        self.generar_reporte_metodologico()

if __name__ == "__main__":
    pipeline = EarthEnginePipeline()
    pipeline.ejecutar()