import glob
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")

class EarthEnginePipeline:
    """
    Pipeline MLOps para la extracción masiva de variables satelitales (Ópticas y Radar)
    utilizando Google Earth Engine (GEE).
    """
    
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
        self.ruta_poligonos = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "poligonos_maestros.geojson"))
        self.dir_salida = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "02_Procesados", "gee_parquet"))
        self.reports_folder = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "reportes_metodologia"))
        
        os.makedirs(self.dir_salida, exist_ok=True)
        os.makedirs(self.reports_folder, exist_ok=True)
        
        self.fecha_inicio = '2017-01-01'
        self.fecha_fin = '2026-12-31'
        self.max_workers = 3

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

    def extraer_datos_poligono(self, pol_id, bounds):
        try:
            ee_geom = ee.Geometry.Polygon([[
                [bounds[0], bounds[1]], [bounds[2], bounds[1]],
                [bounds[2], bounds[3]], [bounds[0], bounds[3]],
                [bounds[0], bounds[1]]
            ]])
            
            landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(ee_geom).filterDate(self.fecha_inicio, self.fecha_fin).filter(ee.Filter.lt('CLOUD_COVER', 30)).map(self._calcular_indices_landsat)
            sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(ee_geom).filterDate(self.fecha_inicio, self.fecha_fin).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).filter(ee.Filter.eq('instrumentMode', 'IW'))
            
            def reducir_optico(img):
                mes = ee.Date(img.get('system:time_start')).format('YYYY-MM')
                stats = img.select(['NDVI', 'BSI', 'GCI']).reduceRegion(reducer=ee.Reducer.mean(), geometry=ee_geom, scale=30, maxPixels=1e9)
                return ee.Feature(None, {'FECHA_MES': mes, 'NDVI': stats.get('NDVI'), 'BSI': stats.get('BSI'), 'GCI': stats.get('GCI')})
                
            def reducir_radar(img):
                mes = ee.Date(img.get('system:time_start')).format('YYYY-MM')
                stats = img.select(['VH', 'VV']).reduceRegion(reducer=ee.Reducer.mean(), geometry=ee_geom, scale=10, maxPixels=1e9)
                return ee.Feature(None, {'FECHA_MES': mes, 'RADAR_VH': stats.get('VH'), 'RADAR_VV': stats.get('VV')})

            fechas_opticas = landsat.map(reducir_optico).getInfo().get('features', [])
            fechas_radar = sentinel1.map(reducir_radar).getInfo().get('features', [])
            
            df_opt = pd.DataFrame([f['properties'] for f in fechas_opticas])
            df_rad = pd.DataFrame([f['properties'] for f in fechas_radar])
            
            if df_opt.empty and df_rad.empty: return False
                
            if not df_opt.empty: df_opt = df_opt.groupby('FECHA_MES').mean().reset_index()
            if not df_rad.empty: df_rad = df_rad.groupby('FECHA_MES').mean().reset_index()
            
            if df_opt.empty: df_final = df_rad
            elif df_rad.empty: df_final = df_opt
            else: df_final = pd.merge(df_opt, df_rad, on='FECHA_MES', how='outer', validate='one_to_one')
                
            df_final['ID_POLIGONO'] = str(pol_id)
            ruta_pq = os.path.join(self.dir_salida, f"gee_poligono_{pol_id}.parquet")
            df_final.to_parquet(ruta_pq, index=False)
            return True
            
        except Exception:
            return False

    def procesar_lote_masivo(self):
        print("\n🛰️ [ETAPA 1] Procesando Extracción Multiespectral y Radar desde GEE...")
        if not os.path.exists(self.ruta_poligonos): return
            
        gdf = gpd.read_file(self.ruta_poligonos)
        tareas = []
        for idx, row in gdf.iterrows():
            pol_id = str(row['ID_POLIGONO']) if 'ID_POLIGONO' in row else str(idx)
            if os.path.exists(os.path.join(self.dir_salida, f"gee_poligono_{pol_id}.parquet")): continue
            tareas.append((pol_id, row.geometry.bounds))

        if not tareas:
            print("   ✅ Todos los polígonos ya han sido extraídos previamente.")
            return
            
        print(f"   ► Conectando a servidores de Google para extraer {len(tareas)} polígonos...")
        exitos = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futuros = {executor.submit(self.extraer_datos_poligono, t[0], t[1]): t for t in tareas}
            for fut in tqdm(as_completed(futuros), total=len(tareas)):
                if fut.result(): exitos += 1
                
        print(f"\n   ✅ Extracción finalizada. {exitos} polígonos descargados.")

    def generar_reporte_metodologico(self):
        print("\n📊 [ETAPA 2] Generando Gráficas de Evidencia Académica...")
        archivos_pq = glob.glob(os.path.join(self.dir_salida, "*.parquet"))
        if not archivos_pq: return
        
        df_list = [pd.read_parquet(f) for f in archivos_pq[:20]]
        if not df_list: return
        
        df_all = pd.concat(df_list, ignore_index=True)
        df_all['FECHA_MES'] = pd.to_datetime(df_all['FECHA_MES'])
        df_all['AÑO'] = df_all['FECHA_MES'].dt.year
        
        cobertura = df_all.groupby('AÑO').agg({'NDVI': lambda x: x.notna().sum(), 'RADAR_VH': lambda x: x.notna().sum()}).reset_index()
        
        # Forzar estilos completamente en blanco (Limpieza de caché de Matplotlib)
        plt.rcParams.update(plt.rcParamsDefault)
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        
        ax.plot(cobertura['AÑO'], cobertura['NDVI'], marker='o', color='#2ca02c', linewidth=2.5, label='Observaciones Ópticas (Landsat 8 - NDVI)')
        ax.plot(cobertura['AÑO'], cobertura['RADAR_VH'], marker='s', color='#1f77b4', linewidth=2.5, linestyle='--', label='Observaciones Radar (Sentinel-1 - VH)')
        
        ax.set_title('Densidad Histórica de Observaciones Satelitales Mensuales', fontsize=14, fontweight='bold', pad=15, color='black')
        ax.set_xlabel('Año de Observación', fontsize=12, color='black')
        ax.set_ylabel('Cantidad de Observaciones Útiles', fontsize=12, color='black')
        ax.tick_params(colors='black', labelsize=10)
        
        ax.legend(fontsize=11, frameon=True, facecolor='white', edgecolor='black')
        
        ruta_grafica7 = os.path.join(self.reports_folder, "07_Cobertura_Sensores_GEE.png")
        plt.savefig(ruta_grafica7, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
        plt.close()
        print(f"   ✅ Gráfica satelital exportada a: {self.reports_folder}")

    def ejecutar(self):
        self.procesar_lote_masivo()
        self.generar_reporte_metodologico()

if __name__ == "__main__":
    pipeline = EarthEnginePipeline()
    pipeline.ejecutar()