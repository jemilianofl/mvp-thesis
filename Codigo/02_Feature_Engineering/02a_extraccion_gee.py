import glob
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import ee
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

class EarthEnginePipeline:
    def __init__(self):
        self.project_id = "flash-physics-471715-s8"
        try:
            ee.Initialize(project=self.project_id)
            print("🌍 Google Earth Engine inicializado correctamente.")
        except Exception:
            print("Autenticando en Google Earth Engine...")
            ee.Authenticate(auth_mode='localhost')
            ee.Initialize(project=self.project_id)
        
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.ruta_malla = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "malla_regional_yucatan.geojson"))
        self.dir_salida = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "02_Procesados", "gee_parquet"))
        
        os.makedirs(self.dir_salida, exist_ok=True)
        
        # AUTO-LIMPIEZA: Borramos los datos temporales/rápidos de pruebas anteriores
        print("🧹 Limpiando base de datos satelital anterior...")
        try:
            [os.remove(f) for f in glob.glob(os.path.join(self.dir_salida, "*.parquet"))]
        except Exception:
            pass
            
        self.max_workers = 4 # Balance perfecto para no saturar los servidores de Google

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

    def extraer_datos_lote(self, batch_df):
        try:
            features = []
            for _, row in batch_df.iterrows():
                pol_id = str(row['ID_POLIGONO'])
                coords = list(row.geometry.exterior.coords)
                features.append(ee.Feature(ee.Geometry.Polygon([coords]), {'ID_POLIGONO': pol_id}))
            
            ee_fc = ee.FeatureCollection(features)
            datos_historicos = []
            
            # Recorremos la historia real: 2017 a 2026
            for anio in range(2017, 2027):
                meses_fin = 11 if anio == 2026 else 13
                
                for mes in range(1, meses_fin):
                    fecha_inicio_mes = f"{anio}-{mes:02d}-01"
                    mes_siguiente = mes + 1 if mes < 12 else 1
                    anio_siguiente = anio if mes < 12 else anio + 1
                    fecha_fin_mes = f"{anio_siguiente}-{mes_siguiente:02d}-01"
                    
                    landsat = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                               .filterBounds(ee_fc)
                               .filterDate(fecha_inicio_mes, fecha_fin_mes)
                               .filter(ee.Filter.lt('CLOUD_COVER', 40))
                               .map(self._calcular_indices_landsat))
                    
                    img_landsat = landsat.median()
                    stats = img_landsat.select(['NDVI', 'BSI', 'GCI']).reduceRegions(
                        collection=ee_fc, reducer=ee.Reducer.mean(), scale=30
                    )
                    
                    resultados = stats.getInfo().get('features', [])
                    df_mes = pd.DataFrame([f['properties'] for f in resultados])
                    
                    if not df_mes.empty:
                        df_mes['FECHA_MES'] = f"{anio}-{mes:02d}"
                        datos_historicos.append(df_mes)
                        
            if datos_historicos:
                df_lote = pd.concat(datos_historicos, ignore_index=True)
                for pol_id, df_pol in df_lote.groupby('ID_POLIGONO'):
                    ruta_pq = os.path.join(self.dir_salida, f"gee_poligono_{pol_id}.parquet")
                    df_pol.to_parquet(ruta_pq, index=False)
            return True
        except Exception:
            return False

    def procesar_lote_masivo(self):
        print("\n🛰️ [ETAPA 1] Descarga Histórica Masiva de GEE (2017-2026)...")
        gdf = gpd.read_file(self.ruta_malla)
        if 'ID_POLIGONO' not in gdf.columns:
            gdf['ID_POLIGONO'] = gdf.index.astype(str)
            
        lotes = [gdf.iloc[i:i + 50] for i in range(0, len(gdf), 50)]
        print(f"   ► Procesando {len(gdf)} cuadrantes en {len(lotes)} lotes. (Esto tomará tiempo)")
        
        exitos = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futuros = {executor.submit(self.extraer_datos_lote, lote): idx for idx, lote in enumerate(lotes)}
            for fut in tqdm(as_completed(futuros), total=len(lotes), desc="Descargando Historia"):
                if fut.result(): exitos += 1
                
        print(f"\n   ✅ Extracción histórica finalizada. Lotes exitosos: {exitos}/{len(lotes)}.")

    def ejecutar(self):
        self.procesar_lote_masivo()

if __name__ == "__main__":
    pipeline = EarthEnginePipeline()
    pipeline.ejecutar()