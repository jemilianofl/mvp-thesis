import os
import glob
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import ee
import warnings

warnings.filterwarnings("ignore")

class InyeccionPotencialPipeline:
    def __init__(self):
        self.project_id = "flash-physics-471715-s8"
        try:
            ee.Initialize(project=self.project_id)
        except Exception:
            ee.Authenticate(auth_mode='localhost')
            ee.Initialize(project=self.project_id)

        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.dir_entrada = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "03_Model_Ready", "tensores_filtrados"))
        self.dir_salida = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "03_Model_Ready", "tensores_finales"))
        
        # FIX: Leer la malla regional gigante, no solo los 38 originales
        self.ruta_poligonos = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "malla_regional_yucatan.geojson"))
        self.ruta_edafo = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "02_Procesados", "capas_inegi", "edafologia_peninsula.geojson"))
        self.reports_folder = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "reportes_metodologia"))
        
        os.makedirs(self.dir_salida, exist_ok=True)
        # AUTO-LIMPIEZA: Elimina archivos de corridas anteriores para evitar mezclas
        try:
            [os.remove(f) for f in glob.glob(os.path.join(self.dir_salida, "*.parquet"))]
        except Exception:
            pass
            
        self.metadatos_poligonos = {}

    def _preprocesar_datos_espaciales(self):
        print("🗺️ Preparando cruce espacial con INEGI y Topografía...")
        gdf_pol = gpd.read_file(self.ruta_poligonos)
        
        if os.path.exists(self.ruta_edafo):
            gdf_edafo = gpd.read_file(self.ruta_edafo)
            posibles_nombres = ['DOMINANTE', 'DESC_TIPO', 'TIPO_SUELO', 'SUELO', 'TIPO', 'CLASE', 'TIPO_PRIM', 'TIPO_SUELO_']
            col_suelo = next((c for c in posibles_nombres if c in gdf_edafo.columns), None)
            if not col_suelo:
                cols_texto = gdf_edafo.select_dtypes(include=['object']).columns
                col_suelo = cols_texto[0] if len(cols_texto) > 0 else None
                
            gdf_pol = gdf_pol.to_crs(gdf_edafo.crs)
            gdf_centroides = gdf_pol.copy()
            gdf_centroides.geometry = gdf_centroides.centroid
            cruce = gpd.sjoin(gdf_centroides, gdf_edafo, how="left", predicate="intersects")
        else:
            cruce = gdf_pol.copy()
            col_suelo = None
        
        print("   ► Obteniendo Elevación Masiva de GEE (Modo Turbo 5s)...")
        # Modo Turbo: Extraer todo GEE en 1 llamada en vez de 8000 bucles lentos
        features = []
        for idx, row in cruce.iterrows():
            pol_id = str(row['ID_POLIGONO']) if 'ID_POLIGONO' in row else str(idx)
            coords = [row.geometry.x, row.geometry.y]
            features.append(ee.Feature(ee.Geometry.Point(coords), {'ID_POLIGONO': pol_id}))
            
        ee_fc = ee.FeatureCollection(features)
        dem = ee.Image('USGS/SRTMGL1_003')
        stats = dem.reduceRegions(collection=ee_fc, reducer=ee.Reducer.mean(), scale=30)
        
        try:
            resultados_elevacion = stats.getInfo().get('features', [])
            diccionario_elev = {str(f['properties']['ID_POLIGONO']): f['properties'].get('mean', 0) for f in resultados_elevacion}
        except Exception:
            diccionario_elev = {}
        
        for idx, row in cruce.iterrows():
            pol_id = str(row['ID_POLIGONO']) if 'ID_POLIGONO' in row else str(idx)
            tipo_suelo = str(row[col_suelo]) if col_suelo and pd.notna(row.get(col_suelo)) else 'Desconocido'
            elevacion = diccionario_elev.get(pol_id, 0)
                
            self.metadatos_poligonos[pol_id] = {
                'TIPO_SUELO': tipo_suelo,
                'ELEVACION': elevacion
            }

    def inyectar_variables(self):
        print("\n⚙️ [ETAPA 1] Inyectando Potencial Espacial a Tensores...")
        self._preprocesar_datos_espaciales()
        
        archivos = glob.glob(os.path.join(self.dir_entrada, "*.parquet"))
        if not archivos: return False
        
        exitos = 0
        tipos_suelo_registrados = []
        
        for archivo in tqdm(archivos, desc="Inyectando a Parquets"):
            pol_id = os.path.basename(archivo).replace("tensor_modelo_", "").replace(".parquet", "")
            df = pd.read_parquet(archivo)
            
            if df.empty or pol_id not in self.metadatos_poligonos: continue
            
            df['ELEVACION'] = self.metadatos_poligonos[pol_id]['ELEVACION']
            df['TIPO_SUELO'] = self.metadatos_poligonos[pol_id]['TIPO_SUELO']
            tipos_suelo_registrados.append(self.metadatos_poligonos[pol_id]['TIPO_SUELO'])
            
            nombre_salida = os.path.basename(archivo).replace("tensor_modelo_", "tensor_final_")
            df.to_parquet(os.path.join(self.dir_salida, nombre_salida), index=False)
            exitos += 1
            
        self.suelos_df = pd.DataFrame({'TIPO_SUELO': tipos_suelo_registrados})
        print(f"   ✅ {exitos} Tensores Finales generados.")
        return True

    def generar_reporte_metodologico(self):
        pass 

    def ejecutar(self):
        if self.inyectar_variables():
            self.generar_reporte_metodologico()

if __name__ == "__main__":
    pipeline = InyeccionPotencialPipeline()
    pipeline.ejecutar()