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
    """
    Pipeline MLOps para inyectar variables estáticas (Espaciales y Topográficas).
    Cruza los polígonos con capas de INEGI (Edafología) y extrae el DEM de GEE.
    """
    
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
        self.ruta_poligonos = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "poligonos_maestros.geojson"))
        self.ruta_edafo = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "02_Procesados", "capas_inegi", "edafologia_peninsula.geojson"))
        self.reports_folder = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "reportes_metodologia"))
        
        os.makedirs(self.dir_salida, exist_ok=True)
        self.metadatos_poligonos = {}

    def _preprocesar_datos_espaciales(self):
        print("🗺️ Preparando cruce espacial con INEGI y Topografía...")
        gdf_pol = gpd.read_file(self.ruta_poligonos)
        gdf_edafo = gpd.read_file(self.ruta_edafo)
        
        # Búsqueda dinámica de la columna de INEGI
        print(f"   ► Columnas del INEGI detectadas: {list(gdf_edafo.columns)}")
        posibles_nombres = ['DOMINANTE', 'DESC_TIPO', 'TIPO_SUELO', 'SUELO', 'TIPO', 'CLASE', 'TIPO_PRIM', 'TIPO_SUELO_']
        col_suelo = next((c for c in posibles_nombres if c in gdf_edafo.columns), None)
        
        if not col_suelo:
            # Si no encuentra ninguna de la lista, toma la primera columna de texto que tenga INEGI
            cols_texto = gdf_edafo.select_dtypes(include=['object']).columns
            col_suelo = cols_texto[0] if len(cols_texto) > 0 else None
            
        print(f"   ► Usando columna '{col_suelo}' para Edafología.")
        
        gdf_pol = gdf_pol.to_crs(gdf_edafo.crs)
        gdf_centroides = gdf_pol.copy()
        gdf_centroides.geometry = gdf_centroides.centroid
        
        cruce = gpd.sjoin(gdf_centroides, gdf_edafo, how="left", predicate="intersects")
        dem = ee.Image('USGS/SRTMGL1_003')
        
        for idx, row in tqdm(cruce.iterrows(), total=len(cruce), desc="Extrayendo Topografía"):
            pol_id = str(row['ID_POLIGONO']) if 'ID_POLIGONO' in row else str(idx)
            
            # Extracción a prueba de errores
            tipo_suelo = str(row[col_suelo]) if col_suelo and pd.notna(row[col_suelo]) else 'Desconocido'
            
            punto_ee = ee.Geometry.Point([row.geometry.x, row.geometry.y])
            try:
                elevacion = dem.reduceRegion(reducer=ee.Reducer.mean(), geometry=punto_ee, scale=30).getInfo().get('elevation', 0)
            except Exception:
                elevacion = 0
                
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
        print("\n📊 [ETAPA 2] Generando Gráfica de Distribución Edafológica...")
        if not hasattr(self, 'suelos_df') or self.suelos_df.empty: return
        
        plt.rcParams.update(plt.rcParamsDefault)
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        conteo = self.suelos_df['TIPO_SUELO'].value_counts().head(7)
        
        sns.barplot(x=conteo.values, y=conteo.index, palette='viridis', edgecolor='black', ax=ax)
        
        ax.set_title('Distribución Edafológica en Polígonos de Entrenamiento (INEGI)', fontsize=14, fontweight='bold', pad=15, color='black')
        ax.set_xlabel('Cantidad de Polígonos de Selva', fontsize=12, color='black')
        ax.set_ylabel('Tipo de Suelo Dominante', fontsize=12, color='black')
        ax.tick_params(colors='black', labelsize=11)
        
        ruta_grafica13 = os.path.join(self.reports_folder, "13_Distribucion_Edafologica.png")
        plt.savefig(ruta_grafica13, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   ✅ Gráfica de suelos exportada a: {self.reports_folder}")

    def ejecutar(self):
        if self.inyectar_variables():
            self.generar_reporte_metodologico()

if __name__ == "__main__":
    pipeline = InyeccionPotencialPipeline()
    pipeline.ejecutar()