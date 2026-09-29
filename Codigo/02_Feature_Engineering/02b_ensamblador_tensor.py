import glob
import os
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from scipy.spatial import cKDTree
from sqlalchemy import create_engine, text
from tqdm import tqdm

warnings.filterwarnings("ignore")

class MasterTensorBuilderPipeline:
    """
    Pipeline MLOps para ensamblar el Tensor Maestro.
    Cruza las series temporales de Google Earth Engine (Mensuales) 
    con las agregaciones climáticas terrestres de la estación más cercana.
    """
    
    def __init__(self):
        # Base de Datos
        load_dotenv()
        self.db_uri = os.getenv("DB_CONNECTION_STRING", "postgresql://postgres:123@localhost:5432/climate_data")
        self.engine = create_engine(self.db_uri, pool_pre_ping=True)
        
        # Rutas de Archivos
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.ruta_poligonos = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "poligonos_maestros.geojson"))
        self.dir_gee = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "02_Procesados", "gee_parquet"))
        self.dir_salida = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "02_Procesados", "tensor_maestro"))
        self.reports_folder = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "reportes_metodologia"))
        
        os.makedirs(self.dir_salida, exist_ok=True)
        os.makedirs(self.reports_folder, exist_ok=True)

    def _obtener_clima_mensual_estacion(self, estacion_id):
        """Agrega los datos diarios de una estación a promedios/sumas mensuales."""
        query = text("""
            SELECT "FECHA", "TMAX_AUM", "TMIN_AUM", "PRECIP_AUM" 
            FROM lecturas_aumentadas 
            WHERE "ESTACION" = :est
        """)
        
        with self.engine.connect() as conn:
            df_clima = pd.read_sql(query, conn, params={"est": estacion_id})
            
        if df_clima.empty:
            return pd.DataFrame()
            
        df_clima['FECHA_MES'] = pd.to_datetime(df_clima['FECHA']).dt.strftime('%Y-%m')
        
        # Agregación climática: Promedio de TMAX/TMIN, Suma Acumulada de Lluvia
        clima_mensual = df_clima.groupby('FECHA_MES').agg({
            'TMAX_AUM': 'mean',
            'TMIN_AUM': 'mean',
            'PRECIP_AUM': 'sum'
        }).reset_index()
        
        clima_mensual.rename(columns={
            'TMAX_AUM': 'TMAX_MEDIAN', 
            'TMIN_AUM': 'TMIN_MEDIAN', 
            'PRECIP_AUM': 'PRECIP_SUM_MES'
        }, inplace=True)
        
        return clima_mensual

    def ensamblar_tensores(self):
        """Encuentra la estación más cercana a cada polígono y cruza GEE con Clima."""
        print("\n⚙️ [ETAPA 1] Ensamblando Tensores Maestros (Satélite + Clima)...")
        
        if not os.path.exists(self.ruta_poligonos):
            print("❌ Archivo de polígonos no encontrado.")
            return False
            
        gdf_poligonos = gpd.read_file(self.ruta_poligonos)
        
        with self.engine.connect() as conn:
            df_est = pd.read_sql('SELECT "ESTACION", "LATITUD", "LONGITUD" FROM estaciones', conn)
            
        if df_est.empty:
            print("❌ No hay estaciones en la base de datos.")
            return False
            
        # Asignación espacial ultra rápida
        arbol_estaciones = cKDTree(df_est[['LONGITUD', 'LATITUD']].values)
        archivos_gee = glob.glob(os.path.join(self.dir_gee, "gee_poligono_*.parquet"))
        print(f"   ► Procesando {len(archivos_gee)} series temporales de satélite...")
        
        exitos = 0
        for archivo in tqdm(archivos_gee, desc="Fusionando Datos"):
            pol_id = os.path.basename(archivo).replace("gee_poligono_", "").replace(".parquet", "")
            
            df_satelite = pd.read_parquet(archivo)
            if df_satelite.empty: continue
            
            # Centroide del polígono para buscar la estación más cercana
            try:
                if pol_id.isdigit() and int(pol_id) < len(gdf_poligonos):
                    poligono_geom = gdf_poligonos.iloc[int(pol_id)]
                else:
                    poligono_geom = gdf_poligonos[gdf_poligonos.index.astype(str) == pol_id].iloc[0]
            except Exception:
                continue
                
            centroide = poligono_geom.geometry.centroid
            _, idx_estacion = arbol_estaciones.query([[centroide.x, centroide.y]], k=1)
            estacion_asignada = df_est.iloc[idx_estacion[0]]['ESTACION']
            
            df_clima_mensual = self._obtener_clima_mensual_estacion(estacion_asignada)
            if df_clima_mensual.empty: continue
                
            # Cruce exacto en la línea de tiempo (Merge temporal)
            df_tensor = pd.merge(df_satelite, df_clima_mensual, on='FECHA_MES', how='inner', validate='one_to_one')
            df_tensor['ESTACION_REF'] = estacion_asignada
            
            ruta_out = os.path.join(self.dir_salida, f"tensor_maestro_{pol_id}.parquet")
            df_tensor.to_parquet(ruta_out, index=False)
            exitos += 1
            
        print(f"\n   ✅ Ensamble completado. {exitos} Tensores Maestros generados.")
        return True

    def generar_reporte_metodologico(self):
        """Genera una Matriz de Correlación para la Tesis (Estilo APA/Académico)."""
        print("\n📊 [ETAPA 2] Generando Gráfica de Matriz de Correlación...")
        
        archivos_tensor = glob.glob(os.path.join(self.dir_salida, "tensor_maestro_*.parquet"))
        if not archivos_tensor: return
        
        df_list = [pd.read_parquet(f) for f in archivos_tensor[:50]]
        df_master = pd.concat(df_list, ignore_index=True)
        
        # Filtrar variables críticas
        variables_analisis = ['NDVI', 'BSI', 'GCI', 'RADAR_VH', 'RADAR_VV', 'TMAX_MEDIAN', 'PRECIP_SUM_MES']
        # Validar qué columnas existen realmente antes de filtrar
        columnas_presentes = [col for col in variables_analisis if col in df_master.columns]
        df_analisis = df_master[columnas_presentes].dropna()
        
        if df_analisis.empty: return
        
        matriz_corr = df_analisis.corr()
        
        # ESTILO ESTRICTAMENTE ACADÉMICO (Fondo blanco)
        plt.style.use('default')
        sns.set_theme(style="white") 
        
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = sns.color_palette("vlag", as_cmap=True) # Paleta azul-rojo formal
        
        sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1, 
                    square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax, 
                    annot_kws={"size": 11, "weight": "bold", "color": "black"})
        
        plt.title('Matriz de Correlación Multi-Sensor\n(Óptica, Radar y Climatología)', 
                  fontsize=14, fontweight='bold', pad=20, color='black')
        
        plt.xticks(rotation=45, ha='right', fontsize=11, color='black')
        plt.yticks(rotation=0, fontsize=11, color='black')
        
        ruta_grafica8 = os.path.join(self.reports_folder, "08_Matriz_Correlacion_Tensor.png")
        plt.savefig(ruta_grafica8, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
        plt.close()
        
        print(f"   ✅ Gráfica académica exportada a: {self.reports_folder}")

    def ejecutar(self):
        if self.ensamblar_tensores():
            self.generar_reporte_metodologico()

if __name__ == "__main__":
    pipeline = MasterTensorBuilderPipeline()
    pipeline.ejecutar()