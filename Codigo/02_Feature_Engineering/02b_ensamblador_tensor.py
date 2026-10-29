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
    Cruza las series temporales de Google Earth Engine con 
    las agregaciones climáticas terrestres de la estación más cercana.
    """
    
    def __init__(self):
        load_dotenv()
        self.db_uri = os.getenv("DB_CONNECTION_STRING", "postgresql://postgres:123@localhost:5432/climate_data")
        self.engine = create_engine(self.db_uri, pool_pre_ping=True)
        
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.ruta_poligonos = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "malla_regional_yucatan.geojson"))
        self.dir_gee = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "02_Procesados", "gee_parquet"))
        self.dir_salida = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "02_Procesados", "tensor_maestro"))
        self.reports_folder = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "reportes_metodologia"))
        
        os.makedirs(self.dir_salida, exist_ok=True)
        os.makedirs(self.reports_folder, exist_ok=True)

    def _obtener_clima_todas_estaciones(self):
        """Descarga y agrega el clima de todas las estaciones de un solo golpe (Cero saturación de BD)."""
        query = text('SELECT "ESTACION", "FECHA", "TMAX_AUM", "TMIN_AUM", "PRECIP_AUM" FROM lecturas_aumentadas')
        with self.engine.connect() as conn:
            df_clima = pd.read_sql(query, conn)
            
        if df_clima.empty:
            return {}
            
        df_clima['FECHA_MES'] = pd.to_datetime(df_clima['FECHA']).dt.strftime('%Y-%m')
        
        clima_mensual = df_clima.groupby(['ESTACION', 'FECHA_MES']).agg({
            'TMAX_AUM': 'mean',
            'TMIN_AUM': 'mean',
            'PRECIP_AUM': 'sum'
        }).reset_index()
        
        clima_mensual.rename(columns={
            'TMAX_AUM': 'TMAX_MEDIAN', 
            'TMIN_AUM': 'TMIN_MEDIAN', 
            'PRECIP_AUM': 'PRECIP_SUM_MES'
        }, inplace=True)
        
        # Generar diccionario en RAM para acceso instantáneo
        return {est: df_grupo.drop(columns=['ESTACION']) for est, df_grupo in clima_mensual.groupby('ESTACION')}

    def ensamblar_tensores(self):
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
            
        arbol_estaciones = cKDTree(df_est[['LONGITUD', 'LATITUD']].values)
        archivos_gee = glob.glob(os.path.join(self.dir_gee, "gee_poligono_*.parquet"))
        print(f"   ► Procesando {len(archivos_gee)} series temporales de satélite...")
        
        # Cargar clima a la RAM
        diccionario_clima = self._obtener_clima_todas_estaciones()
        
        exitos = 0
        for archivo in tqdm(archivos_gee, desc="Fusionando Datos"):
            pol_id = os.path.basename(archivo).replace("gee_poligono_", "").replace(".parquet", "")
            
            df_satelite = pd.read_parquet(archivo)
            if df_satelite.empty: continue
            
            # Autocompletar FECHA_MES si el tensor original venía en formato diario
            if 'FECHA_MES' not in df_satelite.columns:
                col_fecha = 'FECHA' if 'FECHA' in df_satelite.columns else df_satelite.columns[0]
                df_satelite['FECHA_MES'] = pd.to_datetime(df_satelite[col_fecha]).dt.strftime('%Y-%m')
            
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
            
            df_clima_mensual = diccionario_clima.get(estacion_asignada, pd.DataFrame())
            
            if df_clima_mensual.empty: 
                # Si no hay clima, creamos un dataframe vacío con la columna FECHA_MES para que el left join no falle
                df_clima_mensual = pd.DataFrame(columns=['FECHA_MES', 'TMAX_MEDIAN', 'TMIN_MEDIAN', 'PRECIP_SUM_MES'])
                
            # FIX: Left Join para asegurar que nunca se borren los polígonos de GEE
            df_tensor = pd.merge(df_satelite, df_clima_mensual, on='FECHA_MES', how='left')
            df_tensor['ESTACION_REF'] = estacion_asignada
            
            # FIX: Rellenar con 0 los datos climáticos faltantes para no romper la memoria LSTM
            cols_clima = ['TMAX_MEDIAN', 'TMIN_MEDIAN', 'PRECIP_SUM_MES']
            for col in cols_clima:
                if col in df_tensor.columns:
                    df_tensor[col] = df_tensor[col].fillna(0)
            
            ruta_out = os.path.join(self.dir_salida, f"tensor_maestro_{pol_id}.parquet")
            df_tensor.to_parquet(ruta_out, index=False)
            exitos += 1
            
        print(f"\n   ✅ Ensamble completado. {exitos} Tensores Maestros generados.")
        return True

    def generar_reporte_metodologico(self):
        print("\n📊 [ETAPA 2] Generando Gráfica de Matriz de Correlación...")
        
        archivos_tensor = glob.glob(os.path.join(self.dir_salida, "tensor_maestro_*.parquet"))
        if not archivos_tensor: return
        
        df_list = [pd.read_parquet(f) for f in archivos_tensor[:50]]
        df_master = pd.concat(df_list, ignore_index=True)
        
        variables_analisis = ['NDVI', 'BSI', 'GCI', 'RADAR_VH', 'RADAR_VV', 'TMAX_MEDIAN', 'PRECIP_SUM_MES']
        columnas_presentes = [col for col in variables_analisis if col in df_master.columns]
        df_analisis = df_master[columnas_presentes].dropna()
        
        if df_analisis.empty: return
        
        matriz_corr = df_analisis.corr()
        
        plt.style.use('default')
        sns.set_theme(style="white") 
        
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = sns.color_palette("vlag", as_cmap=True)
        
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