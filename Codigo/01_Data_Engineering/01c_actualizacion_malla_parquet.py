import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as ctx
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pyproj import Transformer
from scipy.spatial import cKDTree
import warnings

warnings.filterwarnings("ignore")

class ParquetGridUpdaterPipeline:
    """
    Pipeline MLOps para la generación y actualización de la Malla Espacial Continua (Tensores Parquet).
    Utiliza K-D Trees para asignar la climatología terrestre interpolada a cada píxel de la selva.
    """
    
    def __init__(self):
        load_dotenv()
        self.db_uri = os.getenv("DB_CONNECTION_STRING", "postgresql://postgres:123@localhost:5432/climate_data")
        self.engine = create_engine(self.db_uri, pool_pre_ping=True)
        
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.ruta_parquets = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "02_Procesados", "clima_parquet"))
        self.reports_folder = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "reportes_metodologia"))
        
        os.makedirs(self.ruta_parquets, exist_ok=True)
        os.makedirs(self.reports_folder, exist_ok=True)
        
        self.transformador = Transformer.from_crs("epsg:4326", "epsg:32616", always_xy=True)
        self.patron_archivo = re.compile(r"clima_(.+)_(\d{8})\.parquet")

    def _descubrir_estado_actual(self):
        archivos = glob.glob(os.path.join(self.ruta_parquets, "**", "*.parquet"), recursive=True)
        ultimo_archivo_por_estado = {}
        for archivo in archivos:
            nombre = os.path.basename(archivo)
            match = self.patron_archivo.match(nombre)
            if match:
                estado = match.group(1)
                fecha_dt = pd.to_datetime(match.group(2), format='%Y%m%d')
                if estado not in ultimo_archivo_por_estado or fecha_dt > ultimo_archivo_por_estado[estado]['fecha']:
                    ultimo_archivo_por_estado[estado] = {'fecha': fecha_dt, 'ruta': archivo}
        return ultimo_archivo_por_estado

    def _cargar_climatologia_reciente(self, fecha_corte_global):
        query_estaciones = text('SELECT "ESTACION", "LATITUD", "LONGITUD" FROM estaciones')
        query_lecturas = text("""
            SELECT "ESTACION", "FECHA" AS "FECHA_DT", "TMAX_AUM" AS "TMAX", 
                   "TMIN_AUM" AS "TMIN", "PRECIP_AUM" AS "PRECIP" 
            FROM lecturas_aumentadas WHERE "FECHA" > :fecha
        """)
        with self.engine.connect() as conn:
            df_estaciones = pd.read_sql(query_estaciones, conn)
            df_lecturas = pd.read_sql(query_lecturas, conn, params={"fecha": fecha_corte_global})
            
        if df_lecturas.empty: return None, None
            
        df_lecturas['FECHA_DT'] = pd.to_datetime(df_lecturas['FECHA_DT'])
        utm_x, utm_y = self.transformador.transform(df_estaciones['LONGITUD'].values, df_estaciones['LATITUD'].values)
        df_estaciones['UTM_X'] = utm_x
        df_estaciones['UTM_Y'] = utm_y
        return df_estaciones, df_lecturas

    def generar_mallas_diarias(self):
        print("\n🗺️ [ETAPA 1] Sincronizando Malla Espacial con Base de Datos Híbrida...")
        ultimo_por_estado = self._descubrir_estado_actual()
        if not ultimo_por_estado: return False
            
        fecha_corte_minima = min([datos['fecha'] for datos in ultimo_por_estado.values()]).strftime('%Y-%m-%d')
        df_estaciones, df_lecturas_nuevas = self._cargar_climatologia_reciente(fecha_corte_minima)
        
        if df_lecturas_nuevas is None or df_lecturas_nuevas.empty:
            print("   ✅ La malla Parquet ya está 100% sincronizada hasta el día de hoy.")
            return True
            
        arbol_espacial = cKDTree(df_estaciones[['UTM_X', 'UTM_Y']].values)
        estaciones_ids = df_estaciones['ESTACION'].values
        
        for estado, info in ultimo_por_estado.items():
            fecha_estado = info['fecha']
            archivo_base = info['ruta']
            lecturas_faltantes = df_lecturas_nuevas[df_lecturas_nuevas['FECHA_DT'] > fecha_estado]
            dias_a_generar = sorted(lecturas_faltantes['FECHA_DT'].unique())
            
            if len(dias_a_generar) == 0: continue
            print(f"   🚀 {estado}: Ensamblando {len(dias_a_generar)} tensores diarios nuevos...")
            
            df_malla = pd.read_parquet(archivo_base, columns=['UTM_X', 'UTM_Y', 'ELEVACION', 'ESTADO']).drop_duplicates()
            _, indices = arbol_espacial.query(df_malla[['UTM_X', 'UTM_Y']].values, k=1)
            df_malla['ESTACION'] = estaciones_ids[indices]
            carpeta_destino = os.path.dirname(archivo_base)
            
            for dia_dt in dias_a_generar:
                dia_ts = pd.Timestamp(dia_dt)
                lecturas_del_dia = lecturas_faltantes[lecturas_faltantes['FECHA_DT'] == dia_ts]
                df_diario = df_malla.merge(lecturas_del_dia, on='ESTACION', how='inner')
                df_diario['FECHA'] = int(dia_ts.timestamp() * 1000)
                df_final = df_diario[['FECHA', 'ESTADO', 'UTM_X', 'UTM_Y', 'ELEVACION', 'TMAX', 'TMIN', 'PRECIP']]
                df_final.to_parquet(os.path.join(carpeta_destino, f"clima_{estado}_{dia_ts.strftime('%Y%m%d')}.parquet"), index=False)
        return True

    def generar_reporte_metodologico(self):
        """Genera el Mapa de Zonas de Influencia interpolando un plano denso."""
        print("\n📊 [ETAPA 2] Generando Mapa de Topología Espacial (Voronoi K-D Tree)...")
        with self.engine.connect() as conn:
            df_est = pd.read_sql('SELECT "ESTACION", "LATITUD", "LONGITUD" FROM estaciones', conn)
        if df_est.empty: return

        # 1. Transformar a coordenadas UTM y Mercator
        utm_x, utm_y = self.transformador.transform(df_est['LONGITUD'].values, df_est['LATITUD'].values)
        df_est['UTM_X'] = utm_x
        df_est['UTM_Y'] = utm_y
        
        trans_merc = Transformer.from_crs("epsg:32616", "epsg:3857", always_xy=True)
        merc_x, merc_y = trans_merc.transform(utm_x, utm_y)
        df_est['MERC_X'] = merc_x
        df_est['MERC_Y'] = merc_y

        # 2. Generar malla densa para colorear zonas matemáticas exactas
        min_x, max_x = df_est['UTM_X'].min() - 50000, df_est['UTM_X'].max() + 50000
        min_y, max_y = df_est['UTM_Y'].min() - 50000, df_est['UTM_Y'].max() + 50000
        
        xx, yy = np.meshgrid(np.linspace(min_x, max_x, 400), np.linspace(min_y, max_y, 400))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # 3. K-D Tree sobre la malla densa
        arbol = cKDTree(df_est[['UTM_X', 'UTM_Y']].values)
        _, indices = arbol.query(grid_points, k=1)
        zz = indices.reshape(xx.shape)
        
        xx_merc, yy_merc = trans_merc.transform(xx, yy)

        # 4. Dibujar
        plt.rcParams.update(plt.rcParamsDefault) # Limpiar cualquier caché de estilos oscuros
        plt.style.use('default')
        sns.set_theme(style="white")
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        
        # Mapa de Calor de Zonas (Voronoi)
        ax.pcolormesh(xx_merc, yy_merc, zz, cmap='tab20', alpha=0.4, shading='auto')
        
        # Estaciones Físicas
        ax.scatter(df_est['MERC_X'], df_est['MERC_Y'], c='#2c3e50', edgecolor='white', marker='^', s=100, label='Estaciones CONAGUA', zorder=5)
        
        try:
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldGrayCanvas)
        except Exception:
            pass
            
        plt.title('Zonificación Climática por Área de Influencia (Algoritmo K-D Tree)', fontsize=14, fontweight='bold', pad=15)
        ax.set_axis_off()
        plt.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='black')
        
        ruta_grafica5 = os.path.join(self.reports_folder, "05_Zonificacion_Espacial_KDTree.png")
        plt.savefig(ruta_grafica5, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   ✅ Mapa topológico de Voronoi exportado a: {self.reports_folder}")

    def ejecutar(self):
        if self.generar_mallas_diarias():
            self.generar_reporte_metodologico()

if __name__ == "__main__":
    pipeline = ParquetGridUpdaterPipeline()
    pipeline.ejecutar()