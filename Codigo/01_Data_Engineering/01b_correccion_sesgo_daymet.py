import os
import pandas as pd
import numpy as np
import pydaymet as daymet
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime
import warnings

# Ignorar advertencias de librerías para mantener limpia la consola
warnings.filterwarnings("ignore")

class DaymetBiasCorrectionPipeline:
    """
    Pipeline MLOps para la inyección de datos satelitales (Daymet) 
    y corrección de sesgo climático basado en la historia física de CONAGUA.
    Enfocado exclusivamente en TMAX, TMIN y PRECIP.
    """
    
    def __init__(self):
        # Configuración de Entorno y Base de Datos
        load_dotenv()
        self.db_uri = os.getenv("DB_CONNECTION_STRING", "postgresql://postgres:123@localhost:5432/climate_data")
        self.engine = create_engine(self.db_uri, pool_pre_ping=True)
        
        # Parámetros Temporales
        self.fecha_base_historica = "2000-01-01"
        self.hoy = datetime.now().strftime('%Y-%m-%d')
        
        # Estructura de Directorios
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.reports_folder = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "reportes_metodologia"))
        os.makedirs(self.reports_folder, exist_ok=True)

    def configurar_bd(self):
        """Asegura que la tabla de datos aumentados exista en PostgreSQL."""
        with self.engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS lecturas_aumentadas (
                    "ESTACION" TEXT,
                    "FECHA" DATE,
                    "TMAX_AUM" FLOAT,
                    "TMIN_AUM" FLOAT,
                    "PRECIP_AUM" FLOAT,
                    "ORIGEN" TEXT,
                    UNIQUE ("ESTACION", "FECHA")
                );
            """))

    def _limpiar_columnas_daymet(self, df):
        """Estandariza los nombres y formatos que envía la API satelital."""
        df.columns = [str(c).lower() for c in df.columns]
        
        posibles_fechas = ['time', 'date', 'index', 'datetime']
        col_fecha = next((c for c in posibles_fechas if c in df.columns), df.columns[0])
        df.rename(columns={col_fecha: 'FECHA'}, inplace=True)
        df['FECHA'] = pd.to_datetime(df['FECHA']).dt.tz_localize(None)
        
        # Extraemos Temperaturas y Lluvia
        for var in ['tmax', 'tmin', 'prcp']:
            if var not in df.columns:
                col_alternativa = next((c for c in df.columns if var in c), None)
                if col_alternativa:
                    df.rename(columns={col_alternativa: var}, inplace=True)
                else:
                    df[var] = np.nan 
        return df

    def _calcular_sesgo_historico(self, estacion_id):
        """Extrae la historia física de la estación para calcular su firma climática."""
        query = text('SELECT "FECHA", "PRECIP", "TMAX", "TMIN" FROM lecturas WHERE "ESTACION" = :est')
        with self.engine.connect() as conn:
            df_hist = pd.read_sql(query, conn, params={"est": estacion_id})
            
        if df_hist.empty or len(df_hist) < 365:
            return None, "Crítica/Suspendida"
            
        df_hist['FECHA'] = pd.to_datetime(df_hist['FECHA'])
        df_hist['MES'] = df_hist['FECHA'].dt.month
        
        dias_totales = (df_hist['FECHA'].max() - df_hist['FECHA'].min()).days
        dias_validos = df_hist['PRECIP'].notna().sum()
        completitud = (dias_validos / dias_totales) * 100 if dias_totales > 0 else 0
        
        categoria = "Referencia" if completitud > 70.0 else "Útil" if completitud >= 20.0 else "Crítica/Suspendida"
        if categoria == "Crítica/Suspendida":
            return None, categoria

        # Promedios para la corrección de Daymet
        sesgos = {
            'TMAX_MEAN': df_hist.groupby('MES')['TMAX'].mean(),
            'TMIN_MEAN': df_hist.groupby('MES')['TMIN'].mean(),
            'PRECIP_MEAN': df_hist.groupby('MES')['PRECIP'].mean()
        }
        return sesgos, categoria

    def procesar_aumento_incremental(self):
        """Orquesta la descarga satelital, calcula el sesgo espacial y fusiona los datos."""
        print("🚀 [ETAPA 1] Iniciando Fusión Híbrida y Corrección de Sesgo (Daymet + CONAGUA)...")
        
        with self.engine.connect() as conn:
            estaciones = pd.read_sql("SELECT * FROM estaciones", conn)
            estaciones = estaciones[estaciones['ESTADO'].str.contains('YUC|CAMP|QUIN|ROO', na=False, case=False)]
            
        for _, est in estaciones.iterrows():
            est_id = est['ESTACION']
            
            with self.engine.connect() as conn:
                ultima_f = conn.execute(text('SELECT MAX("FECHA") FROM lecturas_aumentadas WHERE "ESTACION" = :est'), {"est": est_id}).scalar()
                
            fecha_inicio = (pd.to_datetime(ultima_f) + pd.Timedelta(days=1)).strftime('%Y-%m-%d') if ultima_f else self.fecha_base_historica
            if fecha_inicio > self.hoy: continue
            
            print(f"\n📍 {est['NOMBRE']} | Sincronizando TMAX, TMIN y PRECIP desde {fecha_inicio}")
            coords = (est['LONGITUD'], est['LATITUD'])
            
            # 1. Intentar descargar satélite Daymet
            try:
                df_daymet = daymet.get_bycoords(coords, dates=(fecha_inicio, self.hoy), variables=["tmax", "tmin", "prcp"]).reset_index()
                df_daymet = self._limpiar_columnas_daymet(df_daymet)
            except Exception:
                df_daymet = pd.DataFrame(columns=['FECHA', 'tmax', 'tmin', 'prcp'])
                df_daymet['FECHA'] = pd.to_datetime(df_daymet['FECHA']) 

            # 2. Extraer lecturas terrestres
            query_conagua = text('SELECT "FECHA", "PRECIP", "TMAX", "TMIN" FROM lecturas WHERE "ESTACION" = :est AND "FECHA" >= :f_ini')
            with self.engine.connect() as conn:
                df_conagua = pd.read_sql(query_conagua, conn, params={"est": est_id, "f_ini": fecha_inicio})
                df_conagua['FECHA'] = pd.to_datetime(df_conagua['FECHA'])

            # 3. Fusión Híbrida Segura
            df_fusion = pd.merge(df_daymet, df_conagua, on="FECHA", how="outer")
            df_fusion['MES'] = df_fusion['FECHA'].dt.month
            
            sesgos_hist, categoria = self._calcular_sesgo_historico(est_id)
            
            if categoria == "Crítica/Suspendida" or not sesgos_hist:
                df_fusion['TMAX_AUM'] = df_fusion['TMAX'].fillna(df_fusion['tmax'])
                df_fusion['TMIN_AUM'] = df_fusion['TMIN'].fillna(df_fusion['tmin'])
                df_fusion['PRECIP_AUM'] = df_fusion['PRECIP'].fillna(df_fusion['prcp'])
                df_fusion['ORIGEN'] = np.where(df_fusion['TMAX'].notna(), 'CONAGUA_DIRECTO', 'DAYMET_PURO')
            else:
                try:
                    # Corrección Satelital (Temperaturas y Lluvia)
                    daymet_hist = daymet.get_bycoords(coords, dates=(self.fecha_base_historica, "2020-12-31"), variables=["tmax", "tmin", "prcp"]).reset_index()
                    daymet_hist = self._limpiar_columnas_daymet(daymet_hist)
                    daymet_hist['MES'] = daymet_hist['FECHA'].dt.month
                    
                    b_tmax = sesgos_hist['TMAX_MEAN'] - daymet_hist.groupby('MES')['tmax'].mean()
                    b_tmin = sesgos_hist['TMIN_MEAN'] - daymet_hist.groupby('MES')['tmin'].mean()
                    b_prcp = (sesgos_hist['PRECIP_MEAN'] / daymet_hist.groupby('MES')['prcp'].mean().replace(0, 0.01)).clip(upper=5.0)

                    df_fusion['TMAX_AUM'] = df_fusion['TMAX'].fillna(df_fusion.apply(lambda r: r['tmax'] + b_tmax.get(r['MES'], 0), axis=1))
                    df_fusion['TMIN_AUM'] = df_fusion['TMIN'].fillna(df_fusion.apply(lambda r: r['tmin'] + b_tmin.get(r['MES'], 0), axis=1))
                    df_fusion['PRECIP_AUM'] = df_fusion['PRECIP'].fillna(df_fusion.apply(lambda r: r['prcp'] * b_prcp.get(r['MES'], 1), axis=1))
                    df_fusion['ORIGEN'] = np.where(df_fusion['TMAX'].notna(), 'CONAGUA', 'DAYMET_CORREGIDO')
                except Exception:
                    df_fusion['TMAX_AUM'] = df_fusion['TMAX']
                    df_fusion['TMIN_AUM'] = df_fusion['TMIN']
                    df_fusion['PRECIP_AUM'] = df_fusion['PRECIP']
                    df_fusion['ORIGEN'] = 'CONAGUA_DIRECTO'

            # 4. Limpieza final y UPSERT
            df_bd = df_fusion.dropna(subset=['TMAX_AUM', 'TMIN_AUM', 'PRECIP_AUM'], how='all')
            columnas_finales = ['FECHA', 'TMAX_AUM', 'TMIN_AUM', 'PRECIP_AUM', 'ORIGEN']
            df_bd = df_bd[columnas_finales].dropna(subset=['FECHA']).copy()
            df_bd['ESTACION'] = est_id
            
            if df_bd.empty: continue
            
            try:
                with self.engine.begin() as conn:
                    conn.execute(text("CREATE TEMP TABLE temp_aum (LIKE lecturas_aumentadas INCLUDING ALL) ON COMMIT DROP;"))
                    df_bd.to_sql("temp_aum", conn, if_exists='append', index=False)
                    conn.execute(text("""
                        INSERT INTO lecturas_aumentadas ("ESTACION", "FECHA", "TMAX_AUM", "TMIN_AUM", "PRECIP_AUM", "ORIGEN")
                        SELECT "ESTACION", "FECHA", "TMAX_AUM", "TMIN_AUM", "PRECIP_AUM", "ORIGEN" FROM temp_aum
                        ON CONFLICT ("ESTACION", "FECHA") DO UPDATE SET
                            "TMAX_AUM" = EXCLUDED."TMAX_AUM", "TMIN_AUM" = EXCLUDED."TMIN_AUM",
                            "PRECIP_AUM" = EXCLUDED."PRECIP_AUM", "ORIGEN" = EXCLUDED."ORIGEN";
                    """))
                print(f"   ✔️ {len(df_bd)} días de clima híbrido sincronizados.")
            except Exception as e:
                print(f"   ❌ Error SQL: {e}")

    def generar_reporte_metodologico(self):
        print("\n📊 [ETAPA 2] Generando Gráficas Académicas de Validación...")
        query = text('SELECT "FECHA", "TMAX_AUM", "TMIN_AUM", "ORIGEN" FROM lecturas_aumentadas WHERE "FECHA" >= \'2000-01-01\' ORDER BY RANDOM() LIMIT 50000;')
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)
            
        if df.empty: return
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        
        plt.figure(figsize=(9, 6))
        conteo = df['ORIGEN'].value_counts()
        colores = ['#41ab5d' if 'CONAGUA' in orig else '#ef3b2c' if 'PURO' in orig else '#fd8d3c' for orig in conteo.index]
        wedges, texts, autotexts = plt.pie(conteo, labels=conteo.index, autopct='%1.1f%%', startangle=140, colors=colores, wedgeprops=dict(width=0.4, edgecolor='w'))
        plt.setp(autotexts, size=11, weight="bold", color="black")
        plt.title('Recuperación de Vacíos Históricos\n(Estación Física vs Satélite)', fontsize=14, fontweight='bold')
        plt.savefig(os.path.join(self.reports_folder, "03_Composicion_Origen_Datos.png"), dpi=300, bbox_inches='tight')
        plt.close()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        paleta = {'CONAGUA': '#41ab5d', 'DAYMET_CORREGIDO': '#fd8d3c', 'DAYMET_PURO': '#ef3b2c', 'CONAGUA_DIRECTO': '#41ab5d'}
        sns.kdeplot(data=df, x="TMAX_AUM", hue="ORIGEN", fill=True, common_norm=False, palette=paleta, alpha=0.4, linewidth=1.5, ax=ax1)
        ax1.set_title('Corrección de Sesgo (Temp. Máxima)', fontweight='bold')
        ax1.set_xlabel('Temperatura (°C)')
        sns.kdeplot(data=df, x="TMIN_AUM", hue="ORIGEN", fill=True, common_norm=False, palette=paleta, alpha=0.4, linewidth=1.5, ax=ax2)
        ax2.set_title('Corrección de Sesgo (Temp. Mínima)', fontweight='bold')
        ax2.set_xlabel('Temperatura (°C)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_folder, "04_Validacion_Sesgo_Termico_KDE.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def ejecutar(self):
        """Orquestador principal."""
        self.configurar_bd()
        self.procesar_aumento_incremental()
        self.generar_reporte_metodologico()

if __name__ == "__main__":
    pipeline = DaymetBiasCorrectionPipeline()
    pipeline.ejecutar()