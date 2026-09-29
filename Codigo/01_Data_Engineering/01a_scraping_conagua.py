import os
import re
import glob
import time
import random
import urllib3
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sqlalchemy import create_engine, text
from tqdm import tqdm
from dotenv import load_dotenv

# Ignorar advertencias de SSL al hacer scraping
warnings = urllib3.exceptions.InsecureRequestWarning
urllib3.disable_warnings(warnings)

class ConaguaPipeline:
    """
    Pipeline MLOps para la extracción, transformación y carga (ETL) 
    de datos climatológicos del Servicio Meteorológico Nacional (SMN).
    Diseñado con POO para la Tesis de Maestría.
    """
    
    def __init__(self):
        # Configuración de Entorno y Base de Datos
        load_dotenv()
        self.db_uri = os.getenv("DB_CONNECTION_STRING", "postgresql://postgres:123@localhost:5432/climate_data")
        self.engine = create_engine(self.db_uri, pool_pre_ping=True)
        
        # Estructura de Directorios
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.data_folder = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "conagua_txt"))
        self.reports_folder = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "reportes_metodologia"))
        
        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(self.reports_folder, exist_ok=True)
        
        # Parámetros Operativos
        self.estados_interes = ["CAMP", "YUC", "QROO"]
        self.max_workers = 4
        self.batch_size = 100
        
        # Expresiones regulares precompiladas para Parsing Rápido
        self.re_lat = re.compile(r"LATITUD\s*:\s*([\d\.-]+)")
        self.re_lon = re.compile(r"LONGITUD\s*:\s*([\d\.-]+)")
        self.re_est = re.compile(r"ESTADO\s*:\s*(.+)")
        self.re_nom = re.compile(r"NOMBRE\s*:\s*(.+)")

    def configurar_bd(self):
        """Configura el esquema estricto de la base de datos (Restricciones UNIQUE y PK)."""
        with self.engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS estaciones (
                    "ESTACION" TEXT PRIMARY KEY,
                    "NOMBRE" TEXT, "ESTADO" TEXT,
                    "LATITUD" FLOAT, "LONGITUD" FLOAT, "ALTITUD" FLOAT
                );
                CREATE TABLE IF NOT EXISTS lecturas (
                    "ESTACION" TEXT REFERENCES estaciones("ESTACION"),
                    "FECHA" DATE, "PRECIP" FLOAT, "EVAP" FLOAT,
                    "TMAX" FLOAT, "TMIN" FLOAT,
                    UNIQUE ("ESTACION", "FECHA")
                );
            """))

    def _crear_sesion(self):
        """Genera una sesión HTTP tolerante a fallos para interactuar con CONAGUA."""
        session = requests.Session()
        retry = Retry(total=5, backoff_factor=1.5, status_forcelist=[404, 429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        session.verify = False
        return session

    def _descargar_archivo(self, session, tarea):
        """Hilo worker para descargar un TXT individual."""
        url, ruta = tarea
        time.sleep(random.uniform(0.1, 0.3))
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200 and 'text/html' not in resp.headers.get('Content-Type', '').lower():
                with open(ruta, 'wb') as f:
                    f.write(resp.content)
        except Exception:
            pass

    def extraer_datos_smn(self):
        """Web Scraping del catálogo de CONAGUA para descargar archivos físicos (TXT)."""
        print("\n🌐 [ETAPA 1] Scrapeando catálogo de CONAGUA...")
        base_url = "https://smn.conagua.gob.mx/tools/RESOURCES/Normales_Climatologicas/"
        session = self._crear_sesion()
        
        res = session.get("https://smn.conagua.gob.mx/es/climatologia/informacion-climatologica/normales-climatologicas-por-estado")
        soup = BeautifulSoup(res.content, 'html.parser')
        selector = soup.find('select', id='listaestados')
        
        if not selector: 
            print("❌ No se pudo encontrar la lista de estados en la página de CONAGUA.")
            return

        codigos = [opt['value'].split('=')[-1] for opt in selector.find_all('option') if opt.get('value')]
        estados = [c for c in codigos if c.upper() in self.estados_interes]

        for estado in estados:
            folder_est = os.path.join(self.data_folder, estado.upper())
            os.makedirs(folder_est, exist_ok=True)
            
            res_cat = session.get(f"{base_url}catalogo/cat_{estado}.html")
            if res_cat.status_code != 200: continue
            
            soup_cat = BeautifulSoup(res_cat.content, 'html.parser')
            tabla = soup_cat.find('table')
            if not tabla: continue

            tareas = []
            for fila in tabla.find_all('tr'):
                celdas = fila.find_all('td')
                # Excluir estaciones formalmente suspendidas
                if len(celdas) < 5 or "SUSPENDIDA" in celdas[3].get_text(strip=True).upper(): 
                    continue
                
                nom_limpio = re.sub(r'[\\/*?:"<>|]', "_", celdas[1].get_text(strip=True)).strip()
                enlace = celdas[4].find('a')
                if enlace and enlace.has_attr('href'):
                    url_abs = urljoin(base_url, enlace['href'].lstrip('../'))
                    tareas.append((url_abs, os.path.join(folder_est, f"{nom_limpio}.txt")))

            print(f"   ► Descargando {len(tareas)} estaciones activas de {estado.upper()}...")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                list(executor.map(partial(self._descargar_archivo, session), tareas))

    def _procesar_txt(self, file_path):
        """Extrae metadata del encabezado y convierte las lecturas en DataFrame."""
        try:
            encoding = 'utf-8'
            try:
                with open(file_path, 'r', encoding='utf-8') as f: f.read(100)
            except UnicodeDecodeError:
                encoding = 'latin-1'

            with open(file_path, 'r', encoding=encoding) as f:
                encabezado = [next(f) for _ in range(50)]
                head_str = "".join(encabezado)

                est_m = self.re_est.search(head_str)
                lat_m = self.re_lat.search(head_str)
                lon_m = self.re_lon.search(head_str)
                nom_m = self.re_nom.search(head_str)
                
                if not (est_m and lat_m and lon_m and nom_m): 
                    return None, None

                estado = est_m.group(1).split("LATITUD")[0].strip().upper()
                info = {
                    "NOMBRE": nom_m.group(1).split("ESTADO")[0].strip(),
                    "ESTADO": estado, 
                    "LATITUD": float(lat_m.group(1)),
                    "LONGITUD": float(lon_m.group(1)), 
                    "ALTITUD": 10.0
                }

                f.seek(0)
                start_line = next((i for i, line in enumerate(encabezado) if "FECHA" in line), -1)
                if start_line == -1: return None, None

                df = pd.read_csv(f, sep=r'\s+', skiprows=start_line + 2, 
                                 names=["FECHA", "PRECIP", "EVAP", "TMAX", "TMIN"], 
                                 encoding=encoding, engine='python', dtype=str)
                df["ESTACION"] = info["NOMBRE"]
                df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')
                df = df.dropna(subset=['FECHA'])

                for col in ["PRECIP", "EVAP", "TMAX", "TMIN"]:
                    df[col] = pd.to_numeric(df[col].replace('NULO', float('nan')), errors='coerce')

                return info, df if not df.empty else None
        except Exception:
            return None, None

    def _upsert_lote(self, metas, dfs, fecha_minima):
        """Inyecta lotes a la base de datos evitando colapsos por duplicados (UPSERT)."""
        if not metas: return
        try:
            with self.engine.begin() as conn:
                # 1. UPSERT de Estaciones
                df_m = pd.DataFrame(metas).drop_duplicates(subset=['NOMBRE'])
                for _, r in df_m.iterrows():
                    sql_est = text("""
                        INSERT INTO estaciones ("ESTACION", "NOMBRE", "ESTADO", "LATITUD", "LONGITUD", "ALTITUD")
                        VALUES (:est, :nom, :estd, :lat, :lon, :alt)
                        ON CONFLICT ("ESTACION") DO UPDATE SET
                            "LATITUD" = EXCLUDED."LATITUD", "LONGITUD" = EXCLUDED."LONGITUD";
                    """)
                    conn.execute(sql_est, {"est": r['NOMBRE'], "nom": r['NOMBRE'], "estd": r['ESTADO'], 
                                           "lat": r['LATITUD'], "lon": r['LONGITUD'], "alt": r['ALTITUD']})

                # 2. UPSERT de Lecturas (Solo fechas nuevas)
                if dfs:
                    df_l = pd.concat(dfs).drop_duplicates(subset=['ESTACION', 'FECHA'])
                    df_l = df_l[df_l['FECHA'].dt.date > fecha_minima] 
                    
                    if df_l.empty: return
                    
                    conn.execute(text("CREATE TEMP TABLE temp_lecturas (LIKE lecturas INCLUDING ALL) ON COMMIT DROP;"))
                    df_l.to_sql("temp_lecturas", conn, if_exists='append', index=False)
                    
                    sql_upsert = text("""
                        INSERT INTO lecturas ("ESTACION", "FECHA", "PRECIP", "EVAP", "TMAX", "TMIN")
                        SELECT "ESTACION", "FECHA", "PRECIP", "EVAP", "TMAX", "TMIN" FROM temp_lecturas
                        ON CONFLICT ("ESTACION", "FECHA") DO UPDATE SET
                            "PRECIP" = EXCLUDED."PRECIP", "EVAP" = EXCLUDED."EVAP",
                            "TMAX" = EXCLUDED."TMAX", "TMIN" = EXCLUDED."TMIN";
                    """)
                    conn.execute(sql_upsert)
        except Exception as e:
            print(f"❌ Error en UPSERT: {e}")

    def etl_incremental(self):
        """Lee los TXT descargados, los procesa en paralelo y alimenta la Base de Datos."""
        print("\n🔨 [ETAPA 2] Parseando TXTs y ejecutando Carga de Base de Datos...")
        
        with self.engine.connect() as conn:
            res = conn.execute(text('SELECT MAX("FECHA") FROM lecturas')).scalar()
            fecha_max_db = pd.to_datetime(res).date() if res else pd.to_datetime('1900-01-01').date()
        
        files = glob.glob(os.path.join(self.data_folder, "**", "*.txt"), recursive=True)
        batch_m, batch_d = [], []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as exc:
            for info, df in tqdm(exc.map(self._procesar_txt, files), total=len(files), desc="Insertando a BD"):
                if info:
                    batch_m.append(info)
                    if df is not None: batch_d.append(df)
                    
                    # Vaciar memoria al llegar al batch_size
                    if len(batch_m) >= self.batch_size:
                        self._upsert_lote(batch_m, batch_d, fecha_max_db)
                        batch_m, batch_d = [], []
                        
        if batch_m: 
            self._upsert_lote(batch_m, batch_d, fecha_max_db)
            
        print("✅ Base de datos climatológica actualizada con éxito.")

    def generar_reporte_metodologico(self):
        print("\n📊 [ETAPA 3] Generando Gráficas Académicas...")
        with self.engine.connect() as conn:
            df_est = pd.read_sql("SELECT * FROM estaciones", conn)
            df_lec = pd.read_sql('SELECT "FECHA", "PRECIP", "TMAX" FROM lecturas', conn)

        if df_est.empty or df_lec.empty: return

        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=df_est, x='ESTADO', hue='ESTADO', palette='Blues_d', order=df_est['ESTADO'].value_counts().index, legend=False)
        plt.title('Distribución de Estaciones Meteorológicas Crudas (CONAGUA)', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Estado de la República', fontsize=12)
        plt.ylabel('Cantidad de Estaciones Físicas', fontsize=12)
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=12, fontweight='bold')
        plt.savefig(os.path.join(self.reports_folder, "01_Estaciones_por_Estado.png"), dpi=300, bbox_inches='tight')
        plt.close()

        df_lec['AÑO'] = pd.to_datetime(df_lec['FECHA']).dt.year
        df_anual = df_lec.groupby('AÑO').agg({'TMAX': 'count', 'PRECIP': 'count'}).reset_index()
        df_anual = df_anual[df_anual['AÑO'] >= 1980]
        
        plt.figure(figsize=(14, 6))
        plt.plot(df_anual['AÑO'], df_anual['TMAX'], marker='o', color='#b30000', linewidth=2, label='Temperatura')
        plt.plot(df_anual['AÑO'], df_anual['PRECIP'], marker='s', color='#00509d', linewidth=2, label='Precipitación')
        plt.title('Disponibilidad Histórica de Lecturas Terrestres (1980 - Actualidad)', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Año de Observación', fontsize=12)
        plt.ylabel('Volumen Total de Lecturas Diarias', fontsize=12)
        plt.legend(fontsize=11, frameon=True, facecolor='white', edgecolor='black')
        plt.savefig(os.path.join(self.reports_folder, "02_Volumen_Historico_Lecturas.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def ejecutar(self):
        """Orquestador principal que corre todas las etapas en el orden correcto."""
        self.configurar_bd()
        self.extraer_datos_smn()
        self.etl_incremental()
        self.generar_reporte_metodologico()

if __name__ == "__main__":
    pipeline = ConaguaPipeline()
    pipeline.ejecutar()