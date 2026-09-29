import os
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import box
import fiona
import warnings

fiona.drvsupport.supported_drivers['KML'] = 'rw'
fiona.drvsupport.supported_drivers['kml'] = 'rw'
warnings.filterwarnings("ignore")

class InegiGeoSpatialPipeline:
    """
    Pipeline MLOps para el procesamiento, recorte y optimización de 
    capas vectoriales a nivel nacional (INEGI) hacia la región de estudio.
    """
    
    def __init__(self):
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.ruta_edafo_nacional = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "edafologia", "conjunto_de_datos", "cdv_edaf_esc_250k_serie II_cont_nac.shp"))
        self.ruta_geo_nacional = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "geologia", "Geologia_SGM.kml"))
        
        self.dir_salida = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "02_Procesados", "capas_inegi"))
        self.reports_folder = os.path.abspath(os.path.join(self.dir_base, "..", "..", "Datos", "01_Crudos", "reportes_metodologia"))
        
        os.makedirs(self.dir_salida, exist_ok=True)
        os.makedirs(self.reports_folder, exist_ok=True)
        
        self.bbox_wgs84 = box(-92.5, 17.8, -86.7, 21.6)
        self.gdf_caja = gpd.GeoDataFrame(geometry=[self.bbox_wgs84], crs="EPSG:4326")
        
        self.ruta_edafo_out = os.path.join(self.dir_salida, "edafologia_peninsula.geojson")
        self.ruta_geo_out = os.path.join(self.dir_salida, "geologia_peninsula.geojson")

    def procesar_edafologia(self):
        print("\n🌱 [ETAPA 1] Procesando Edafología Nacional...")
        if not os.path.exists(self.ruta_edafo_nacional): return False

        try:
            muestra = gpd.read_file(self.ruta_edafo_nacional, rows=1)
            crs_nativo = muestra.crs if muestra.crs is not None else "EPSG:6372"
            
            gdf_caja_proy = self.gdf_caja.to_crs(crs_nativo)
            bbox_metros = tuple(gdf_caja_proy.total_bounds)

            gdf_edafo = gpd.read_file(self.ruta_edafo_nacional, bbox=bbox_metros)
            if gdf_edafo.crs is None: gdf_edafo.set_crs(crs_nativo, inplace=True)

            gdf_edafo = gdf_edafo.to_crs("EPSG:4326")
            gdf_edafo.to_file(self.ruta_edafo_out, driver="GeoJSON")
            print(f"   ✔️ Extraídos {len(gdf_edafo)} polígonos.")
            return True
        except Exception as e:
            return False

    def procesar_geologia(self):
        print("\n🪨 [ETAPA 2] Procesando Geología (KML)...")
        if not os.path.exists(self.ruta_geo_nacional): return False

        try:
            gdf_geo = gpd.read_file(self.ruta_geo_nacional, driver='KML')
            if gdf_geo.crs is None: gdf_geo.set_crs("EPSG:4326", inplace=True)
            
            gdf_geo_recortado = gdf_geo.cx[-92.5:-86.7, 17.8:21.6]
            gdf_geo_recortado.to_file(self.ruta_geo_out, driver="GeoJSON")
            print(f"   ✔️ Extraídos {len(gdf_geo_recortado)} polígonos.")
            return True
        except Exception as e:
            return False

    def generar_reporte_metodologico(self):
        print("\n📊 [ETAPA 3] Generando Gráfica de Área de Estudio (Bounding Box)...")
        gdf_caja_mercator = self.gdf_caja.to_crs(epsg=3857)
        
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 8))
        
        gdf_caja_mercator.plot(ax=ax, facecolor='none', edgecolor='#b30000', linewidth=3, linestyle='--')
        
        try:
            # MAPA DE ESRI GRAY CANVAS: Sin bloqueos y altamente profesional
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldGrayCanvas)
        except Exception:
            print("   ⚠️ No se pudo descargar el mapa base.")
            
        ax.set_title("Delimitación del Área de Estudio (Península de Yucatán)", fontsize=14, fontweight='bold', pad=15)
        ax.set_axis_off()
        
        texto_bbox = "BBox (WGS84):\nMin Lon: -92.5\nMax Lon: -86.7\nMin Lat: 17.8\nMax Lat: 21.6"
        plt.figtext(0.15, 0.15, texto_bbox, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'), fontsize=10)

        ruta_grafica = os.path.join(self.reports_folder, "06_Area_de_Estudio_Bounding_Box.png")
        plt.savefig(ruta_grafica, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Mapa exportado a: {self.reports_folder}")

    def ejecutar(self):
        print("🗺️ Iniciando integración Geo-Edafológica de INEGI...")
        self.procesar_edafologia()
        self.procesar_geologia()
        self.generar_reporte_metodologico()
        print("\n✅ Proceso completado.")

if __name__ == "__main__":
    pipeline = InegiGeoSpatialPipeline()
    pipeline.ejecutar()