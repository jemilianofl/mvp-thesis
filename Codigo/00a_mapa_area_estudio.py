import os
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as cx
from shapely.geometry import Point
import warnings

warnings.filterwarnings("ignore")

class MapaAreaEstudio:
    def __init__(self):
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.ruta_poligonos = os.path.abspath(os.path.join(self.dir_base, "..", "Datos", "01_Crudos", "poligonos_maestros.geojson"))
        self.reports_folder = os.path.abspath(os.path.join(self.dir_base, "..", "Datos", "01_Crudos", "reportes_metodologia"))
        os.makedirs(self.reports_folder, exist_ok=True)

    def generar_mapas(self):
        print("🗺️ Generando Mapas de Área de Estudio (Alta Visibilidad)...")
        
        if not os.path.exists(self.ruta_poligonos):
            print("❌ No se encontró el archivo de polígonos.")
            return

        poligonos = gpd.read_file(self.ruta_poligonos)
        poligonos = poligonos.to_crs(epsg=3857)
        
        centroides = poligonos.copy()
        centroides.geometry = centroides.geometry.centroid

        plt.rcParams.update(plt.rcParamsDefault)
        plt.style.use('default')
        sns.set_theme(style="white")

        import matplotlib.lines as mlines
        marcador_punto = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                                      markersize=10, markeredgecolor='white', markeredgewidth=1.5, 
                                      label='Zonas de Muestreo (Centroide)')

        # =========================================================
        # 1. MAPA GENERAL 
        # =========================================================
        fig1, ax1 = plt.subplots(figsize=(10, 10), facecolor='white')
        
        centroides.plot(ax=ax1, color='red', marker='o', markersize=80, edgecolor='white', linewidth=1.5, zorder=5)
        
        # ORDEN CORRECTO: Primero expandimos la ventana a toda la Península
        limites_peninsula = gpd.GeoSeries([Point(-91.5, 17.5), Point(-86.5, 22.0)], crs="EPSG:4326").to_crs(epsg=3857).total_bounds
        ax1.set_xlim([limites_peninsula[0], limites_peninsula[2]])
        ax1.set_ylim([limites_peninsula[1], limites_peninsula[3]])
        
        # Y LUEGO descargamos el mapa satelital para que llene todo el cuadro
        cx.add_basemap(ax1, source=cx.providers.Esri.WorldImagery)
        
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title('Área de Estudio: Contexto Regional en la Península de Yucatán', fontsize=15, fontweight='bold', pad=15)
        ax1.legend(handles=[marcador_punto], loc='lower right', frameon=True, facecolor='white', edgecolor='black')

        ruta_general = os.path.join(self.reports_folder, "00a_Area_Estudio_General.png")
        plt.savefig(ruta_general, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig1)

        # =========================================================
        # 2. MAPA DE DETALLE (Zoom)
        # =========================================================
        fig2, ax2 = plt.subplots(figsize=(10, 10), facecolor='white')
        
        poligonos.plot(ax=ax2, facecolor='none', edgecolor='#00ff00', linewidth=3, zorder=5)
        centroides.plot(ax=ax2, color='red', marker='+', markersize=100, linewidth=2, zorder=6)
        
        # ORDEN CORRECTO: Primero el zoom
        bounds = poligonos.total_bounds
        margen_x = (bounds[2] - bounds[0]) * 0.20 
        margen_y = (bounds[3] - bounds[1]) * 0.20
        ax2.set_xlim([bounds[0] - margen_x, bounds[2] + margen_x])
        ax2.set_ylim([bounds[1] - margen_y, bounds[3] + margen_y])
        
        # LUEGO el mapa base
        cx.add_basemap(ax2, source=cx.providers.Esri.WorldImagery)
        
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title('Zonas de Muestreo: Distribución de Polígonos de Entrenamiento', fontsize=15, fontweight='bold', pad=15)
        
        marcador_borde = mlines.Line2D([], [], color='none', marker='s', linestyle='None',
                                      markeredgecolor='#00ff00', markeredgewidth=2, markersize=12,
                                      label='Perímetro del Polígono')
                                      
        ax2.legend(handles=[marcador_punto, marcador_borde], loc='lower right', frameon=True, facecolor='white', edgecolor='black')

        ruta_zoom = os.path.join(self.reports_folder, "00b_Area_Estudio_Zoom.png")
        plt.savefig(ruta_zoom, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig2)

        print(f"   ✅ Gráficas satelitales 00a y 00b exportadas en: {self.reports_folder}")

if __name__ == "__main__":
    mapa = MapaAreaEstudio()
    mapa.generar_mapas()