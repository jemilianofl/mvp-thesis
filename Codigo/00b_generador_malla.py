import os
import math
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import warnings

warnings.filterwarnings("ignore")

class GeneradorHexagonosPuros:
    def __init__(self):
        self.dir_base = os.path.dirname(os.path.abspath(__file__))
        self.dir_salida = os.path.abspath(os.path.join(self.dir_base, "Datos", "01_Crudos"))
        self.ruta_salida = os.path.join(self.dir_salida, "malla_regional_yucatan.geojson")
        os.makedirs(self.dir_salida, exist_ok=True)
        
        self.xmin, self.ymin = -92.5, 17.5
        self.xmax, self.ymax = -86.5, 21.6
        self.radio = 0.035 # Tamaño del hexágono (~3.5 km)

    def crear_hexagono(self, x_centro, y_centro, radio):
        vertices = []
        for i in range(6):
            angulo_deg = 60 * i
            angulo_rad = math.pi / 180 * angulo_deg
            x = x_centro + radio * math.cos(angulo_rad)
            y = y_centro + radio * math.sin(angulo_rad)
            vertices.append((x, y))
        return Polygon(vertices)

    def generar_malla(self):
        print("Generando auténtica Malla Hexagonal (Trigonometría Pura)...")
        
        # Distancias entre centros para un teselado perfecto
        dx = 3/2 * self.radio
        dy = math.sqrt(3) * self.radio
        
        cols = int((self.xmax - self.xmin) / dx)
        rows = int((self.ymax - self.ymin) / dy)
        
        geometrias = []
        ids = []
        contador = 0
        
        print("   ► Dibujando panal hexagonal sobre la Península...")
        for col in range(cols):
            for row in range(rows):
                x = self.xmin + col * dx
                # Desplazamiento vertical para intercalar los hexágonos
                offset_y = (dy / 2) if col % 2 == 1 else 0
                y = self.ymin + row * dy + offset_y
                
                geometrias.append(self.crear_hexagono(x, y, self.radio))
                ids.append(f"HEX_{contador}")
                contador += 1
                
        gdf_hex = gpd.GeoDataFrame({'ID_POLIGONO': ids, 'geometry': geometrias}, crs="EPSG:4326")
        
        # Recortar rápido a los límites aproximados (evitar exceso en el mar)
        print(f"   ► Se generaron {len(gdf_hex)} hexágonos.")
        gdf_hex.to_file(self.ruta_salida, driver="GeoJSON")
        print(f"✅ Malla exportada a: {self.ruta_salida}")

if __name__ == "__main__":
    GeneradorHexagonosPuros().generar_malla()